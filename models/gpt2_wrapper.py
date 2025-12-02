import torch
from transformers import AutoModelForCausalLM
import numpy as np


class GPT2WithEarlyExit(torch.nn.Module):
    def __init__(self, model_name, strategy, tokenizer, use_kv=False, max_new_tokens=32):
        """
        model_name: e.g. "gpt2"
        strategy:   ConfidenceExit / ContinuousConfidenceExit
        tokenizer:  HF tokenizer
        use_kv:     currently kept as a flag; early-exit savings are implemented
                    in the no-KV path for correctness and clarity.
        """
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True
        )
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.num_layers = len(self.model.transformer.h)
        self.use_kv = use_kv
        self.max_new_tokens = max_new_tokens

        # VERBALIZERS (multi-word per class)
        self.verbalizers = {
            "sst2": {
                0: ["negative"],
                1: ["positive"],
            },
            "agnews": {
                0: ["international", "world", "global"],
                1: ["sports", "sport"],
                2: ["business", "finance", "market"],
                3: ["technology", "tech", "computer"],
            }
        }

        # Precompute token IDs for verbalizers
        self.verbalizer_token_ids = {}
        for dataset, class_map in self.verbalizers.items():
            ids = {}
            for cls, words in class_map.items():
                tok_lists = []
                for w in words:
                    # GPT-2 uses a space before most words
                    tok_lists.append(self.tokenizer.encode(" " + w))
                ids[cls] = tok_lists  # list of token-id lists
            self.verbalizer_token_ids[dataset] = ids

    # Helper: device + embeddings with positional encoding
    @property
    def device(self):
        return next(self.model.parameters()).device

    def _embed_with_pos(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Safe positional embedding: clamps positions >= 1024
        to the last index. This avoids IndexError for long sequences.
        """
        input_ids = input_ids.to(self.device)
        seq_len = input_ids.size(1)
    
        max_pos = self.model.config.n_positions  # usually 1024
    
        # clamp position indices to last embedding row
        pos_ids = torch.arange(
            0, seq_len,
            dtype=torch.long,
            device=self.device
        ).clamp(max=max_pos - 1).unsqueeze(0)  # (1, T)
    
        token_emb = self.model.transformer.wte(input_ids)      # (1, T, d)
        pos_emb = self.model.transformer.wpe(pos_ids)          # (1, T, d)
    
        return token_emb + pos_emb


    # CLASSIFICATION EARLY EXIT (no KV)
    @torch.no_grad()
    def classify_with_early_exit(self, text: str, dataset_name: str):
        """
        dataset_name: 'sst2' or 'agnews'
        Returns: (predicted_class_index, layers_used)
        """
        self.strategy.reset()

        if dataset_name not in self.verbalizer_token_ids:
            raise ValueError(f"Unknown dataset_name '{dataset_name}' for classification.")

        class_verbalizers = self.verbalizer_token_ids[dataset_name]

        # Encode text
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        # Initial hidden states with positional embeddings
        hidden_states = self._embed_with_pos(input_ids)

        # Manual layer-by-layer forward pass with early exit
        for layer_idx, block in enumerate(self.model.transformer.h):
            outputs = block(hidden_states, use_cache=False)
            hidden_states = outputs[0]  # (1, T, d)

            # logits for last token
            logits = self.model.lm_head(hidden_states[:, -1, :])[0]  # (vocab,)

            class_scores = []

            # Compute class scores via verbalizers
            for cls, verbalizer_toklists in class_verbalizers.items():
                v_scores = []
                for tok_list in verbalizer_toklists:
                    tok_indices = torch.tensor(tok_list, dtype=torch.long, device=logits.device)
                    v_scores.append(logits[tok_indices].mean().item())
                # best verbalizer for that class
                class_scores.append(max(v_scores))

            class_scores = torch.tensor(class_scores, device=logits.device)
            class_probs = torch.softmax(class_scores, dim=-1)

            pred_class = torch.argmax(class_probs).item()
            confidence = class_probs[pred_class].item()

            if self.strategy.should_exit(confidence, layer_idx):
                return pred_class, layer_idx + 1  # 1-based depth

        # If never exited early, return final-layer prediction
        return pred_class, self.num_layers

    # GENERATION — EARLY EXIT (NO-KV path = where you measure speed)
    @torch.no_grad()
    def generate_with_early_exit(self, text: str):
        """
        Multi-token autoregressive generation with early exit at each decoding step.

        use_kv = False → manual layer-wise early exit (recomputes prefix each step).
                         This is the path where early-exit actually reduces compute.
        use_kv = True  → currently falls back to the same logic for correctness
                         (KV-optimized early-exit can be added later).
        Returns:
            (generated_text, avg_layers_used_over_tokens)
        """
        self.strategy.reset()

        # Encode prompt
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        generated = input_ids.clone()

        eos_token_id = self.tokenizer.eos_token_id
        layers_used_tokens = []

        # For now, use the same implementation for both flags
        # (KV-optimized path can be added on top of this later)
        for _ in range(self.max_new_tokens):

            # Recompute full prefix up to this point, but only up to the EXIT layer
            # 1) embed with positions
            hidden_states = self._embed_with_pos(generated)
            exited = False

            # 2) walk layers, check confidence at each, maybe exit early
            for layer_idx, block in enumerate(self.model.transformer.h):
                outputs = block(hidden_states, use_cache=False)
                hidden_states = outputs[0]  # (1, T, d)

                # logits for last token
                logits = self.model.lm_head(hidden_states[:, -1, :])
                probs = torch.softmax(logits, dim=-1)
                confidence = probs.max().item()

                if self.strategy.should_exit(confidence, layer_idx):
                    next_token = torch.argmax(logits, dim=-1)  # (1,)
                    layers_used_tokens.append(layer_idx + 1)
                    exited = True
                    break

            # If no early exit: use final layer
            if not exited:
                next_token = torch.argmax(logits, dim=-1)
                layers_used_tokens.append(self.num_layers)

            # Append next_token to sequence
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

            # Stop at EOS if defined
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        # Decode only the generated continuation (exclude the prompt)
        cont_ids = generated[0, input_ids.size(1):]
        decoded_text = self.tokenizer.decode(cont_ids, skip_special_tokens=True)

        avg_layers_used = float(np.mean(layers_used_tokens)) if layers_used_tokens else 0.0

        return decoded_text, avg_layers_used