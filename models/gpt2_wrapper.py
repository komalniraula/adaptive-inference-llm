import torch
from transformers import AutoModelForCausalLM
import numpy as np


class GPT2WithEarlyExit(torch.nn.Module):
    def __init__(self, model_name, strategy, tokenizer, use_kv=False, max_new_tokens=32):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.num_layers = len(self.model.transformer.h)
        self.use_kv = use_kv
        self.max_new_tokens = max_new_tokens

        # VERBALIZERS
        self.verbalizers = {
            "sst2": {0: "negative", 1: "positive"},
            "agnews": {0: "world", 1: "sports", 2: "business", 3: "tech"}
        }

        # Precompute token IDs
        self.verbalizer_token_ids = {}
        for dataset, vocab in self.verbalizers.items():
            ids = {}
            for cls, word in vocab.items():
                ids[cls] = self.tokenizer.encode(" " + word)[0]
            self.verbalizer_token_ids[dataset] = ids

    # CLASSIFICATION EARLY EXIT (no KV needed)
    @torch.no_grad()
    def classify_with_early_exit(self, text, dataset_name):

        self.strategy.reset()
        token_ids = self.verbalizer_token_ids[dataset_name]

        inputs = self.tokenizer(text, return_tensors="pt")
        hidden_states = self.model.transformer.wte(inputs["input_ids"])

        for layer_idx, block in enumerate(self.model.transformer.h):

            outputs = block(hidden_states, use_cache=False)
            hidden_states = outputs[0]

            logits = self.model.lm_head(hidden_states[:, -1, :])
            probs = torch.softmax(logits, dim=-1)[0]
            class_probs = {cls: probs[tok].item() for cls, tok in token_ids.items()}

            pred_class = max(class_probs, key=class_probs.get)
            confidence = class_probs[pred_class]

            if self.strategy.should_exit(confidence, layer_idx):
                return pred_class, layer_idx + 1

        return pred_class, self.num_layers

    # GENERATION — EARLY EXIT (supports KV or no-KV paths)
    @torch.no_grad()
    def generate_with_early_exit(self, text):
        """
        use_kv=False → slow, full recomputation (no cache)
        use_kv=True  → fast CALM-style KV-cached early exit
        """
        self.strategy.reset()
    
        # Encode
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        generated = input_ids.clone()
    
        eos_token = self.tokenizer.eos_token_id
        layers_used_tokens = []
    
        # NO KV (full recomputation)
        if not self.use_kv:
            for _ in range(self.max_new_tokens):
    
                hidden_states = self.model.transformer.wte(generated)
                exited = False
    
                for layer_idx, block in enumerate(self.model.transformer.h):
                    outputs = block(hidden_states, use_cache=False)
                    hidden_states = outputs[0]
    
                    logits = self.model.lm_head(hidden_states[:, -1, :])
                    probs = torch.softmax(logits, dim=-1)
                    confidence = probs.max().item()
    
                    if self.strategy.should_exit(confidence, layer_idx):
                        next_token = torch.argmax(logits, dim=-1)
                        layers_used_tokens.append(layer_idx + 1)
                        exited = True
                        break
    
                if not exited:
                    next_token = torch.argmax(logits, dim=-1)
                    layers_used_tokens.append(self.num_layers)
    
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
    
                if eos_token is not None and next_token.item() == eos_token:
                    break
    
            final_ids = generated[0, input_ids.size(1):]
            return self.tokenizer.decode(final_ids, skip_special_tokens=True), np.mean(layers_used_tokens)
    

        # WITH KV CACHE (copy KV)    
        # Initial forward pass for prompt → get KV and last hidden
        outputs = self.model(input_ids, use_cache=True, output_hidden_states=True)
        past = list(outputs.past_key_values)          # list of length num_layers
        last_hidden = outputs.hidden_states[-1][:, -1:, :]  # (1, 1, d)
    
        for _ in range(self.max_new_tokens):
    
            exited = False
            hidden_states = last_hidden
            new_past = [None] * self.num_layers
    
            for layer_idx, block in enumerate(self.model.transformer.h):
                layer_past = past[layer_idx]
    
                out = block(
                    hidden_states,
                    layer_past=layer_past,
                    use_cache=True,
                )
    
                # out can be (hidden_states,) or (hidden_states, present, ...) depending on version
                hidden_states = out[0]
    
                if len(out) > 1 and out[1] is not None:
                    # normal case: KV returned
                    new_past[layer_idx] = out[1]
                else:
                    # fallback: keep previous KV for this layer
                    new_past[layer_idx] = past[layer_idx]
    
                # Compute logits for last token at this layer
                logits = self.model.lm_head(hidden_states[:, -1, :])
                probs = torch.softmax(logits, dim=-1)
                confidence = probs.max().item()
    
                if self.strategy.should_exit(confidence, layer_idx):
                    next_token = torch.argmax(logits, dim=-1)
                    layers_used_tokens.append(layer_idx + 1)
                    exited = True
    
                    # carry over KV for deeper layers untouched
                    for tl in range(layer_idx + 1, self.num_layers):
                        new_past[tl] = past[tl]
    
                    break
    
            if not exited:
                # No early exit → used final layer
                next_token = torch.argmax(logits, dim=-1)
                layers_used_tokens.append(self.num_layers)
                # new_past is already filled for all layers in that case
    
            # Update KV cache and last hidden for next token
            past = new_past
            last_hidden = self.model.transformer.wte(next_token.unsqueeze(0))
    
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
    
            if eos_token is not None and next_token.item() == eos_token:
                break
    
        final_ids = generated[0, input_ids.size(1):]
        return self.tokenizer.decode(final_ids, skip_special_tokens=True), np.mean(layers_used_tokens)