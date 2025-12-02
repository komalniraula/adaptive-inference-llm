import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np


class GPT2WithEarlyExit(torch.nn.Module):
    def __init__(self, model_name, strategy, tokenizer, max_new_tokens=32):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.num_layers = len(self.model.transformer.h)
        self.max_new_tokens = max_new_tokens
        
        # VERBALIZERS FOR CLASSIFICATION
        # You can override these externally for custom datasets.
        
        self.verbalizers = {
            "sst2": {                    # sentiment
                0: "negative",
                1: "positive",
            },
            "agnews": {                  # 4 news categories
                0: "world",
                1: "sports",
                2: "business",
                3: "tech",
            }
        }

        # Precompute token IDs for verbalizers
        self.verbalizer_token_ids = {}
        for dataset, mapping in self.verbalizers.items():
            tok_ids = {}
            for cls, word in mapping.items():
                tok_ids[cls] = self.tokenizer.encode(" " + word)[0]
            self.verbalizer_token_ids[dataset] = tok_ids

    # CLASSIFICATION EARLY EXIT 
    @torch.no_grad()
    def classify_with_early_exit(self, text, dataset_name):
        """
        dataset_name: 'sst2' or 'agnews'
        Returns: predicted class index
        """
        self.strategy.reset()
        token_ids = self.verbalizer_token_ids[dataset_name]

        inputs = self.tokenizer(text, return_tensors="pt")
        hidden_states = self.model.transformer.wte(inputs["input_ids"])
        past_key_values = None
        
        for layer_idx, block in enumerate(self.model.transformer.h):

            outputs = block(hidden_states, layer_past=past_key_values, use_cache=False)
            hidden_states = outputs[0]

            # Compute logits for last token
            logits = self.model.lm_head(hidden_states[:, -1, :])
            probs = torch.softmax(logits, dim=-1)[0]

            # Compute class probabilities only for verbalizer tokens
            class_probs = {cls: probs[t_id].item() for cls, t_id in token_ids.items()}
            pred_class = max(class_probs, key=class_probs.get)
            confidence = class_probs[pred_class]

            if self.strategy.should_exit(confidence, layer_idx):
                return pred_class, layer_idx+1

        # If no exit, use final layer
        return pred_class, self.num_layers


    # GENERATION EARLY EXIT 
    @torch.no_grad()
    def generate_with_early_exit(self, text):
        """
        Single-token generation early exit.
        Summarization, translation, QA.
        """
        self.strategy.reset()

        inputs = self.tokenizer(text, return_tensors="pt")
        hidden_states = self.model.transformer.wte(inputs["input_ids"])
        past_key_values = None

        for layer_idx, block in enumerate(self.model.transformer.h):
            outputs = block(hidden_states, use_cache=False)
            hidden_states = outputs[0]

            logits = self.model.lm_head(hidden_states[:, -1, :])
            probs = torch.softmax(logits, dim=-1)
            confidence = probs.max().item()

            if self.strategy.should_exit(confidence, layer_idx):
                next_token = torch.argmax(logits, dim=-1)
                decoded = self.tokenizer.decode(next_token)
                return decoded, layer_idx+1

        next_token = torch.argmax(logits, dim=-1)
        decoded = self.tokenizer.decode(next_token)
        return decoded, self.num_layers