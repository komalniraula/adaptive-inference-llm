import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class GPT2WithEarlyExit(torch.nn.Module):

    def __init__(self, model_name, strategy, tokenizer, max_new_tokens=20):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.num_layers = len(self.model.transformer.h)
        self.max_new_tokens = max_new_tokens # Can be implemented in future (havn't implemented now)

    @torch.no_grad()
    def generate_with_early_exit(self, text):
        self.strategy.reset()

        inputs = self.tokenizer(text, return_tensors="pt")
        hidden_states = self.model.transformer.wte(inputs["input_ids"])
        past = None

        # Layer by layer manual forward pass
        for layer_idx, block in enumerate(self.model.transformer.h):

            outputs = block(hidden_states, layer_past=past, use_cache=False)
            hidden_states = outputs[0]

            # Compute logits for confidence
            logits = self.model.lm_head(hidden_states[:, -1, :])
            probs = torch.softmax(logits, dim=-1)
            confidence = probs.max().item()

            if self.strategy.should_exit(confidence, layer_idx):
                # Exit now = generate one token greedily
                next_token = torch.argmax(logits, dim=-1)
                decoded = self.tokenizer.decode(next_token)
                return decoded, layer_idx + 1

        # If no exit = use last layer output
        logits = self.model.lm_head(hidden_states[:, -1, :])
        next_token = torch.argmax(logits, dim=-1)
        decoded = self.tokenizer.decode(next_token)
        return decoded, self.num_layers
