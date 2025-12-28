from .base_strategy import EarlyExitStrategy

class ConfidenceExit(EarlyExitStrategy):
    """
    Early exit when confidence >= threshold
    and layer_idx is in allowed_layers.
    """

    def __init__(self, threshold, allowed_layers):
        self.threshold = threshold
        self.allowed_layers = set(allowed_layers)

    def reset(self):
        pass

    def should_exit(self, confidence, layer_idx):
        if layer_idx in self.allowed_layers and confidence >= self.threshold:
            return True
        return False
