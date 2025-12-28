from .base_strategy import EarlyExitStrategy

class ContinuousConfidenceExit(EarlyExitStrategy):
    """
    Require high confidence for K consecutive layers
    before exiting.
    """

    def __init__(self, threshold, required_consecutive, allowed_layers=None):
        self.threshold = threshold
        self.required_consecutive = required_consecutive
        self.allowed_layers = (
            set(allowed_layers) if allowed_layers else None
        )
        self.counter = 0

    def reset(self):
        self.counter = 0

    def should_exit(self, confidence, layer_idx):
        if self.allowed_layers and layer_idx not in self.allowed_layers:
            self.counter = 0
            return False

        if confidence >= self.threshold:
            self.counter += 1
        else:
            self.counter = 0

        return self.counter >= self.required_consecutive
