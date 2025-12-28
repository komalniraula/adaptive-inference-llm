class EarlyExitStrategy:
    """Base class for all early exit strategies."""

    def reset(self):
        """Reset internal state before each new sample."""
        pass

    def should_exit(self, confidence, layer_idx):
        """
        Returns (exit_now: bool)
        """
        raise NotImplementedError
