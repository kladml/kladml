
import pytest
from kladml.training.callbacks.metrics import MetricsCallback

class TestMetricsCallback:
    """Tests for MetricsCallback."""
    
    def test_history_accumulation(self):
        """Test that metrics are accumulated in history."""
        cb = MetricsCallback()
        assert cb.history == {}
        
        # Epoch 1
        cb.on_epoch_end(1, {"loss": 0.5, "acc": 0.8})
        assert cb.history["loss"] == [0.5]
        assert cb.history["acc"] == [0.8]
        
        # Epoch 2
        cb.on_epoch_end(2, {"loss": 0.4, "acc": 0.82})
        assert cb.history["loss"] == [0.5, 0.4]
        assert cb.history["acc"] == [0.8, 0.82]

    def test_ignore_non_scalar(self):
        """Test that non-scalar values are ignored."""
        cb = MetricsCallback()
        
        # Dictionary or string should be ignored
        cb.on_epoch_end(1, {
            "loss": 0.5, 
            "metadata": {"step": 10}, 
            "string_val": "test"
        })
        
        assert "loss" in cb.history
        assert "metadata" not in cb.history
        assert "string_val" not in cb.history
