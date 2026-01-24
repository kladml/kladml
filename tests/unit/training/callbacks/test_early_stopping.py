
import pytest
from kladml.training.callbacks.early_stopping import EarlyStoppingCallback

class TestEarlyStoppingCallback:
    """Tests for EarlyStoppingCallback."""
    
    def test_initialization(self):
        """Test default initialization."""
        es = EarlyStoppingCallback(patience=5)
        assert es.patience == 5
        assert es.mode == "min"
        assert es.min_delta == 0.0
        assert es.best_value is None
        assert es.counter == 0

    def test_min_mode_logic(self):
        """Test improvement logic in minimization mode (e.g. loss)."""
        es = EarlyStoppingCallback(patience=2, mode="min")
        
        # Initial improvement
        es.on_epoch_end(1, {"val_loss": 1.0})
        assert es.best_value == 1.0
        assert es.counter == 0
        
        # Improvement
        es.on_epoch_end(2, {"val_loss": 0.9})
        assert es.best_value == 0.9
        assert es.counter == 0
        
        # No improvement (1st time)
        es.on_epoch_end(3, {"val_loss": 0.9})
        assert es.counter == 1
        assert not es.should_stop
        
        # No improvement (2nd time - patience limit)
        es.on_epoch_end(4, {"val_loss": 0.95})
        assert es.counter == 2
        assert es.should_stop

    def test_max_mode_logic(self):
        """Test improvement logic in maximization mode (e.g. accuracy)."""
        es = EarlyStoppingCallback(patience=2, mode="max", metric="acc")
        
        # Initial
        es.on_epoch_end(1, {"acc": 0.80})
        assert es.best_value == 0.80
        
        # Improvement
        es.on_epoch_end(2, {"acc": 0.85})
        assert es.best_value == 0.85
        assert es.counter == 0
        
        # No improvement
        es.on_epoch_end(3, {"acc": 0.84})
        assert es.counter == 1
        assert not es.should_stop

    def test_min_delta(self):
        """Test minimum delta requirement."""
        # Must improve by at least 0.1
        es = EarlyStoppingCallback(patience=1, min_delta=0.1)
        
        es.on_epoch_end(1, {"val_loss": 1.0})
        
        # Improvement of 0.05 is NOT enough
        es.on_epoch_end(2, {"val_loss": 0.95})
        assert es.counter == 1
        
        # Improvement of 0.2 IS enough (relative to best 1.0)
        # Note: logic usually compares to best_score
        es.on_epoch_end(3, {"val_loss": 0.7})
        assert es.counter == 0
        assert es.best_value == 0.7
