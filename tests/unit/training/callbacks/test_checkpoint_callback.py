
import pytest
from unittest.mock import MagicMock
from kladml.training.callbacks.checkpoint import CheckpointCallback

class TestCheckpointCallback:
    """Tests for CheckpointCallback."""
    
    def test_on_epoch_end_trigger(self):
        """Test that callback triggers manager.save_checkpoint."""
        mock_manager = MagicMock()
        cb = CheckpointCallback(mock_manager)
        
        # Mock trainer
        mock_trainer = MagicMock()
        mock_trainer.model = "my_model"
        mock_trainer.optimizer = "my_opt"
        cb.set_trainer(mock_trainer)
        
        # Trigger
        metrics = {"val_loss": 0.5}
        cb.on_epoch_end(epoch=1, logs=metrics)
        
        mock_manager.save_checkpoint.assert_called_with(
            model="my_model",
            optimizer="my_opt",
            epoch=1,
            metrics=metrics,
            scaler=None 
        )

    def test_no_trainer_no_trigger(self):
        """Test safe fail if trainer not attached."""
        mock_manager = MagicMock()
        cb = CheckpointCallback(mock_manager)
        # No set_trainer called
        
        cb.on_epoch_end(1, {})
        mock_manager.save_checkpoint.assert_not_called()
