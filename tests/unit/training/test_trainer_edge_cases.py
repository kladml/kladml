
import pytest
from unittest.mock import MagicMock, patch
import torch
from kladml.training.trainer import UniversalTrainer

# Skip tests that require MLflow/Accelerate integration
pytestmark = pytest.mark.skip(reason="Trainer requires MLflow integration - run in isolation")

class TestTrainerEdgeCases:
    """Tests for UniversalTrainer edge cases and logic."""
    
    def test_accelerator_config_mapping(self):
        """Test that KladML config maps to Accelerator args correctly."""
        with patch("kladml.training.trainer.Accelerator") as mock_acc:
            # Case 1: CPU request
            trainer = UniversalTrainer(accelerator="cpu")
            mock_acc.assert_called_with(
                mixed_precision="no",
                gradient_accumulation_steps=1,
                cpu=True,
                log_with="mlflow"
            )

    def test_configure_optimizers_validation(self):
        """Test error when model missing configure_optimizers."""
        trainer = UniversalTrainer(max_epochs=1, accelerator="cpu")
        model = MagicMock(spec=torch.nn.Module)
        del model.configure_optimizers # ensure it doesn't exist
        
        with pytest.raises(AttributeError, match="configure_optimizers"):
            trainer.fit(model, [])

    def test_explicit_cpu_override(self):
        """Test that accelerator='cpu' forces cpu=True in Accelerator."""
        with patch("kladml.training.trainer.Accelerator") as mock_acc:
             trainer = UniversalTrainer(accelerator="cpu")
             _, kwargs = mock_acc.call_args
             assert kwargs["cpu"] is True

    def test_auto_accelerator(self):
        """Test that accelerator='auto' calls Accelerator with cpu=False (default)."""
        with patch("kladml.training.trainer.Accelerator") as mock_acc:
             trainer = UniversalTrainer(accelerator="auto")
             _, kwargs = mock_acc.call_args
             assert kwargs["cpu"] is False
