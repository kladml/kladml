
import pytest
from unittest.mock import MagicMock, patch
import torch
from kladml.training.trainer import UniversalTrainer

class TestTrainerEdgeCases:
    """Tests for UniversalTrainer edge cases and device logic."""
    
    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_setup_device_cpu_fallback(self, mock_mps, mock_cuda):
        """Test fallback to CPU when no accelerators available."""
        trainer = UniversalTrainer(accelerator="auto")
        assert trainer.device == torch.device("cpu")
        
    @patch("torch.cuda.is_available", return_value=True)
    def test_setup_device_auto_cuda(self, mock_cuda):
        """Test auto detection prefers CUDA."""
        trainer = UniversalTrainer(accelerator="auto")
        assert trainer.device == torch.device("cuda:0")

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_setup_device_auto_mps(self, mock_mps, mock_cuda):
        """Test auto detection uses MPS if CUDA missing."""
        trainer = UniversalTrainer(accelerator="auto")
        assert trainer.device == torch.device("mps")

    @patch("torch.cuda.is_available", return_value=False)
    def test_setup_device_cuda_unavailable_warning(self, mock_cuda):
        """Test warning when explicit CUDA requested but unavailable."""
        with patch("loguru.logger.warning") as mock_warn:
            trainer = UniversalTrainer(accelerator="cuda")
            assert trainer.device == torch.device("cpu")
            mock_warn.assert_called_with("CUDA requested but not available. Falling back to CPU.")

    def test_configure_optimizers_validation(self):
        """Test error when model missing configure_optimizers."""
        trainer = UniversalTrainer(max_epochs=1)
        model = MagicMock(spec=torch.nn.Module)
        del model.configure_optimizers # ensure it doesn't exist
        
        with pytest.raises(AttributeError, match="configure_optimizers"):
            trainer.fit(model, [])

    def test_move_batch_recursive(self):
        """Test recursive batch moving."""
        trainer = UniversalTrainer(accelerator="cpu")
        
        t1 = torch.tensor([1])
        batch_list = [t1, t1]
        batch_dict = {"a": t1, "b": [t1]}
        
        # We Mock .to() to verify call
        with patch.object(torch.Tensor, 'to', return_value="moved") as mock_to:
            res_list = trainer._move_batch_to_device(batch_list)
            assert res_list == ["moved", "moved"]
            
            res_dict = trainer._move_batch_to_device(batch_dict)
            assert res_dict["a"] == "moved"
            assert res_dict["b"] == ["moved"]
