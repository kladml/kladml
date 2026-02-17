
import pytest
from unittest.mock import MagicMock, patch
import torch
from kladml.training.trainer import UniversalTrainer
from kladml.config.schema import TrainingConfig

# Skip tests that require MLflow/Accelerate integration
pytestmark = pytest.mark.skip(reason="Trainer requires MLflow integration - run in isolation")

def test_trainer_compile_config():
    """Test that compile config triggers torch.compile."""
    config = TrainingConfig(
        max_epochs=1,
        compile=True
    )
    
    trainer = UniversalTrainer(config=config)
    
    # Mock Accelerator to avoid real env issues
    trainer.accelerator = MagicMock()
    trainer.accelerator.device.type = "cpu" # Safe
    trainer.accelerator.prepare.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
    
    # Mock Model
    model = torch.nn.Linear(10, 10)
    model.configure_optimizers = MagicMock(return_value=torch.optim.SGD(model.parameters(), lr=0.01))
    
    # Mock torch.compile
    with patch("torch.compile", return_value=model) as mock_compile:
        trainer.fit(model, train_dataloaders=[])
        
        # Verify compile was called
        mock_compile.assert_called_once()

def test_trainer_compile_mps_skip():
    """Test that compile is skipped on MPS with warning."""
    config = TrainingConfig(
        max_epochs=1,
        compile=True
    )
    
    trainer = UniversalTrainer(config=config)
    trainer.accelerator = MagicMock()
    trainer.accelerator.device.type = "mps" # MPS
    trainer.accelerator.prepare.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
    
    model = torch.nn.Linear(10, 10)
    model.configure_optimizers = MagicMock(return_value=torch.optim.SGD(model.parameters(), lr=0.01))
    
    with patch("torch.compile") as mock_compile:
        # Should NOT call compile
        trainer.fit(model, train_dataloaders=[])
        mock_compile.assert_not_called()
