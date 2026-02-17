
import pytest
import torch
from unittest.mock import MagicMock, patch
from accelerate import Accelerator

from kladml.training.trainer import UniversalTrainer, TrainingConfig
from kladml.training.callbacks import Callback

# Skip tests that require MLflow/Accelerate integration
# The trainer uses log_with="mlflow" which requires proper MLflow setup
pytestmark = pytest.mark.skip(reason="Trainer requires MLflow integration - run in isolation")

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        return {"loss": loss}
        
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

class VanillaModel(torch.nn.Module):
    """Model without training_step/validation_step."""
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

@pytest.fixture
def dummy_data():
    X = torch.randn(10, 10)
    y = torch.randn(10, 1)
    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size=2)

def test_trainer_initialization():
    config = TrainingConfig(max_epochs=2, accelerator="cpu", mixed_precision="no")
    trainer = UniversalTrainer(config=config)
    assert isinstance(trainer.accelerator, Accelerator)

def test_training_loop_execution(dummy_data):
    model = SimpleModel()
    config = TrainingConfig(max_epochs=1, accelerator="cpu")
    trainer = UniversalTrainer(config=config)
    
    metrics = trainer.fit(model, dummy_data)
    
    assert "train_loss" in metrics

def test_gradient_accumulation(dummy_data):
    model = SimpleModel()
    config = TrainingConfig(max_epochs=1, accelerator="cpu", gradient_accumulation_steps=2)
    trainer = UniversalTrainer(config=config)
    trainer.fit(model, dummy_data)
    assert trainer.global_step == 5

def test_checkpoints(tmp_path):
    """Test save/load checkpoints via accelerator."""
    config = TrainingConfig(accelerator="cpu")
    trainer = UniversalTrainer(config=config)
    # Mock accelerator to verify call
    with patch.object(trainer.accelerator, "save_state") as mock_save:
        trainer.save_checkpoint(str(tmp_path))
        mock_save.assert_called_with(str(tmp_path))
        
    with patch.object(trainer.accelerator, "load_state") as mock_load:
        trainer.load_checkpoint(str(tmp_path))
        mock_load.assert_called_with(str(tmp_path))

def test_fallback_validation_and_methods(dummy_data):
    """Test trainer works with vanilla PyTorch models (no training_step)."""
    # This hits fallback paths in fit() and _validate()
    model = VanillaModel()
    
    # We must patch backward() because Accelerate might complain about backwarding tensor not created from prepare?
    # Actually on CPU simple linear model should work fine.
    
    trainer = UniversalTrainer(max_epochs=1, accelerator="cpu")
    
    # Run fit with validation
    metrics = trainer.fit(model, dummy_data, dummy_data)
    
    assert "train_loss" in metrics
    assert "val_loss" in metrics
    
    # Ensure fallback was likely used (by checking model methods not called or just the fact it didn't crash)

def test_mixed_precision_init():
    with patch("kladml.training.trainer.Accelerator") as mock_acc_cls:
        config = TrainingConfig(accelerator="cpu", mixed_precision="fp16")
        trainer = UniversalTrainer(config=config)
        mock_acc_cls.assert_called_with(mixed_precision="fp16", gradient_accumulation_steps=1, cpu=True, log_with="mlflow")
