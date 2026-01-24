
import pytest
from unittest.mock import MagicMock, call
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from kladml.training.trainer import UniversalTrainer
from kladml.training.callbacks import Callback

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 1)
        
    def forward(self, x):
        return self.layer(x)
        
    def training_step(self, batch, batch_idx):
        # Allow models to define their own step logic if needed
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        return {"val_loss": loss}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

@pytest.fixture
def dummy_loaders():
    # Create simple dataset
    X = torch.randn(20, 10)
    y = torch.randn(20, 1)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=4)
    return loader, loader

def test_trainer_initialization():
    """Test that trainer initializes with correct defaults."""
    trainer = UniversalTrainer(max_epochs=5)
    assert trainer.max_epochs == 5
    assert trainer.devices == "auto"  # Check the config, not the resolved device

def test_trainer_fit_loop(dummy_loaders):
    """Test standard training loop execution."""
    train_loader, val_loader = dummy_loaders
    model = SimpleModel()
    
    # Mock callback to verify loop progression
    mock_callback = MagicMock(spec=Callback)
    
    trainer = UniversalTrainer(
        max_epochs=2,
        callbacks=[mock_callback],
        accelerator="cpu" # Force CPU for unit test
    )
    
    metrics = trainer.fit(model, train_loader, val_loader)
    
    assert metrics is not None
    # Check that model parameters changed (training happened)
    # (In a real scenario we'd clone params before and compare)
    
    # Verify Callback calls
    assert mock_callback.on_train_begin.called
    assert mock_callback.on_epoch_begin.call_count == 2
    assert mock_callback.on_epoch_end.call_count == 2
    assert mock_callback.on_train_end.called
    
def test_trainer_handles_custom_step(dummy_loaders):
    """Test that trainer uses model's custom training_step if defined."""
    train_loader, _ = dummy_loaders
    model = SimpleModel()
    model.training_step = MagicMock(wraps=model.training_step)
    
    trainer = UniversalTrainer(max_epochs=1, accelerator="cpu")
    trainer.fit(model, train_loader)
    
    assert model.training_step.called
