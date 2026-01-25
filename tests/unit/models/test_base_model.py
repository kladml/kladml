
import pytest
from unittest.mock import MagicMock
from kladml.models.base import BaseModel
import tempfile
from pathlib import Path

# Concrete implementation for testing
class TestModel(BaseModel):
    @property
    def ml_task(self):
        return "classification"
        
    def train(self, X_train, **kwargs):
        self.on_train_begin()
        # Simulate training loop
        self.on_epoch_end(1, {"loss": 0.5})
        self.on_train_end()
        return {"loss": 0.5}
        
    def predict(self, X):
        return [0, 1]
        
    def evaluate(self, X_test):
        return {"acc": 0.9}
        
    def save(self, path):
        Path(path).touch()
        
    def load(self, path):
        if not Path(path).exists():
            raise FileNotFoundError()

def test_basemodel_init():
    """Test initialization and config."""
    config = {"lr": 0.01}
    model = TestModel(config)
    assert model.config["lr"] == 0.01
    assert len(model.callbacks) == 0

def test_callback_registration():
    """Test standard callbacks (Optuna)."""
    # Simulate Optuna callback logic
    # The base class has _init_standard_callbacks which checks for OptunaTrial
    
    # Manually register mock callback
    model = TestModel()
    cb = MagicMock()
    model.callbacks.append(cb)
    
    # Trigger callbacks via internal methods
    model.on_train_begin()
    cb.on_train_begin.assert_called_once()
    
    model.on_epoch_end(1, {"val": 1})
    cb.on_epoch_end.assert_called_with(1, {"val": 1})
    
    model.on_train_end()
    cb.on_train_end.assert_called_once()

def test_pruning_integration():
    """Test if OptunaPruningCallback is added when trial present."""
    # This tests the _init_standard_callbacks logic in BaseModel
    trial = MagicMock()
    config = {"optuna_trial": trial}
    
    # We need to mock OptunaPruningCallback imports inside base.py?
    # Or just check if it tries to add it.
    
    # If successful, model.callbacks should have 1 item.
    
    # We need to make sure 'optuna' is available (it is in dev env).
    
    model = TestModel(config)
    # Call the initialization method (usually called by Executor)
    model._init_standard_callbacks(run_id="test_run", project_name="test_proj", experiment_name="test_exp")
    
    assert len(model.callbacks) >= 1
    # Check if any callback is pruning
    callback_names = [type(cb).__name__ for cb in model.callbacks]
    assert any("Pruning" in name for name in callback_names), f"Callbacks: {callback_names}"

def test_abstract_methods():
    """Test that BaseModel cannot be instantiated."""
    with pytest.raises(TypeError):
        BaseModel({})
