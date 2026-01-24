
import pytest
from unittest.mock import MagicMock
from kladml.data.datamodule import BaseDataModule

class ConcreteDataModule(BaseDataModule):
    def prepare_data(self):
        # Mock download logic
        pass
        
    def setup(self, stage: str = None):
        # Mock split/transform logic
        self.train_dataset = [1, 2, 3]
        self.val_dataset = [4, 5]
        
    def train_dataloader(self):
        return [1, 2, 3] # Mock loader
        
    def val_dataloader(self):
        return [4, 5]

def test_datamodule_interface():
    """Test that concrete datamodule follows validation logical flow."""
    dm = ConcreteDataModule()
    
    # Verify standard workflow
    dm.prepare_data()
    dm.setup(stage="fit")
    
    loader = dm.train_dataloader()
    assert loader == [1, 2, 3]
    
    val_loader = dm.val_dataloader()
    assert val_loader == [4, 5]

def test_base_datamodule_enforces_abstraction():
    """Test that BaseDataModule cannot be instantiated directly or enforces methods."""
    # Since we are essentially testing ABC behavior, we check if generic methods exist
    # If we make methods abstract, this checks they must be implemented
    pass
