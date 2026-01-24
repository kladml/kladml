
import pytest
import torch
import json
from pathlib import Path
from kladml.training.checkpoint import CheckpointManager

class TestCheckpointManager:
    """Tests for CheckpointManager."""
    
    @pytest.fixture
    def manager(self, tmp_path):
        """Create a CheckpointManager instance with a temporary directory."""
        return CheckpointManager(
            project_name="test_proj",
            experiment_name="test_exp",
            run_id="run_123",
            base_dir=str(tmp_path),
            checkpoint_frequency=2
        )
    
    @pytest.fixture
    def dummy_objects(self):
        """Create dummy model and optimizer."""
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        return model, optimizer

    def test_directory_structure(self, manager, tmp_path):
        """Test directory creation."""
        expected_dir = tmp_path / "test_proj" / "test_exp" / "run_123" / "checkpoints"
        assert expected_dir.exists()
        assert manager.checkpoint_dir == expected_dir

    def test_save_checkpoint_periodic(self, manager, dummy_objects):
        """Test periodic checkpoint saving."""
        model, optimizer = dummy_objects
        metrics = {"val_loss": 0.5}
        
        # Pre-set best metric so epoch 1 is NOT best
        manager._best_metric = 0.1
        
        # Epoch 1 (Should NOT save periodic, and 0.5 > 0.1 so not best)
        path = manager.save_checkpoint(model, optimizer, epoch=1, metrics=metrics)
        assert path is None
        assert not (manager.checkpoint_dir / "checkpoint_epoch_1.pt").exists()
        
        # Epoch 2 (Should SAVE)
        path = manager.save_checkpoint(model, optimizer, epoch=2, metrics=metrics)
        assert path is not None
        assert Path(path).exists()
        assert (manager.checkpoint_dir / "checkpoint_epoch_2.pt").exists()

    def test_save_best_model(self, manager, dummy_objects):
        """Test saving best model logic."""
        model, optimizer = dummy_objects
        
        # 1. First save (Best)
        manager.save_checkpoint(model, optimizer, epoch=1, metrics={"val_loss": 0.5})
        assert (manager.checkpoint_dir / "best_model.pt").exists()
        assert manager.best_metric == 0.5
        assert manager.best_epoch == 1
        
        # 2. Worse metric (Should NOT update best)
        manager.save_checkpoint(model, optimizer, epoch=2, metrics={"val_loss": 0.8})
        assert manager.best_metric == 0.5
        
        # 3. Better metric (Should UPDATE best)
        manager.save_checkpoint(model, optimizer, epoch=3, metrics={"val_loss": 0.3})
        assert manager.best_metric == 0.3
        assert manager.best_epoch == 3

    def test_metadata_persistence(self, manager, dummy_objects):
        """Test that metadata is saved and loaded."""
        model, optimizer = dummy_objects
        manager.save_checkpoint(model, optimizer, epoch=5, metrics={"val_loss": 0.1})
        
        metadata_path = manager.checkpoint_dir / "metadata.json"
        assert metadata_path.exists()
        
        with open(metadata_path) as f:
            data = json.load(f)
            assert data["best_epoch"] == 5
            assert data["best_metric"] == 0.1
            
        # Simulate new manager instance loading metadata
        new_manager = CheckpointManager(
            project_name="test_proj",
            experiment_name="test_exp",
            run_id="run_123",
            base_dir=str(manager.base_dir),
            checkpoint_frequency=2
        )
        assert new_manager.best_epoch == 5
        assert new_manager.best_metric == 0.1

    def test_load_checkpoint(self, manager, dummy_objects):
        """Test loading checkpoint state."""
        model, optimizer = dummy_objects
        
        # Save explicit state
        torch.nn.init.constant_(model.weight, 1.0)
        manager.save_checkpoint(model, optimizer, epoch=10, metrics={"val_loss": 0.2})
        
        # Modify model
        torch.nn.init.constant_(model.weight, 0.0)
        
        # Load back
        epoch, metrics, config = manager.load_checkpoint("best", model=model, optimizer=optimizer)
        
        assert epoch == 10
        assert metrics["val_loss"] == 0.2
        assert torch.all(model.weight == 1.0)

    def test_list_checkpoints(self, manager, dummy_objects):
        """Test listing functionality."""
        model, optimizer = dummy_objects
        manager.save_checkpoint(model, optimizer, epoch=2, metrics={"val_loss": 0.5})
        manager.save_checkpoint(model, optimizer, epoch=4, metrics={"val_loss": 0.4})
        
        ckpts = manager.list_checkpoints()
        assert len(ckpts) == 3 # 2 periodic + 1 best
        
        types = [c["type"] for c in ckpts]
        assert "best" in types
        assert types.count("periodic") == 2
