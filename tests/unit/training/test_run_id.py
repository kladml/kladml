
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from kladml.training.run_id import generate_run_id, get_run_checkpoint_dir

class TestRunIdGenerator:
    """Tests for run ID generation logic."""
    
    def test_generate_run_id_first_run(self, tmp_path):
        """Test first run ID generation."""
        # Using tmp_path as base_dir
        run_id = generate_run_id(
            project_name="proj",
            experiment_name="exp",
            base_dir=str(tmp_path)
        )
        assert run_id.startswith("run_001_")
        # Check directories created
        assert (tmp_path / "proj" / "exp").exists()

    def test_generate_run_id_increment(self, tmp_path):
        """Test that run numbers increment correctly."""
        base = tmp_path
        exp_dir = base / "proj" / "exp"
        exp_dir.mkdir(parents=True)
        
        # Simulate existing runs
        (exp_dir / "run_001_20240101").mkdir()
        (exp_dir / "run_002_20240101").mkdir()
        
        run_id = generate_run_id(
            project_name="proj",
            experiment_name="exp",
            base_dir=str(base)
        )
        
        assert run_id.startswith("run_003_")

    def test_generate_run_id_with_family(self, tmp_path):
        """Test handling of family structure."""
        run_id = generate_run_id(
            project_name="proj",
            experiment_name="exp",
            family_name="fam",
            base_dir=str(tmp_path)
        )
        assert (tmp_path / "proj" / "fam" / "exp").exists()
        assert run_id.startswith("run_001_")

class TestCheckpointDir:
    """Tests for checkpoint directory path generation."""
    
    def test_get_run_checkpoint_dir(self, tmp_path):
        path = get_run_checkpoint_dir(
            project_name="proj",
            experiment_name="exp",
            run_id="run_123",
            base_dir=str(tmp_path)
        )
        expected = tmp_path / "proj_exp" / "run_123"
        assert path == expected
        assert path.exists()
