
import pytest
from pathlib import Path
import sys
import json
from kladml.utils.loading import load_model_class_from_path, resolve_model_class, detect_ml_task
from kladml.models.base import BaseModel
from kladml.tasks import MLTask
from unittest.mock import patch, MagicMock

# --- Success Helper ---
def create_dummy_model_file(path: Path, class_name: str = "MyModel"):
    content = f"""
from kladml.models.base import BaseModel
class {class_name}(BaseModel):
    @property
    def ml_task(self): return "classification"
    def train(self, *args, **kwargs): pass
    def predict(self, *args, **kwargs): pass
    def evaluate(self, *args, **kwargs): pass
    def save(self, *args, **kwargs): pass
    def load(self, *args, **kwargs): pass
"""
    path.write_text(content)

def create_invalid_model_file(path: Path):
    content = """
class NotAModel:
    pass
"""
    path.write_text(content)

# --- Tests ---

def test_load_from_path_success(tmp_path):
    f = tmp_path / "model.py"
    create_dummy_model_file(f, "CustomModel")
    
    cls = load_model_class_from_path(str(f))
    assert cls.__name__ == "CustomModel"
    assert issubclass(cls, BaseModel)

def test_load_from_path_not_found():
    with pytest.raises(FileNotFoundError, match="not found"):
        load_model_class_from_path("/non/existent/path.py")

def test_load_from_path_no_subclass(tmp_path):
    f = tmp_path / "nomodel.py"
    create_invalid_model_file(f)
    
    with pytest.raises(ValueError, match="No BaseModel subclass found"):
        load_model_class_from_path(str(f))

def test_resolve_by_path_string(tmp_path):
    f = tmp_path / "model.py"
    create_dummy_model_file(f, "ResolvedModel")
    
    # Pass as string path
    cls = resolve_model_class(str(f))
    assert cls.__name__ == "ResolvedModel"

def test_resolve_by_registry_success():
    # Mock importlib.import_module
    with patch("importlib.import_module") as mock_import:
        mock_module = MagicMock()
        # Define a mock class on the module
        class RegistryModel(BaseModel):
            pass
        mock_module.RegistryModel = RegistryModel
        mock_import.return_value = mock_module
        
        cls = resolve_model_class("cool_model")
        assert cls == RegistryModel
        mock_import.assert_called_with("kladml.models.cool_model")

    def test_resolve_by_registry_not_found():
        with patch("importlib.import_module", side_effect=ImportError):
             with pytest.raises(ValueError, match="not found"):
                 resolve_model_class("unknown_model")

def test_resolve_registry_no_class():
    with patch("importlib.import_module") as mock_import:
        mock_module = MagicMock()
        # Empty module
        mock_import.return_value = mock_module
        
        # Second try logic (submodule .model) will also fail if we mock it to fail or exist empty
        # Let's verify it tries finding it.
        # It loops through dir(module).
        
        # Mock submodule import failure
        # side_effect needs to handle multiple calls
        # 1. kladml.models.xyz (success)
        # 2. kladml.models.xyz.model (failure)
        
        def side_effect(name):
            if name == "kladml.models.bad":
                return mock_module
            raise ImportError
            
        mock_import.side_effect = side_effect
        
        with pytest.raises(ValueError, match="No BaseModel subclass found"):
             resolve_model_class("bad")

def test_resolve_registry_submodule_success():
    """Test finding model in package.module submodule."""
    with patch("importlib.import_module") as mock_import:
        mock_pkg = MagicMock()
        mock_sub = MagicMock()
        
        class SubModel(BaseModel): pass
        mock_sub.SubModel = SubModel
        
        def side_effect(name):
            if name == "kladml.models.complex":
                return mock_pkg # Empty init
            if name == "kladml.models.complex.model":
                return mock_sub
            raise ImportError(name)
        
        mock_import.side_effect = side_effect
        
        cls = resolve_model_class("complex")
        assert cls == SubModel


# --- detect_ml_task Tests ---

def create_model_with_task(path: Path, task: str = "timeseries_forecasting", class_name: str = "TaskModel"):
    """Create a model file with specific ml_task."""
    content = f"""
from kladml.models.base import BaseModel
from kladml.tasks import MLTask

class {class_name}(BaseModel):
    @property
    def ml_task(self):
        return MLTask.{task.upper()}

    def train(self, *args, **kwargs): pass
    def predict(self, *args, **kwargs): pass
    def evaluate(self, *args, **kwargs): pass
    def save(self, *args, **kwargs): pass
    def load(self, *args, **kwargs): pass
"""
    path.write_text(content)


def test_detect_task_from_checkpoint_metadata(tmp_path):
    """Test detecting task from metadata.json in checkpoint dir."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    metadata = {"task": "timeseries_forecasting", "best_epoch": 10}
    (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata))

    result = detect_ml_task(checkpoint_dir=str(checkpoint_dir))
    assert result == MLTask.TIMESERIES_FORECASTING


def test_detect_task_from_run_dir(tmp_path):
    """Test detecting task from run_dir/checkpoints/metadata.json."""
    run_dir = tmp_path / "run_001"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    metadata = {"task": "regression"}
    (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata))

    result = detect_ml_task(run_dir=str(run_dir))
    assert result == MLTask.REGRESSION


def test_detect_task_from_model_file(tmp_path):
    """Test detecting task from model .py file via ml_task property."""
    model_path = tmp_path / "my_model.py"
    create_model_with_task(model_path, "classification")

    result = detect_ml_task(model_path=str(model_path))
    assert result == MLTask.CLASSIFICATION


def test_detect_task_from_model_file_timeseries(tmp_path):
    """Test detecting timeseries task from model file."""
    model_path = tmp_path / "ts_model.py"
    create_model_with_task(model_path, "timeseries_forecasting", "TSModel")

    result = detect_ml_task(model_path=str(model_path))
    assert result == MLTask.TIMESERIES_FORECASTING


def test_detect_task_no_metadata_no_model():
    """Test returns None when no metadata or model available."""
    result = detect_ml_task()
    assert result is None


def test_detect_task_invalid_metadata_json(tmp_path):
    """Test handles invalid JSON gracefully."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    (checkpoint_dir / "metadata.json").write_text("not valid json")

    result = detect_ml_task(checkpoint_dir=str(checkpoint_dir))
    assert result is None


def test_detect_task_metadata_without_task_field(tmp_path):
    """Test returns None when metadata doesn't have task field."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    metadata = {"best_epoch": 5, "best_metric": 0.1}
    (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata))

    result = detect_ml_task(checkpoint_dir=str(checkpoint_dir))
    assert result is None


def test_detect_task_priority_checkpoint_over_model(tmp_path):
    """Test checkpoint metadata takes priority over model file."""
    # Create checkpoint with task
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    metadata = {"task": "regression"}
    (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata))

    # Create model with different task
    model_path = tmp_path / "model.py"
    create_model_with_task(model_path, "classification")

    # Should return checkpoint's task, not model's
    result = detect_ml_task(
        model_path=str(model_path),
        checkpoint_dir=str(checkpoint_dir)
    )
    assert result == MLTask.REGRESSION
