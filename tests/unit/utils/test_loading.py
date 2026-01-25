
import pytest
from pathlib import Path
import sys
from kladml.utils.loading import load_model_class_from_path, resolve_model_class
from kladml.models.base import BaseModel
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
         with pytest.raises(ValueError, match="not found in kladml.models"):
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
