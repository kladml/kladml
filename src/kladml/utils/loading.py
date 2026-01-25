
import importlib
import importlib.util
import sys
from pathlib import Path
from kladml.models.base import BaseModel

def load_model_class_from_path(model_path: str):
    """Dynamically load a model class from a .py file."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Use stem as module name to avoid conflicts if possible, or random
    module_name = f"user_model_{path.stem}"
    
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {model_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type) 
            and issubclass(obj, BaseModel) 
            # and obj is not BaseModel # subclass check ensures it validates, check not BaseModel
            and obj.__name__ != "BaseModel" # safer string check sometimes
        ):
            return obj
    
    raise ValueError(f"No BaseModel subclass found in {model_path}.")


def resolve_model_class(model_identifier: str):
    """
    Resolve model class from identifier (name or path).
    
    Args:
        model_identifier: Model name (e.g. "gluformer") or path to .py file
        
    Returns:
        Model class
    """
    # 1. Try loading as file path
    if model_identifier.endswith(".py") or Path(model_identifier).exists():
        return load_model_class_from_path(model_identifier)
        
    # 2. Try loading as architecture name
    try:
        # Import module: kladml.models.{name}
        module_path = f"kladml.models.{model_identifier}"
        try:
            module = importlib.import_module(module_path)
        except ImportError:
             raise ValueError(f"Model '{model_identifier}' not found in kladml.models")

        # Check module's __init__ for a subclass of BaseModel
        for name in dir(module):
            obj = getattr(module, name)
            if (
                isinstance(obj, type) 
                and issubclass(obj, BaseModel) 
                and obj is not BaseModel
            ):
                return obj
                
        # If not found in __init__, try .model submodule
        try:
            model_submodule = importlib.import_module(f"{module_path}.model")
            for name in dir(model_submodule):
                obj = getattr(model_submodule, name)
                if (
                    isinstance(obj, type) 
                    and issubclass(obj, BaseModel) 
                    and obj is not BaseModel
                ):
                    return obj
        except ImportError:
            pass
            
        raise ValueError(f"No BaseModel subclass found in {module_path}")
        
    except Exception as e:
        raise ValueError(f"Could not load model '{model_identifier}': {e}")
