
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



# Common aliases for ease of use
MODEL_ALIASES = {
    "gluformer": "kladml.models.timeseries.transformer.gluformer",
    "gluformer_model": "kladml.models.timeseries.transformer.gluformer",
    "transformer": "kladml.models.timeseries.transformer.base",
}

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
        
    # 2. Check aliases
    module_path = MODEL_ALIASES.get(model_identifier.lower())
    
    # 3. If not alias, try direct import (e.g. "timeseries.transformer.gluformer" relative to models)
    if not module_path:
        # If it contains dots, assume full path relative to kladml.models
        if "." in model_identifier:
            module_path = f"kladml.models.{model_identifier}"
        else:
             # Try simple mapping
             module_path = f"kladml.models.{model_identifier}"
             
    # Attempt import
    try:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
             # Fallback: maybe the user passed "timeseries.transformer.gluformer" without kladml.models prefix
             # or maybe it's just not found.
             if module_path.startswith("kladml.models."):
                 module_path_short = module_path.replace("kladml.models.", "")
                 # Retry with different prefix if needed? No, standard is kladml.models.
                 pass
             raise ValueError(f"Module '{module_path}' not found.")

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

