
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Optional

from kladml.models.base import BaseModel
from kladml.tasks import MLTask

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


def detect_ml_task(
    model_path: Optional[str] = None,
    run_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
) -> Optional[MLTask]:
    """
    Detect the ML task from model or run metadata.

    Tries multiple strategies in order:
    1. Load checkpoint metadata.json and check for task field
    2. Load the model class and check its ml_task property
    3. Check run params from tracker (if available)

    Args:
        model_path: Path to model file (.py or checkpoint)
        run_dir: Path to run directory containing checkpoints
        checkpoint_dir: Direct path to checkpoint directory

    Returns:
        MLTask if detected, None otherwise
    """
    # Strategy 1: Check checkpoint metadata
    metadata_paths = []
    if checkpoint_dir:
        metadata_paths.append(Path(checkpoint_dir) / "metadata.json")
    if run_dir:
        metadata_paths.append(Path(run_dir) / "checkpoints" / "metadata.json")
        metadata_paths.append(Path(run_dir) / "metadata.json")

    for meta_path in metadata_paths:
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                if "task" in meta:
                    return MLTask(meta["task"])
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

    # Strategy 2: Load model class and check ml_task property
    if model_path:
        try:
            # Try loading as a model class definition
            if model_path.endswith(".py"):
                model_cls = load_model_class_from_path(model_path)
                # Instantiate with empty config to get ml_task
                instance = model_cls(config={})
                return instance.ml_task

            # Try loading as a checkpoint (torch .pt/.pth file)
            if model_path.endswith((".pt", ".pth", ".ckpt")):
                # Try to find model definition alongside checkpoint
                checkpoint_path = Path(model_path)
                model_py = checkpoint_path.parent / "model.py"
                if model_py.exists():
                    model_cls = load_model_class_from_path(str(model_py))
                    instance = model_cls(config={})
                    return instance.ml_task

                # Check for metadata alongside checkpoint
                meta_path = checkpoint_path.parent / "metadata.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                    if "task" in meta:
                        return MLTask(meta["task"])

        except Exception:
            pass

    return None


