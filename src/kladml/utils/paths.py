"""
Path resolution utilities for KladML local data structure.

Standardizes the local filesystem structure:
data/
├── datasets/        # Global datasets
├── preprocessors/   # Preprocessing scripts/objects
├── models/          # Saved models
└── projects/        # Project artifacts
"""

import os
from pathlib import Path
from typing import Union

# Constants for directory structure
DATA_DIR = "data"
DATASETS_DIR = "datasets"
PREPROCESSORS_DIR = "preprocessors"
REGISTRY_DIR = "registry"
PROJECTS_DIR = "projects"

def get_root_data_path() -> Path:
    """Get the root data directory path (relative to CWD)."""
    return Path.cwd() / DATA_DIR

def resolve_dataset_path(path_or_name: str) -> Path:
    """
    Resolve a dataset path.
    
    If absolute, returns as is.
    If relative, checks if it exists.
    If not, assumes it's a name in data/datasets/.
    """
    path = Path(path_or_name)
    
    if path.is_absolute():
        return path
        
    if path.exists():
        return path.resolve()
        
    # fallback to data/datasets/name
    return get_root_data_path() / DATASETS_DIR / path_or_name

def resolve_preprocessor_path(path_or_name: str) -> Path:
    """
    Resolve a preprocessor path.
    
    Mirrors logic of resolve_dataset_path but for preprocessors.
    """
    path = Path(path_or_name)
    
    if path.is_absolute():
        return path
        
    if path.exists():
        return path.resolve()
        
    return get_root_data_path() / PREPROCESSORS_DIR / path_or_name

def ensure_data_structure():
    """Create the standard data directory structure if not exists."""
    root = get_root_data_path()
    
    (root / DATASETS_DIR).mkdir(parents=True, exist_ok=True)
    (root / PREPROCESSORS_DIR).mkdir(parents=True, exist_ok=True)
    (root / REGISTRY_DIR).mkdir(parents=True, exist_ok=True)
    (root / PROJECTS_DIR).mkdir(parents=True, exist_ok=True)
    
    return root
