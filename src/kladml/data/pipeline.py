
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Union
import json
import os

@dataclass
class DatasetMetadata:
    """
    Metadata schema for datasets.
    Acts as a 'Passport' for compatibility checks.
    """
    source_type: str  # e.g., "j1939", "cgm"
    format: str       # e.g., "parquet", "hdf5"
    
    # Time properties
    freq: Optional[str] = None      # e.g., "0.5s"
    is_equispaced: bool = False
    
    # Feature properties
    columns: List[str] = field(default_factory=list)
    
    # Statistics (Optional)
    num_samples: Optional[int] = None
    
    def save(self, path: str):
        """Save metadata to JSON."""
        data = self.__dict__.copy()
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
            
    @classmethod
    def load(cls, path: str) -> 'DatasetMetadata':
        """Load metadata from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class PipelineComponent(ABC):
    """
    Abstract base class for pipeline stages.
    Transformers, Parsers, Scalers etc.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
    @abstractmethod
    def fit(self, data: Any) -> 'PipelineComponent':
        """Learn parameters (if any)."""
        pass
        
    @abstractmethod
    def transform(self, data: Any) -> Any:
        """Apply transformation."""
        pass
        
    def fit_transform(self, data: Any) -> Any:
        return self.fit(data).transform(data)


class DataPipeline(PipelineComponent):
    """
    Composite pipeline that runs components sequentially.
    """
    def __init__(self, steps: List[PipelineComponent]):
        super().__init__()
        self.steps = steps
        
    def fit(self, data: Any) -> 'DataPipeline':
        """
        Fit all steps sequentially.
        Note: requires output of step N to be input of step N+1.
        """
        current_data = data
        for step in self.steps:
            current_data = step.fit_transform(current_data)
        return self
        
    def transform(self, data: Any) -> Any:
        current_data = data
        for step in self.steps:
            current_data = step.transform(current_data)
        return current_data
        
    def append(self, step: PipelineComponent):
        self.steps.append(step)

    @classmethod
    def from_yaml(cls, config_path: str) -> 'DataPipeline':
        """Load pipeline from YAML config."""
        import yaml
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        steps_config = config.get("pipeline", [])
        if not steps_config and isinstance(config, list):
             # Support pure list based config
             steps_config = config
             
        components = []
        for step_cfg in steps_config:
            # step_cfg: {"J1939Parser": {args...}} or {"type": "J1939Parser", ...}
            
            if isinstance(step_cfg, str):
                name = step_cfg
                kwargs = {}
            elif isinstance(step_cfg, dict):
                # Check for "type" style
                if "type" in step_cfg:
                    name = step_cfg.pop("type")
                    kwargs = step_cfg
                # Check for "Key: Value" style
                elif len(step_cfg) == 1:
                    name = list(step_cfg.keys())[0]
                    kwargs = step_cfg[name]
                else:
                    raise ValueError(f"Ambiguous step config: {step_cfg}")
            else:
                 raise ValueError(f"Invalid step config: {step_cfg}")
                 
            component_cls = ComponentRegistry.get(name)
            if not component_cls:
                raise ValueError(f"Unknown component: {name}")
                
            components.append(component_cls(**kwargs))
            
        return cls(steps=components)


class ComponentRegistry:
    """
    Registry for pipeline components.
    Allows creating components by name string.
    """
    _registry = {}
    
    @classmethod
    def register(cls, name: str, component_cls: type):
        cls._registry[name] = component_cls
        
    @classmethod
    def get(cls, name: str) -> Optional[type]:
        return cls._registry.get(name)

# Auto-register logic (imports components to trigger registration)
def _register_defaults():
    # Only import if needed to avoid circular imports?
    # Better to import inside method or rely on users importing them.
    # For CLI, we will explicitely import standard components.
    pass

