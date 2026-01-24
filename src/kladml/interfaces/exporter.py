from abc import ABC, abstractmethod
from typing import Any

class ExporterInterface(ABC):
    """
    Interface for model exporters.
    """
    
    @property
    @abstractmethod
    def format_name(self) -> str:
        """Name of the format (e.g., 'onnx', 'torchscript')."""
        pass
        
    @abstractmethod
    def export(self, model: any, output_path: str, input_sample: Any = None, **kwargs) -> str:
        """
        Export the model to the target format.
        
        Args:
            model: The loaded model object (usually PyTorch nn.Module)
            output_path: Path to save the artifact
            input_sample: Sample input for tracing/shape inference
            **kwargs: format-specific options (e.g. quantize=True)
            
        Returns:
            str: Path to the exported artifact
        """
        pass
        
    @abstractmethod
    def validate(self, exported_path: str, input_sample: Any = None) -> bool:
        """
        Validate the exported model (optional).
        """
        return True
