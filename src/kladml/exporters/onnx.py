import torch
from typing import Any
from kladml.interfaces.exporter import ExporterInterface
from kladml.exporters.registry import ExporterRegistry

@ExporterRegistry.register("onnx")
class ONNXExporter(ExporterInterface):
    """
    Exports PyTorch models to ONNX.
    """
    
    @property
    def format_name(self) -> str:
        return "onnx"
        
    def export(self, model: torch.nn.Module, output_path: str, input_sample: Any = None, **kwargs) -> str:
        if input_sample is None:
            raise ValueError("ONNX export requires an input sample for tracing.")
            
        opset_version = kwargs.get("opset_version", 11)
        dynamic_axes = kwargs.get("dynamic_axes", None)
        
        torch.onnx.export(
            model,
            input_sample,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )
        return output_path
        
    def validate(self, exported_path: str, input_sample: Any = None) -> bool:
        try:
            import onnx
            onnx_model = onnx.load(exported_path)
            onnx.checker.check_model(onnx_model)
            return True
        except ImportError:
            print("ONNX not installed, skipping validation.")
            return True
        except Exception as e:
            print(f"Validation failed: {e}")
            return False
