import torch
from typing import Any
from kladml.interfaces.exporter import ExporterInterface
from kladml.exporters.registry import ExporterRegistry

@ExporterRegistry.register("torchscript")
class TorchScriptExporter(ExporterInterface):
    """
    Exports PyTorch models to TorchScript (JIT).
    """
    
    @property
    def format_name(self) -> str:
        return "torchscript"
        
    def export(self, model: torch.nn.Module, output_path: str, input_sample: Any = None, **kwargs) -> str:
        # TODO: Handle scaler embedding if needed via wrapper
        
        if input_sample is not None:
             # Trace
             scripted_model = torch.jit.trace(model, input_sample)
        else:
             # Script
             scripted_model = torch.jit.script(model)
             
        # Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        scripted_model.save(output_path)
        return output_path
        
    def validate(self, exported_path: str, input_sample: Any = None) -> bool:
        try:
            loaded = torch.jit.load(exported_path)
            loaded.eval()
            if input_sample is not None:
                loaded(input_sample)
            return True
        except Exception as e:
            print(f"Validation failed: {e}")
            return False
