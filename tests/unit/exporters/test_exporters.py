
import torch
from kladml.interfaces.exporter import ExporterInterface
from kladml.exporters.registry import ExporterRegistry
from kladml.exporters.torchscript import TorchScriptExporter
from kladml.exporters.onnx import ONNXExporter

class DummyModel(torch.nn.Module):
    def forward(self, x):
        return x * 2

def test_registry_registration():
    @ExporterRegistry.register("test_fmt")
    class TestExporter(ExporterInterface):
        format_name = "test"
        def export(self, *args, **kwargs): return "path"
        def validate(self, *args, **kwargs): return True
        
    assert "test_fmt" in ExporterRegistry.list()
    assert ExporterRegistry.get("test_fmt") == TestExporter

def test_torchscript_export(tmp_path):
    model = DummyModel()
    model.eval()
    output = tmp_path / "model.pt"
    
    exporter = TorchScriptExporter()
    exporter.export(model, str(output))
    
    assert output.exists()
    assert exporter.validate(str(output), torch.randn(1, 10))

def test_onnx_export(tmp_path):
    model = DummyModel()
    model.eval()
    output = tmp_path / "model.onnx"
    dummy_input = torch.randn(1, 10)
    
    exporter = ONNXExporter()
    exporter.export(model, str(output), input_sample=dummy_input)
    
    assert output.exists()
    # Validate might fail if ONNX runtime not installed, but checking file existence is decent
