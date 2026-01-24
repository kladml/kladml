"""
Tests for model export functionality (ONNX, TorchScript).
"""
import pytest
import torch
import os
import tempfile
import numpy as np

from kladml.models.timeseries.transformer.canbus.model import CanBusModel


@pytest.fixture
def canbus_model():
    """Create a minimal CanBus model for testing."""
    config = {
        "d_model": 32,  # Small for fast tests
        "n_heads": 2,
        "e_layers": 1,
        "seq_len": 10,
        "num_features": 4,
        "device": "cpu"
    }
    model = CanBusModel(config)
    model.build_model()
    return model


class TestTorchScriptExport:
    """Tests for TorchScript export."""
    
    def test_export_torchscript_creates_file(self, canbus_model):
        """Test that export_torchscript creates a .pt file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            canbus_model.export_torchscript(path)
            assert os.path.exists(path)
    
    def test_exported_torchscript_is_loadable(self, canbus_model):
        """Test that exported TorchScript can be loaded without source code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            canbus_model.export_torchscript(path)
            
            # Load without any class definitions
            loaded = torch.jit.load(path)
            assert loaded is not None
    
    def test_exported_torchscript_produces_correct_output(self, canbus_model):
        """Test that TorchScript output matches PyTorch output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            canbus_model.export_torchscript(path)
            
            # Create test input
            test_input = torch.randn(1, 10, 4)
            
            # PyTorch output
            canbus_model.model.eval()
            with torch.no_grad():
                pytorch_out = canbus_model.model(test_input).numpy()
            
            # TorchScript output
            loaded = torch.jit.load(path)
            loaded.eval()
            with torch.no_grad():
                ts_out = loaded(test_input).numpy()
            
            assert np.allclose(pytorch_out, ts_out, atol=1e-5)


class TestONNXExport:
    """Tests for ONNX export."""
    
    def test_export_onnx_creates_file(self, canbus_model):
        """Test that export_onnx creates an .onnx file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            canbus_model.export_onnx(path, validate=False)
            assert os.path.exists(path)
    
    def test_export_model_dispatcher_onnx(self, canbus_model):
        """Test that export_model with format='onnx' works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            canbus_model.export_model(path, format="onnx")
            assert os.path.exists(path)
    
    def test_export_model_dispatcher_torchscript(self, canbus_model):
        """Test that export_model with format='torchscript' works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            canbus_model.export_model(path, format="torchscript")
            assert os.path.exists(path)
    
    @pytest.mark.skipif(
        not pytest.importorskip("onnxruntime", reason="onnxruntime not installed"),
        reason="onnxruntime not installed"
    )
    def test_onnx_validation_passes(self, canbus_model):
        """Test that ONNX validation passes (outputs match)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            # This should not raise if validation passes
            canbus_model.export_onnx(path, validate=True)


class TestExportModelFormat:
    """Tests for export_model format handling."""
    
    def test_invalid_format_raises(self, canbus_model):
        """Test that invalid format raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.xyz")
            with pytest.raises(ValueError, match="Unsupported export format"):
                canbus_model.export_model(path, format="invalid_format")
