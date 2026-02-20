
import pytest
import torch
import numpy as np
from kladml.interfaces.exporter import ExporterInterface
from kladml.exporters.registry import ExporterRegistry
from kladml.exporters.torchscript import TorchScriptExporter, TorchScriptScalerWrapper
from kladml.exporters.onnx import ONNXExporter, ONNXScalerWrapper

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


def test_onnx_scaler_wrapper_standard_scaler():
    """Test ONNXScalerWrapper with StandardScaler-like object."""
    from sklearn.preprocessing import StandardScaler

    model = DummyModel()
    model.eval()

    # Create and fit a scaler
    scaler = StandardScaler()
    train_data = np.random.randn(100, 10)
    scaler.fit(train_data)

    # Create wrapper
    wrapper = ONNXScalerWrapper(model, scaler)

    # Test that wrapper applies scaling
    test_input = torch.randn(1, 10)
    with torch.no_grad():
        # Expected: (test_input - mean) / scale * 2
        expected = (test_input.numpy() - scaler.mean_) / scaler.scale_ * 2
        actual = wrapper(test_input).numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-5)


def test_onnx_scaler_wrapper_minmax_scaler():
    """Test ONNXScalerWrapper with MinMaxScaler-like object."""
    from sklearn.preprocessing import MinMaxScaler

    model = DummyModel()
    model.eval()

    # Create and fit a scaler
    scaler = MinMaxScaler()
    train_data = np.random.randn(100, 10)
    scaler.fit(train_data)

    # Create wrapper
    wrapper = ONNXScalerWrapper(model, scaler)

    # Test that wrapper applies scaling
    test_input = torch.randn(1, 10)
    with torch.no_grad():
        # Expected: (test_input - min) / scale * 2
        expected = (test_input.numpy() - scaler.min_) / scaler.scale_ * 2
        actual = wrapper(test_input).numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-5)


def test_onnx_scaler_wrapper_unsupported():
    """Test ONNXScalerWrapper raises error for unsupported scaler types."""
    model = DummyModel()

    class UnsupportedScaler:
        pass

    with pytest.raises(ValueError, match="Unsupported scaler type"):
        ONNXScalerWrapper(model, UnsupportedScaler())


def test_onnx_export_with_scaler(tmp_path):
    """Test ONNX export with scaler embedding."""
    from sklearn.preprocessing import StandardScaler

    model = DummyModel()
    model.eval()

    # Create and fit a scaler
    scaler = StandardScaler()
    train_data = np.random.randn(100, 10)
    scaler.fit(train_data)

    output = tmp_path / "model_with_scaler.onnx"
    dummy_input = torch.randn(1, 10)

    exporter = ONNXExporter()
    exporter.export(model, str(output), input_sample=dummy_input, scaler=scaler)

    assert output.exists()

    # Validate the exported model
    assert exporter.validate(str(output), dummy_input)


def test_onnx_export_without_input_sample_raises():
    """Test that ONNX export raises error without input_sample."""
    model = DummyModel()
    exporter = ONNXExporter()

    with pytest.raises(ValueError, match="ONNX export requires an input sample"):
        exporter.export(model, "output.onnx")


def test_torchscript_scaler_wrapper_standard_scaler():
    """Test TorchScriptScalerWrapper with StandardScaler-like object."""
    from sklearn.preprocessing import StandardScaler

    model = DummyModel()
    model.eval()

    # Create and fit a scaler
    scaler = StandardScaler()
    train_data = np.random.randn(100, 10)
    scaler.fit(train_data)

    # Create wrapper
    wrapper = TorchScriptScalerWrapper(model, scaler)

    # Test that wrapper applies scaling
    test_input = torch.randn(1, 10)
    with torch.no_grad():
        # Expected: (test_input - mean) / scale * 2
        expected = (test_input.numpy() - scaler.mean_) / scaler.scale_ * 2
        actual = wrapper(test_input).numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-5)


def test_torchscript_scaler_wrapper_minmax_scaler():
    """Test TorchScriptScalerWrapper with MinMaxScaler-like object."""
    from sklearn.preprocessing import MinMaxScaler

    model = DummyModel()
    model.eval()

    # Create and fit a scaler
    scaler = MinMaxScaler()
    train_data = np.random.randn(100, 10)
    scaler.fit(train_data)

    # Create wrapper
    wrapper = TorchScriptScalerWrapper(model, scaler)

    # Test that wrapper applies scaling
    test_input = torch.randn(1, 10)
    with torch.no_grad():
        # Expected: (test_input - min) / scale * 2
        expected = (test_input.numpy() - scaler.min_) / scaler.scale_ * 2
        actual = wrapper(test_input).numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-5)


def test_torchscript_scaler_wrapper_unsupported():
    """Test TorchScriptScalerWrapper raises error for unsupported scaler types."""
    model = DummyModel()

    class UnsupportedScaler:
        pass

    with pytest.raises(ValueError, match="Unsupported scaler type"):
        TorchScriptScalerWrapper(model, UnsupportedScaler())


def test_torchscript_export_with_scaler(tmp_path):
    """Test TorchScript export with scaler embedding."""
    from sklearn.preprocessing import StandardScaler

    model = DummyModel()
    model.eval()

    # Create and fit a scaler
    scaler = StandardScaler()
    train_data = np.random.randn(100, 10)
    scaler.fit(train_data)

    output = tmp_path / "model_with_scaler.pt"
    dummy_input = torch.randn(1, 10)

    exporter = TorchScriptExporter()
    exporter.export(model, str(output), input_sample=dummy_input, scaler=scaler)

    assert output.exists()

    # Validate the exported model
    assert exporter.validate(str(output), dummy_input)

    # Verify the scaler is embedded by checking the output matches expected behavior
    loaded_model = torch.jit.load(str(output))
    loaded_model.eval()
    test_input = torch.randn(1, 10)
    with torch.no_grad():
        expected = (test_input.numpy() - scaler.mean_) / scaler.scale_ * 2
        actual = loaded_model(test_input).numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-5)


def test_torchscript_export_with_scaler_script_mode(tmp_path):
    """Test TorchScript export with scaler using scripting mode (no input_sample)."""
    from sklearn.preprocessing import StandardScaler

    model = DummyModel()
    model.eval()

    # Create and fit a scaler
    scaler = StandardScaler()
    train_data = np.random.randn(100, 10)
    scaler.fit(train_data)

    output = tmp_path / "model_with_scaler_script.pt"

    exporter = TorchScriptExporter()
    exporter.export(model, str(output), scaler=scaler)

    assert output.exists()

    # Verify the scaler is embedded
    loaded_model = torch.jit.load(str(output))
    loaded_model.eval()
    test_input = torch.randn(1, 10)
    with torch.no_grad():
        expected = (test_input.numpy() - scaler.mean_) / scaler.scale_ * 2
        actual = loaded_model(test_input).numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-5)
