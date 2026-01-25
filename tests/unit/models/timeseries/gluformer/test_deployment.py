
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from kladml.models.timeseries.transformer.gluformer.deployment import (
    GluformerDeploymentWrapper, 
    export_to_torchscript
)

class MockGluformer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x_id, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Return dummy mean/logvar
        # Output shape: [Batch, PredLen, 1]
        # x_dec shape is [Batch, Label+Pred, 1]
        # We just return random
        batch_size = x_enc.size(0)
        # We need to know pred_len, but here we just return matching x_dec length for simplicity of mocking
        # Wait, wrapper expects [Batch, PredLen, 1]
        # Let's assume PredLen=12
        return torch.randn(batch_size, 12, 1), torch.randn(batch_size, 12, 1)

def test_deployment_wrapper_forward():
    inner_model = MockGluformer()
    label_len = 5
    pred_len = 12
    temp = 0.5
    
    wrapper = GluformerDeploymentWrapper(inner_model, label_len, pred_len, temp)
    
    # Input: [Batch, SeqLen, 1]
    # SeqLen must be > label_len
    x_enc = torch.randn(2, 60, 1)
    
    # Run forward
    mean, logvar = wrapper(x_enc)
    
    assert mean.shape == (2, 12, 1)
    assert logvar.shape == (2, 12, 1)
    
    # Verify logic using a spy or mock if needed
    # But shape check confirms basic flow.
    # Check temperature scaling: 
    # Mock inner model to return fixed logvar
    with patch.object(inner_model, 'forward') as mock_fwd:
        mock_fwd.return_value = (torch.zeros(2, 12, 1), torch.ones(2, 12, 1)) # logvar=1.0
        
        mean, logvar = wrapper(x_enc)
        
        # logvar_scaled = logvar - log(temp)
        # 1.0 - log(0.5) = 1.0 - (-0.693) = 1.693
        expected = 1.0 - torch.log(torch.tensor(0.5))
        assert torch.allclose(logvar, expected)

def test_export_to_torchscript_success(tmp_path):
    model = MockGluformer()
    output_path = tmp_path / "model.pt"
    
    # Mock scaler
    scaler = MagicMock()
    scaler.mean_ = [50.0]
    scaler.scale_ = [0.5]
    
    with patch("torch.jit.trace") as mock_trace:
        mock_traced_module = MagicMock()
        mock_trace.return_value = mock_traced_module
        
        export_to_torchscript(
            model=model,
            output_path=str(output_path),
            scaler=scaler,
            temperature=0.8
        )
        
        # Verify save called with metadata
        mock_traced_module.save.assert_called_once()
        args, kwargs = mock_traced_module.save.call_args
        extra = kwargs["_extra_files"]
        
        assert extra["scaler_mean"] == b"50.0"
        assert extra["scaler_scale"] == b"0.5"
        assert extra["temperature"] == b"0.8"

def test_export_to_torchscript_no_scaler(tmp_path):
    model = MockGluformer()
    output_path = tmp_path / "model.pt"
    
    with patch("torch.jit.trace") as mock_trace:
        mock_traced_module = MagicMock()
        mock_trace.return_value = mock_traced_module
        
        export_to_torchscript(model, str(output_path), scaler=None)
        
        args, kwargs = mock_traced_module.save.call_args
        extra = kwargs["_extra_files"]
        assert extra["scaler_mean"] == b"0.0" # Default
        assert extra["scaler_scale"] == b"1.0"

def test_export_failure_handling(tmp_path):
    model = MockGluformer()
    with patch("torch.jit.trace", side_effect=RuntimeError("Trace failed")):
        with pytest.raises(RuntimeError):
            export_to_torchscript(model, str(tmp_path / "fail.pt"))
