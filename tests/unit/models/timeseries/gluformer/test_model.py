
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from kladml.models.timeseries.transformer.gluformer.model import GluformerModel

class MockScaler:
    def __init__(self):
        self.scale_ = [2.0]
    
    def transform(self, x):
        return x / 2.0
        
    def inverse_transform(self, x):
        return x * 2.0

@pytest.fixture
def gluformer_config():
    return {
        "seq_len": 10,
        "pred_len": 5,
        "label_len": 5,
        "d_model": 32,
        "n_heads": 2,
        "e_layers": 1,
        "d_layers": 1,
        "warmup_epochs": 2,
        "loss_mode": "nll"
    }

@pytest.fixture
def model(gluformer_config):
    return GluformerModel(gluformer_config)

def test_config_defaults():
    # Test that defaults are loaded if config is empty
    m = GluformerModel()
    assert m.seq_len == 60 # Default
    assert m.loss_mode == "nll"

def test_config_override(gluformer_config):
    # Test overrides
    m = GluformerModel(gluformer_config)
    assert m.seq_len == 10
    assert m.d_model == 32

def test_build_model(model):
    # Test architecture instantiation
    net = model.build_model()
    assert net is not None
    # Check if moved to device (cpu default or mps/cuda if avail)
    device_type = next(net.parameters()).device.type
    assert device_type in ["cpu", "mps", "cuda"]

def test_loss_function_switching(model):
    # Warmup epoch (0 < 2) -> Should be MSE
    loss_fn_mse = model._get_loss_function(epoch=0)
    
    # MSE: (mean, var, target) -> ignores var
    mean = torch.tensor([1.0], requires_grad=True)
    var = torch.tensor([0.5], requires_grad=True)
    target = torch.tensor([1.0])
    
    loss = loss_fn_mse(mean, var, target)
    assert loss == 0.0 # MSE(1.0, 1.0)
    
    # Post-warmup (2 >= 2) -> Should be NLL
    loss_fn_nll = model._get_loss_function(epoch=2)
    
    # NLL logic check
    # NLL = 0.5 * (exp(-logvar) * (y-mean)^2 + logvar)
    # If mean=1, target=1 -> (y-mean)^2 = 0
    # NLL = 0.5 * (0 + logvar) + reg
    loss_nll = loss_fn_nll(mean, var, target)
    assert loss_nll != 0.0

def test_predict_flow(model):
    model.build_model()
    model.model.eval()
    
    # Mock scaler
    model._scaler = MockScaler()
    
    # Mock architecture output to return fixed values
    # Output is (mean, logvar)
    # We want mean to result in specific risk assessment
    # Normal: 100
    # Hypo: 60
    # Hyper: 200
    
    # We need to patch the forward pass of the internal torch model
    with patch.object(model.model, 'forward') as mock_fwd:
        # Case 1: Normal (100)
        # Scaler inverse is * 2.0. So model should output 50.
        # Scaler transform is / 2.0.
        
        # Mock returns (mean, logvar) tensors
        # Shape: [batch, pred_len, features] -> features=1 for univariate
        # Actually predict returns [batch, pred_len, 1] usually
        mock_fwd.return_value = (
            torch.full((1, 5, 1), 50.0), 
            torch.full((1, 5, 1), 0.0)
        )
        
        input_seq = [100.0] * 10
        result = model.predict(input_seq)
        
        assert result["risk_assessment"] == "normal"
        assert result["forecast"][0] == 100.0 # 50 * 2
        
        # Case 2: Hypo (<70)
        # Target 60. Model out 30.
        mock_fwd.return_value = (
            torch.full((1, 5, 1), 30.0), 
            torch.full((1, 5, 1), 0.0)
        )
        result = model.predict(input_seq)
        assert result["risk_assessment"] == "hypoglycemia_risk"
        
        # Case 3: Hyper (>180)
        # Target 200. Model out 100.
        mock_fwd.return_value = (
            torch.full((1, 5, 1), 100.0), 
            torch.full((1, 5, 1), 0.0)
        )
        result = model.predict(input_seq)
        assert result["risk_assessment"] == "hyperglycemia_risk"

def test_predict_input_validation(model):
    model.build_model()
    # Too short input
    with pytest.raises(ValueError, match="Need at least"):
        model.predict([1.0, 2.0])

def test_training_step(model):
    model.build_model()
    # Mock batch
    batch = {
        "x_enc": torch.randn(2, 10, 2),
        "x_id": torch.randn(2, 10, 2),
        "x_dec": torch.randn(2, 10, 2),
        "y": torch.randn(2, 5, 1) # Pred len 5
    }
    
    # Mock output
    with patch.object(model.model, 'forward') as mock_fwd:
        mock_fwd.return_value = (torch.randn(2, 5, 1), torch.randn(2, 5, 1))
        
        # Run step
        metrics = model.training_step(batch, 0)
        assert "loss" in metrics

def test_export_unsupported(model):
    with pytest.raises(NotImplementedError):
        model.export_model("path", format="unknown")
