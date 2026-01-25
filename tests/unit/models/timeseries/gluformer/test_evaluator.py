
import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch
from kladml.models.timeseries.transformer.gluformer.evaluator import GluformerEvaluator

@pytest.fixture
def evaluator(tmp_path):
    return GluformerEvaluator(
        run_dir=tmp_path,
        model_path=tmp_path / "model.pt",
        data_path=tmp_path / "data.pkl"
    )

def test_load_model_jit(evaluator):
    # Mock torch.jit.load
    with patch("torch.jit.load") as mock_load:
        # Check extra files logic
        def side_effect(path, _extra_files):
            _extra_files["scaler_mean"] = "50.0"
            _extra_files["scaler_scale"] = "2.0"
            return MagicMock()
            
        mock_load.side_effect = side_effect
        
        model = evaluator.load_model()
        
        assert evaluator._scaler_mean == 50.0
        assert evaluator._scaler_scale == 2.0
        assert model is not None

def test_load_data_pkl_list(evaluator, tmp_path):
    # Create fake pkl data
    import joblib
    data = [np.random.randn(100) for _ in range(5)] # 5 series
    data_path = tmp_path / "data.pkl"
    joblib.dump(data, data_path)
    
    inputs, targets = evaluator.load_data()
    
    # SeqLen=60, PredLen=12. 
    # For each 100 len series: 100 - 60 - 12 + 1 = 29 windows
    # Total = 5 * 29 = 145
    assert len(inputs) == 145
    assert inputs.shape[1] == 60
    assert targets.shape[1] == 12

def test_inference_loop(evaluator):
    model = MagicMock()
    # Mock model output: mean, logvar
    # Batch size 256. If we have 10 inputs.
    model.return_value = (torch.zeros(10, 12, 1), torch.zeros(10, 12, 1))
    
    inputs = np.zeros((10, 60))
    targets = np.zeros((10, 12))
    
    evaluator._scaler_mean = 0.0
    evaluator._scaler_scale = 1.0
    
    preds, tgt = evaluator.inference(model, (inputs, targets))
    
    assert "mean" in preds
    assert "logvar" in preds
    assert preds["mean"].shape == (10, 12)
    assert np.all(preds["mean"] == 0.0)
