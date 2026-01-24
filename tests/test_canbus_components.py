
import pytest
import torch
import numpy as np
import pandas as pd
import os
from kladml.models.timeseries.transformer.canbus.dataset import CanBusDataset
from kladml.models.timeseries.transformer.canbus.architecture import CanBusTransformer
from kladml.models.timeseries.transformer.canbus.model import CanBusModel

# Create strict dummy data for testing
@pytest.fixture
def dummy_parquet(tmp_path):
    # create 3 trips:
    # Trip 1: 100 samples (gap=0.5s)
    # Trip 2: 50 samples (gap=0.5s) -> Starts after 10s gap
    # Trip 3: 130 samples (gap=0.5s) -> Starts after 10s gap
    
    data = []
    current_time = pd.Timestamp("2021-01-01 00:00:00")
    
    # Trip 1 (100 rows)
    for i in range(100):
        data.append({
            "timestamp": current_time, 
            "rpm": 1000 + i, "torque": 50, "speed_kmh": 10 + i*0.1, "accel_pedal": 20
        })
        current_time += pd.Timedelta(seconds=0.5)
        
    # Gap 10s
    current_time += pd.Timedelta(seconds=10)
    
    # Trip 2 (50 rows) - Too short for window=60 ?
    for i in range(50):
        data.append({
            "timestamp": current_time, 
            "rpm": 2000, "torque": 60, "speed_kmh": 30, "accel_pedal": 40
        })
        current_time += pd.Timedelta(seconds=0.5)

    # Gap 10s
    current_time += pd.Timedelta(seconds=10)
    
    # Trip 3 (130 rows)
    for i in range(130):
        data.append({
            "timestamp": current_time, 
            "rpm": 3000, "torque": 10, "speed_kmh": 0, "accel_pedal": 0
        })
        current_time += pd.Timedelta(seconds=0.5)

    df = pd.DataFrame(data)
    path = tmp_path / "test_data.parquet"
    df.to_parquet(path)
    return str(path)

def test_dataset_windowing(dummy_parquet):
    """Test if dataset correctly creates windows and respects trip boundaries."""
    # Window size 60 samples
    dataset = CanBusDataset(dummy_parquet, window_size=60, step=1)
    
    # Analyze expected windows:
    # Trip 1 (100 rows, w=60) -> 100 - 60 + 1 = 41 valid windows
    # Trip 2 (50 rows, w=60) -> 0 valid windows (too short)
    # Trip 3 (130 rows, w=60) -> 130 - 60 + 1 = 71 valid windows
    # Total expected: 41 + 71 = 112
    
    assert len(dataset) == 112, f"Expected 112 windows, got {len(dataset)}"
    
    # Check item shape
    x, y = dataset[0]
    assert x.shape == (60, 4) # 4 features
    assert torch.equal(x, y) # Autoencoder target is input
    
    # Check Normalization (approx)
    # RPMs are 1000, 2000, 3000. Mean should be roughly 2000.
    # Scaled values should cover -1 to 1 range approx.
    assert torch.max(x) < 10.0 and torch.min(x) > -10.0

def test_model_architecture():
    """Test CanBusTransformer forward pass and shapes."""
    batch_size = 8
    seq_len = 60
    num_features = 4
    
    model = CanBusTransformer(
        num_features=num_features,
        d_model=32,
        n_heads=2,
        e_layers=1,
        seq_len=seq_len
    )
    
    x = torch.randn(batch_size, seq_len, num_features)
    y_hat = model(x)
    
    assert y_hat.shape == x.shape, f"Output shape mismatch: {y_hat.shape} vs {x.shape}"

def test_model_wrapper_initialization():
    """Test CanBusModel wrapper initialization."""
    config = {
        "num_features": 4,
        "d_model": 32,
        "device": "cpu"
    }
    wrapper = CanBusModel(config)
    wrapper.build_model()
    
    assert wrapper.model is not None
    assert sum(p.numel() for p in wrapper.model.parameters()) > 0
