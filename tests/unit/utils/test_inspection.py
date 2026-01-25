
import pytest
import polars as pl
import numpy as np
from datetime import datetime, timedelta
from kladml.utils.inspection import inspect_dataset, generate_smart_config
from kladml.models.base import BaseModel
from unittest.mock import patch, MagicMock

@pytest.fixture
def dummy_parquet(tmp_path):
    path = tmp_path / "data.parquet"
    df = pl.DataFrame({
        "feat1": np.random.randn(50),
        "feat2": np.random.randn(50),
        "feat3": np.random.randn(50),
        "target": np.random.randint(0, 2, 50),
        "id": ["a"] * 50
    })
    df.write_parquet(path)
    return str(path)

@pytest.fixture
def dummy_timeseries(tmp_path):
    path = tmp_path / "ts.parquet"
    
    start = datetime(2023, 1, 1).replace(tzinfo=None)
    end = start + timedelta(minutes=5*199)
    # Use datetime_range for sub-day intervals or just manually generate list if version issues
    # pl.datetime_range is often cleaner for timestamps
    times = pl.datetime_range(start, end, "5m", eager=True)
    
    df = pl.DataFrame({
        "glucose": np.random.randn(200),
        "insulin": np.random.randn(200),
        "target": np.random.randn(200),
        "time": times
    })
    df.write_parquet(path)
    return str(path)

def test_inspect_classification(dummy_parquet):
    heuristics = inspect_dataset(dummy_parquet)
    
    assert heuristics["input_dim"] == 3 # feat1, feat2, feat3
    assert heuristics["num_rows"] == 50
    assert heuristics["num_classes"] == 2

def test_inspect_timeseries(dummy_timeseries):
    heuristics = inspect_dataset(dummy_timeseries)
    
    # glucose, insulin (target is excluded from input features typically?? 
    # Logic says exclude target.
    # Logic in code: exclude = {"target", "label", ...}
    # So features = glucose, insulin.
    assert heuristics["input_dim"] == 2 
    assert heuristics["num_rows"] == 200
    # No num_classes for float target

def test_generate_smart_config_mock_model(dummy_parquet):
    # Mock resolve_model_class to return a dummy model with defaults
    with patch("kladml.utils.inspection.resolve_model_class") as mock_resolve:
        MockModel = MagicMock()
        MockModel.default_config.return_value = {"d_model": 512, "batch_size": 128}
        mock_resolve.return_value = MockModel
        
        config = generate_smart_config("mymodel", dummy_parquet)
        
        # input_dim=3 -> d_model ~ 12 (clamped to 64)
        assert config["d_model"] == 64
        # rows=50 -> batch_size=16
        assert config["batch_size"] == 16
        # num_classes=2 -> loss remains default (or unset if not in heuristic logic for 2 classes)
        # Logic says if > 2 switch to cross_entropy.
        assert "loss_mode" not in config # remains default
