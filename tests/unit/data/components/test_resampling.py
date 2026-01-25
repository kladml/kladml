
import pytest
import polars as pl
from datetime import datetime, timedelta
from kladml.data.components.resampling import TimeResampler

def test_time_resampler():
    # 10 Hz data (0.1s)
    # We want to resample to 0.5s (5 Hz)
    
    # Create valid timestamps
    timestamps = [datetime(2024, 1, 1, 10, 0, 0) + (i * timedelta(milliseconds=100)) for i in range(20)] # 2 seconds
    # Values: 0, 1, ... 19
    
    df = pl.DataFrame({
        "timestamp": timestamps,
        "val": [float(i) for i in range(20)]
    })
    
    # rate=0.5 -> We expect points at 0.0, 0.5, 1.0, 1.5, (2.0?)
    resampler = TimeResampler(rate=0.5)
    result = resampler.transform(df)
    
    assert "timestamp" in result.columns
    assert "val" in result.columns
    
    # Check intervals
    # Since original data hits exactly on 0.0, 0.5 (indices 0, 5, 10...), values should be exact.
    # index 0 (0.0s) -> 0.0
    # index 5 (0.5s) -> 5.0
    
    # We can check specific values if "inner" join strategy worked
    # result might have fewer rows if we filter strict grid
    
    t0 = datetime(2024, 1, 1, 10, 0, 0)
    
    val_at_t0 = result.filter(pl.col("timestamp") == t0).select("val").item()
    assert val_at_t0 == 0.0
    
    val_at_t0_5 = result.filter(pl.col("timestamp") == (t0 + timedelta(milliseconds=500))).select("val").item()
    assert val_at_t0_5 == 5.0
    
    # Test Interpolation (Missing points)
    # Data: 0.0s (0), 1.0s (10). Missing 0.5s.
    df_sparse = pl.DataFrame({
        "timestamp": [t0, t0 + timedelta(seconds=1)],
        "val": [0.0, 10.0]
    })
    
    res_sparse = resampler.transform(df_sparse)
    
    # Should contain 0.5s point with value 5.0
    val_mid = res_sparse.filter(pl.col("timestamp") == (t0 + timedelta(milliseconds=500))).select("val").item()
    assert val_mid == 5.0
