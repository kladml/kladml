
import pytest
import polars as pl
from datetime import datetime, timedelta
from kladml.data.components.cleaning import J1939Cleaner, TripSegmenter

def test_j1939_cleaner():
    cleaner = J1939Cleaner(limits={"rpm": 8000.0})
    
    df = pl.DataFrame({
        "rpm": [1000.0, 5000.0, 9000.0, 2000.0],
        "speed": [10.0, 20.0, 30.0, 40.0]
    })
    
    cleaned = cleaner.transform(df)
    
    assert cleaned["rpm"][0] == 1000.0
    assert cleaned["rpm"][2] == 0.0  # Masked (9000 > 8000)
    assert cleaned["speed"][2] == 30.0 # Untouched

def test_trip_segmenter():
    # Create dataset with 2 trips separated by a gap
    t0 = datetime(2024, 1, 1, 10, 0, 0)
    
    # Trip 1: 100 seconds
    timestamps1 = [t0 + timedelta(seconds=i) for i in range(101)]
    
    # Gap of 100 seconds
    t1 = timestamps1[-1] + timedelta(seconds=100)
    
    # Trip 2: only 10 seconds (should be filtered out if min=60)
    timestamps2 = [t1 + timedelta(seconds=i) for i in range(11)]
    
    df = pl.DataFrame({
        "timestamp": timestamps1 + timestamps2,
        "val": range(len(timestamps1) + len(timestamps2))
    })
    
    segmenter = TripSegmenter(gap_seconds=10.0, min_duration_seconds=60.0)
    result = segmenter.transform(df)
    
    # Trip 1 should remain, Trip 2 should be gone
    assert "trip_id" in result.columns
    assert len(result) == 101 # Only Trip 1 points
    assert result["trip_id"][0] == 0
    
    # Verify Trip 2 (short) is gone
    # Trip 2 would have started after index 100
    assert result["timestamp"].max() == timestamps1[-1]
