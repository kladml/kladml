
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from ..pipeline import PipelineComponent

class J1939Cleaner(PipelineComponent):
    """
    Cleans J1939 data by masking error codes (fe, ff bytes) and impossible values.
    """
    def __init__(self, limits: Dict[str, float] = None):
        # Default limits based on SAE J1939 standard ranges
        self.limits = limits or {
            "rpm": 8031.875,
            "speed_kmh": 250.996,
            "accel_pedal": 100.0,
            "torque": 10000.0 
        }
        super().__init__(config={"limits": self.limits})

    def fit(self, data: Any) -> 'J1939Cleaner':
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
            
        df = df.copy()
        
        for col, limit in self.limits.items():
            if col in df.columns:
                # Mask values exceeding physical limits (usually error codes or FF)
                mask = df[col] > limit
                if mask.any():
                    # Set to 0 or NaN?
                    # User script set to 0.0. We'll stick to that for compatibility,
                    # but maybe NaN is better for models? 
                    # Let's keep 0.0 as per "prepare_canbus_dataset.py" behavior.
                    df.loc[mask, col] = 0.0
                    
        return df

class TripSegmenter(PipelineComponent):
    """
    Identifies trips based on time gaps.
    """
    def __init__(self, gap_seconds: float = 2.0, min_duration_seconds: float = 60.0):
        self.gap_seconds = gap_seconds
        self.min_duration_seconds = min_duration_seconds
        super().__init__(config={"gap": gap_seconds, "min_dur": min_duration_seconds})
        
    def fit(self, data: Any) -> 'TripSegmenter':
        return self
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
            
        df = df.copy()
        
        # Ensure timestamp column availability or index
        if 'timestamp' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
             df['timestamp'] = df.index
             
        if 'timestamp' not in df.columns:
            # Fallback if just index
            raise ValueError("TripSegmenter requires 'timestamp' column or DatetimeIndex")

        # 1. Identify Changes
        # Calculate diff in seconds
        time_diff = df['timestamp'].diff().dt.total_seconds().fillna(0)
        
        # New trip if gap > threshold
        trip_change = time_diff > self.gap_seconds
        df['trip_id'] = trip_change.cumsum()
        
        # 2. Filter Short Trips
        # We need sampling rate to convert seconds to count? Or just use time span?
        # Using time span is more robust.
        
        trip_durations = df.groupby('trip_id')['timestamp'].agg(lambda x: (x.max() - x.min()).total_seconds())
        valid_trips = trip_durations[trip_durations >= self.min_duration_seconds].index
        
        # Filter
        df_clean = df[df['trip_id'].isin(valid_trips)].copy()
        
        return df_clean
