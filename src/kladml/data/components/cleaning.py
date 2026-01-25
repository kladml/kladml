
import polars as pl
from typing import Any
from ..pipeline import PipelineComponent

class J1939Cleaner(PipelineComponent):
    """
    Cleans J1939 data by masking error codes (fe, ff bytes) and impossible values.
    """
    def __init__(self, limits: dict[str, float] | None = None):
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

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return df
            
        # We process columns that exist
        exprs = []
        for col, limit in self.limits.items():
            if col in df.columns:
                # Mask values exceeding physical limits (set to 0.0)
                # Logic: if val > limit, then 0.0, else val
                exprs.append(
                    pl.when(pl.col(col) > limit)
                    .then(0.0)
                    .otherwise(pl.col(col))
                    .alias(col)
                )
        
        if exprs:
            return df.with_columns(exprs)
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
        
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
             return df
            
        # Ensure timestamp column availability
        # Need to check if proper datetime type
        if 'timestamp' not in df.columns:
            raise ValueError("TripSegmenter requires 'timestamp' column in DataFrame")

        # 1. Identify Changes
        # Calculate diff in seconds
        # Polars: diff() returns Duration
        time_diff = df["timestamp"].diff().dt.total_seconds().fill_null(0.0)
        
        # New trip if gap > threshold
        trip_change = time_diff > self.gap_seconds
        trip_id = trip_change.cum_sum().alias("trip_id")
        
        df = df.with_columns([trip_id])
        
        # 2. Filter Short Trips
        # Group by trip_id, calculate duration (max - min)
        # We can use a window function or groupby-agg-join
        
        trip_stats = (
            df.group_by("trip_id")
            .agg([
                (pl.col("timestamp").max() - pl.col("timestamp").min()).dt.total_seconds().alias("duration")
            ])
            .filter(pl.col("duration") >= self.min_duration_seconds)
        )
        
        # Filter original df to keep only valid trips
        # Semi-join
        valid_trips = trip_stats.select("trip_id")
        df_clean = df.join(valid_trips, on="trip_id", how="inner")
        
        return df_clean
