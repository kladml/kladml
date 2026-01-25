
import polars as pl
from typing import Any
from ..pipeline import PipelineComponent

class TimeResampler(PipelineComponent):
    """
    Resamples time-series data to a fixed frequency grid using interpolation.
    """
    def __init__(self, rate: float = 0.5, method: str = 'linear', limit: int = 4):
        """
        Args:
            rate: Frequency in seconds (e.g., 0.5)
            method: Interpolation method (Polars supports 'linear', 'nearest')
            limit: Max consecutive nulls to fill (not directly supported in Polars interpolate logic universally, handled via gap thresholds?)
                   Polars interpolate fills ALL unless we mask?
                   We'll assume linear interpolation across small gaps.
        """
        super().__init__(config={"rate": rate, "method": method, "limit": limit})
        self.rate = rate
        # Polars duration string (e.g. "500ms")
        self.interval = f"{int(rate * 1000000)}us" # Microseconds for precision
        self.method = method # ignored, usually linear
        
    def fit(self, data: Any) -> 'TimeResampler':
        return self
        
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Resamples the input DataFrame.
        """
        if df.is_empty():
            return df
            
        if "timestamp" not in df.columns:
            raise ValueError("TimeResampler input must have 'timestamp' column")

        # 1. Define Grid
        # Create a lazy frame for the grid
        start = df["timestamp"].min()
        end = df["timestamp"].max()
        
        # We want to align to strict grid?
        # Polars: upsample() works on DataFrame.
        # But usually you want `group_by_dynamic` or create a grid and join.
        # Simple Upsample:
        
        # Cast to datetime if needed
        # df = df.with_columns(pl.col("timestamp").cast(pl.Datetime))
        
        # Sort is required for upsample
        q = df.lazy().sort("timestamp")
        
        # Upsample acts like 'reindex' + filling gaps with nulls
        # We use upsample + interpolate
        
        # Note: Polars upsample is on DataFrame, not LazyFrame usually?
        # Let's collect for upsample
        df_sorted = q.collect()
        
        upsampled = df_sorted.upsample(time_column="timestamp", every=self.interval)
        
        # Interpolate numeric columns
        # Filter numeric columns
        numeric_cols = [c for c, t in df.schema.items() if t in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)]
        
        interpolated = upsampled.with_columns([
            pl.col(c).interpolate() for c in numeric_cols
        ])
        
        # Filter to keep only the exact grid points? 
        # Upsample keeps original points too? No, upsample ADDS points.
        # If we want a strict grid (e.g. exactly at .0, .5) regardless of original timestamps:
        # We should generate the grid and join_asof/interpolate?
        # The Pandas implementation used `union` then `reindex(grid)`.
        # This means it keeps grid points INTERPOLATED from nearby points.
        
        # Faster approach: Generate grid, join_asof (nearest? or linear?)
        # Polars doesn't have join_asof check strategy='linear'.
        # Best way to emulate Pandas resampling-interpolation:
        # 1. Concat original + grid.
        # 2. Sort.
        # 3. Interpolate.
        # 4. Filter for grid points.
        
        # Grid generation
        grid = pl.datetime_range(start, end, self.interval, eager=True).alias("timestamp").to_frame()
        
        # Add original data
        combined = pl.concat([df_sorted, grid], how="diagonal")
        
        # Deduplicate timestamps? If grid matches existing point, we have duplicates.
        # If duplicates, we should prioritise original data?
        # sort() + unique(keep='first')
        
        combined = (
            combined
            .sort("timestamp")
            .unique(subset=["timestamp"], keep="first")
        )
        
        # Interpolate
        combined = combined.with_columns([
            pl.col(c).interpolate() for c in numeric_cols
        ])
        
        # Filter only grid points
        # usage semi-join
        result = combined.join(grid, on="timestamp", how="inner")
        
        return result
