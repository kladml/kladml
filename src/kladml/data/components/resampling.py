
import pandas as pd
import numpy as np
from typing import Any, Optional, Dict
from ..pipeline import PipelineComponent

class TimeResampler(PipelineComponent):
    """
    Resamples time-series data to a fixed frequency grid using interpolation.
    """
    def __init__(self, rate: float = 0.5, method: str = 'time', limit: int = 4):
        """
        Args:
            rate: Frequency in seconds (e.g., 0.5)
            method: Interpolation method (default: 'time' for weighted linear)
            limit: Max consecutive NaNs to fill (e.g. 4 steps = 2s at 0.5s rate)
        """
        super().__init__(config={"rate": rate, "method": method, "limit": limit})
        self.rate = rate
        self.rate_pd = f"{int(rate*1000)}ms"
        self.method = method
        self.limit = limit
        
    def fit(self, data: Any) -> 'TimeResampler':
        return self
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resamples the input DataFrame.
        Assumes 'timestamp' index or convertable index.
        """
        if df.empty:
            return df
            
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            # Try to find timestamp col? Or assume caller prepared it?
            # Pipeline contract: Expects datetime index for time operations.
            raise ValueError("TimeResampler input must have DatetimeIndex")
            
        # 1. Define Grid
        t_start = df.index.min().floor(self.rate_pd)
        t_end = df.index.max().ceil(self.rate_pd)
        
        grid = pd.date_range(start=t_start, end=t_end, freq=self.rate_pd)
        
        # 2. Union Index
        combined_idx = df.index.union(grid).unique().sort_values()
        combined = df.reindex(combined_idx)
        
        # 3. Interpolate
        combined = combined.interpolate(method=self.method, limit=self.limit, limit_direction='both')
        
        # 4. Extract Grid
        resampled = combined.reindex(grid)
        
        # 5. Drop NaNs (gaps larger than limit)
        resampled.dropna(inplace=True)
        
        return resampled
