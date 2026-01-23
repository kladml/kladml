
import pandas as pd
import numpy as np
import os
from typing import Any, Dict, List, Tuple
from ..pipeline import PipelineComponent

class ChronologicalSplitter(PipelineComponent):
    """
    Splits dataset into Train/Val/Test preserving chronological order.
    Respects trip boundaries (doesn't cut a trip in half).
    """
    def __init__(self, output_dir: str, train_ratio: float = 0.60, val_ratio: float = 0.20):
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        super().__init__(config={"out": output_dir, "ratios": [train_ratio, val_ratio]})

    def fit(self, data: Any) -> 'ChronologicalSplitter':
        return self

    def transform(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Splits DataFrame and saves parts.
        Returns paths to saved files.
        """
        if df.empty:
            return {}
            
        if 'trip_id' not in df.columns:
            # If no trips, just straight split? Or error?
            # Assume single trip
            df['trip_id'] = 0
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Trip Stats
        # Sort trips by start time
        trip_stats = df.groupby('trip_id').agg(
            start_time=('timestamp', 'min'), 
            count=('timestamp', 'count')
        ).sort_values('start_time')
        
        total_rows = trip_stats['count'].sum()
        target_train = total_rows * self.train_ratio
        target_val = total_rows * self.val_ratio
        
        train_trips = []
        val_trips = []
        test_trips = []
        
        current_count = 0
        
        # Greedy allocation respecting order
        for trip_id, row in trip_stats.iterrows():
            count = row['count']
            
            if current_count < target_train:
                train_trips.append(trip_id)
            elif current_count < (target_train + target_val):
                val_trips.append(trip_id)
            else:
                test_trips.append(trip_id)
            
            current_count += count
            
        # Save
        parts = {}
        for name, trips in [('train', train_trips), ('val', val_trips), ('test', test_trips)]:
            if not trips:
                continue
                
            subset = df[df['trip_id'].isin(trips)].copy()
            # Drop aux columns if cleaner output desired? Keep timestamp.
            # Maybe drop trip_id if not needed downstream? Keep it for debug.
            
            path = os.path.join(self.output_dir, f"{name}.parquet")
            subset.to_parquet(path)
            parts[name] = path
            
        # Also save split metadata?
        return parts
