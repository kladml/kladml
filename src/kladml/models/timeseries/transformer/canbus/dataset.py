
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional

class CanBusDataset(Dataset):
    """
    Dataset for CAN Bus Anomaly Detection.
    
    Features:
    - Loads Parquet file.
    - Normalizes data (StandardScaler).
    - Creates sliding windows respecting trip boundaries.
    - Returns (window, window) for Autoencoder training.
    """
    
    def __init__(
        self, 
        path: str, 
        scaler_stats: Optional[dict] = None, # {'mean': [...], 'std': [...]}
        window_size: int = 120, # 60 seconds
        step: int = 1,
        features: List[str] = ['rpm', 'torque', 'speed_kmh', 'accel_pedal']
    ):
        self.path = path
        self.window_size = window_size
        self.step = step
        self.features = features
        
        # Load Data
        df = pd.read_parquet(path)
        self.raw_data = df[features].values.astype(np.float32)
        timestamps = pd.to_datetime(df['timestamp'])
        
        # Calculate Scaler (if not provided) or Apply
        if scaler_stats is None:
            print("Calculating Scaler stats from data (assuming Train set)...")
            self.mean = np.mean(self.raw_data, axis=0)
            self.std = np.std(self.raw_data, axis=0)
            # Avoid division by zero
            self.std[self.std < 1e-6] = 1.0
            self.scaler_stats = {'mean': self.mean, 'std': self.std}
        else:
            self.mean = scaler_stats['mean']
            self.std = scaler_stats['std']
            self.scaler_stats = scaler_stats
            
        # Normalize
        self.data = (self.raw_data - self.mean) / self.std
        
        # Build Index (Sliding Window Config)
        self.indices = self._build_indices(timestamps)
        
    def _build_indices(self, timestamps: pd.Series) -> np.ndarray:
        """
        Create a list of (start_idx) for valid windows.
        A window is valid if it lies entirely within one trip (gap <= 2.0s).
        """
        # Identify gaps > 2.0s
        # True if gap > 2.0s at that index
        # We need to vectorized this.
        
        # Calculate time diffs
        # We can detect trips first
        gaps = timestamps.diff().dt.total_seconds().fillna(0).values
        trip_ids = (gaps > 2.0).cumsum()
        
        valid_starts = []
        total_len = len(self.data)
        
        # We iterate through data. Vectorized approach:
        # A window [i : i+w] is valid if trip_ids[i] == trip_ids[i+w-1]
        
        # Generate all possible starts
        possible_starts = np.arange(0, total_len - self.window_size + 1, self.step)
        
        # Check trip consistency
        # trip_start = trip_ids[possible_starts]
        # trip_end = trip_ids[possible_starts + self.window_size - 1]
        # valid_mask = (trip_start == trip_end)
        
        start_trips = trip_ids[possible_starts]
        end_trips = trip_ids[possible_starts + self.window_size - 1]
        
        valid_mask = (start_trips == end_trips)
        valid_starts = possible_starts[valid_mask]
        
        print(f"Dataset {self.path}: Found {len(valid_starts)} valid windows out of {len(possible_starts)} possible.")
        return valid_starts

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end = start + self.window_size
        
        # Extract window
        x = self.data[start:end]
        
        # Convert to tensor
        x_tensor = torch.from_numpy(x)
        
        # Autoencoder: Input == Target
        return x_tensor, x_tensor
