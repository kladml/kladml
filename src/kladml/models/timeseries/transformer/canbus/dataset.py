
import polars as pl
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional

class CanBusDataset(Dataset):
    """
    Dataset for CAN Bus Anomaly Detection.
    Loads Parquet using Polars (Fast).
    """
    
    def __init__(
        self, 
        path: str, 
        scaler_stats: Optional[dict] = None, # {'mean': [...], 'std': [...]}
        window_size: int = 120, # 60 seconds
        step: int = 1,
        features: list[str] = ['rpm', 'torque', 'speed_kmh', 'accel_pedal']
    ):
        self.path = path
        self.window_size = window_size
        self.step = step
        self.features = features
        
        # Load Data (Polars)
        df = pl.read_parquet(path)
        
        # Extract Numpy Arrays
        # Select features, cast to float32, to_numpy
        self.raw_data = df.select(features).to_numpy().astype(np.float32)
        
        # Timestamps for logic
        # Polars timestamps to numpy (datetime64[ns])
        self.timestamps_ns = df["timestamp"].to_numpy()
        
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
        self.indices = self._build_indices(self.timestamps_ns)
        
    def _build_indices(self, timestamps: np.ndarray) -> np.ndarray:
        """
        Create a list of (start_idx) for valid windows.
        A window is valid if it lies entirely within one trip (gap <= 2.0s).
        """
        # Calculate time diffs in seconds
        # timestamps is numpy datetime64[ns]
        # diff in ns -> / 1e9 -> seconds
        
        diffs = np.zeros(len(timestamps), dtype=np.float64)
        if len(timestamps) > 1:
            diffs[1:] = (timestamps[1:] - timestamps[:-1]).astype(float) / 1e9
            
        trip_ids = (diffs > 2.0).cumsum()
        
        total_len = len(self.data)
        possible_starts = np.arange(0, total_len - self.window_size + 1, self.step)
        
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
