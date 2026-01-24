
import logging
from typing import Optional, Any
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import joblib

from kladml.data.datamodule import BaseDataModule
from kladml.models.timeseries.transformer.gluformer.dataset import GluformerDataset
from kladml.data.hdf5_dataset import HDF5GluformerDataset

logger = logging.getLogger(__name__)

class GluformerDataModule(BaseDataModule):
    """
    DataModule for Gluformer.
    Handles loading from HDF5 or Pickle/List, scaling, and Dataloader creation.
    """
    
    def __init__(
        self, 
        train_path: Any, 
        val_path: Optional[Any] = None,
        seq_len: int = 60,
        pred_len: int = 12,
        label_len: int = 48,
        batch_size: int = 64,
        num_workers: int = 0,
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.scaler = StandardScaler()
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load data and fit scaler.
        """
        # 1. Fit Scaler (Logic extracted from original GluformerModel)
        is_hdf5 = isinstance(self.train_path, str) and (self.train_path.endswith('.h5') or self.train_path.endswith('.hdf5'))
        
        if is_hdf5:
             # Lazy loading path
             import h5py
             with h5py.File(self.train_path, 'r') as f:
                 # Check for pre-computed stats in metadata
                 if 'metadata' in f and 'scaler_mean' in f['metadata'].attrs:
                     self.scaler.mean_ = np.array([f['metadata'].attrs['scaler_mean']])
                     self.scaler.scale_ = np.array([f['metadata'].attrs['scaler_scale']])
                     self.scaler.var_ = self.scaler.scale_ ** 2
                     logger.info(f"Loaded scaler stats from HDF5 metadata: mean={self.scaler.mean_}")
                 else:
                     logger.warning("No pre-computed scaler stats in HDF5. Fitting on first 100 series (APPROXIMATE).")
                     # Fit on subset
                     subset_values = []
                     count = 0
                     if 'series' in f:
                         for k in f['series']:
                             subset_values.extend(f['series'][k]['glucose'][:])
                             count += 1
                             if count >= 100: break
                     
                     if subset_values:
                         self.scaler.fit(np.array(subset_values).reshape(-1, 1))
                     else:
                         logger.warning("Could not fit scaler (dataset empty?). Inference may be unscaled.")
        else:
            # Legacy PKL/List handling
            if isinstance(self.train_path, str):
                train_data = joblib.load(self.train_path)
            else:
                train_data = self.train_path
            
            all_values = []
            if isinstance(train_data, list):
                for item in train_data:
                    if hasattr(item, 'values'):
                        all_values.extend(item.values.flatten())
                    elif isinstance(item, dict):
                         all_values.extend(item['glucose'].flatten())
                    else:
                        all_values.extend(np.array(item).flatten())
            else:
                all_values = np.array(train_data).flatten()
            
            self.scaler.fit(np.array(all_values).reshape(-1, 1))

        # 2. Create Train Dataset
        if is_hdf5:
            self.train_dataset = HDF5GluformerDataset(
                self.train_path,
                input_chunk_length=self.seq_len,
                output_chunk_length=self.pred_len,
                label_len=self.label_len,
                scaler=self.scaler,
            )
        else:
            self.train_dataset = GluformerDataset(
                self.train_path,
                input_chunk_length=self.seq_len,
                output_chunk_length=self.pred_len,
                label_len=self.label_len,
                scaler=self.scaler,
            )
            
        # 3. Create Val Dataset
        if self.val_path:
            is_val_hdf5 = isinstance(self.val_path, str) and (self.val_path.endswith('.h5') or self.val_path.endswith('.hdf5'))
            if is_val_hdf5:
                self.val_dataset = HDF5GluformerDataset(
                    self.val_path,
                    input_chunk_length=self.seq_len,
                    output_chunk_length=self.pred_len,
                    label_len=self.label_len,
                    scaler=self.scaler,
                )
            else:
                self.val_dataset = GluformerDataset(
                    self.val_path,
                    input_chunk_length=self.seq_len,
                    output_chunk_length=self.pred_len,
                    label_len=self.label_len,
                    scaler=self.scaler,
                )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available() or torch.backends.mps.is_available(),
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if not self.val_dataset:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
