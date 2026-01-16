
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple, List

class HDF5GluformerDataset(Dataset):
    """
    Lazy-loading dataset for Gluformer backed by HDF5.
    
    Reads data from disk on-demand, enabling training on datasets 
    larger than available RAM.
    
    Structure of HDF5 file:
        metadata/
            num_series (int)
        series/
            <index>/
                glucose (N,)
                insulin (N,) [optional]
    
    Args:
        h5_path: Path to .h5 file
        input_chunk_length: Length of input sequence (encoder)
        output_chunk_length: Length of output sequence (decoder future)
        label_len: Length of start token (decoder past)
        scaler: Optional sklearn scaler (must implement transform)
        c_in: Number of input channels (1 for univariate, 2 for multivariate)
    """
    
    def __init__(
        self,
        h5_path: str,
        input_chunk_length: int = 60,
        output_chunk_length: int = 12,
        label_len: int = 48,
        scaler: Optional[object] = None,
        c_in: int = 1,
    ):
        super().__init__()
        self.h5_path = str(h5_path)
        self.input_len = input_chunk_length
        self.output_len = output_chunk_length
        self.label_len = label_len
        self.scaler = scaler
        self.c_in = c_in
        
        # Open file to read metadata and build index
        # We close it immediately to ensure picklability for DataLoader workers
        with h5py.File(self.h5_path, 'r') as f:
            self.num_series = f.attrs.get('num_series') 
            # Fallback if attribute missing
            if self.num_series is None and 'series' in f:
                self.num_series = len(f['series'])
            elif self.num_series is None:
                # Try new format structure if needed, or default
                if 'metadata' in f:
                    self.num_series = f['metadata'].attrs.get('num_series')
            
            if self.num_series is None:
                raise ValueError(f"Could not determine num_series in {self.h5_path}")
                
            # Pre-compute valid windows to avoid checking during training
            # This requires iterating sizes, which is fast with HDF5 headers
            self.windows = self._build_windows(f)
            
        print(f"[HDF5Dataset] Indexed {len(self.windows)} windows from {self.num_series} series.")
        
        # File handle for lazy loading (per-worker)
        self._h5_file = None

    def _build_windows(self, f: h5py.File) -> List[Tuple[int, int]]:
        """Scan all series lengths and build valid window indices."""
        windows = []
        stride = 1
        total_len = self.input_len + self.output_len
        
        series_grp = f['series']
        
        # Iterate over series keys. Assuming keys are str(int) indices "0", "1", ...
        # Standardize iteration order
        for i in range(self.num_series):
            s_key = str(i)
            if s_key not in series_grp:
                continue
                
            # Get length without loading data
            ds = series_grp[s_key]['glucose']
            length = ds.shape[0]
            
            if length < total_len:
                continue
            
            # Simple sliding window
            # Range: [0, length - total_len]
            # If length == total_len, range(0, 1) -> start=0
            for start in range(0, length - total_len + 1, stride):
                windows.append((i, start))
                
        return windows

    def _open_file(self):
        """Lazy opener for HDF5 file."""
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r')
        return self._h5_file

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        series_idx, start = self.windows[idx]
        
        f = self._open_file()
        
        # Access the specific series group
        grp = f['series'][str(series_idx)]
        
        # Calculate end index
        total_len = self.input_len + self.output_len
        end = start + total_len
        
        # Slice data from HDF5 (loads only this chunk into RAM)
        glucose = grp['glucose'][start:end]
        
        # Handle scaling if provided
        if self.scaler:
            # Note: scaler.transform expects 2D array (N, 1)
            # glucose is (N,) or (N,1) depending on HDF5 saving
            if glucose.ndim == 1:
                g_reshaped = glucose.reshape(-1, 1)
            else:
                g_reshaped = glucose
                
            glucose = self.scaler.transform(g_reshaped).flatten()
            
        # Prepare inputs based on channels
        if self.c_in == 1:
            # Univariate
            x_enc = glucose[:self.input_len].reshape(-1, 1).astype(np.float32)
            
            # Decoder input (Start Token + Zeros)
            # Start Token: last label_len points of encoder input
            token_start = self.input_len - self.label_len
            start_token = glucose[token_start:self.input_len].reshape(-1, 1).astype(np.float32)
            zeros = np.zeros((self.output_len, 1), dtype=np.float32)
            x_dec = np.concatenate([start_token, zeros], axis=0)
            
        else:
            # Multivariate (Glucose + Insulin)
            if 'insulin' in grp:
                insulin = grp['insulin'][start:end]
            else:
                insulin = np.zeros_like(glucose)
            
            # Note: We typically don't scale insulin or we assume it's pre-scaled/features
            # If needed, a separate scaler for insulin would be passed
            
            x_gl = glucose[:self.input_len].reshape(-1, 1).astype(np.float32)
            x_ins = insulin[:self.input_len].reshape(-1, 1).astype(np.float32)
            x_enc = np.concatenate([x_gl, x_ins], axis=1)
            
            # Decoder
            token_start = self.input_len - self.label_len
            t_gl = x_enc[token_start:, 0:1]
            t_ins = x_enc[token_start:, 1:2]
            start_token = np.concatenate([t_gl, t_ins], axis=1)
            
            zeros = np.zeros((self.output_len, self.c_in), dtype=np.float32)
            x_dec = np.concatenate([start_token, zeros], axis=0)

        # Target (Future Glucose)
        y = glucose[self.input_len:].reshape(-1, 1).astype(np.float32)
        
        # ID (Static feature) - just series index for now
        x_id = np.array([float(series_idx % 1000)], dtype=np.float32)
        
        return {
            'x_enc': torch.tensor(x_enc),
            'x_dec': torch.tensor(x_dec),
            'x_id': torch.tensor(x_id),
            'y': torch.tensor(y)
        }

    def __del__(self):
        """Ensure file is closed when dataset is destroyed"""
        if self._h5_file is not None:
            self._h5_file.close()
