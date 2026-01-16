
import torch
import numpy as np
import joblib
from torch.utils.data import Dataset

class GluformerDataset(Dataset):
    """
    Dataset for Gluformer that loads .pkl files.
    
    Supports:
    - Univariate: List of numpy arrays (glucose only)
    - Multivariate: List of dicts {'glucose': arr, 'insulin': arr}
    
    Args:
        pkl_path: Path to .pkl file
        input_chunk_length: Input sequence length (default: 60)
        output_chunk_length: Prediction length (default: 12)
        label_len: Decoder label length (default: 48)
        mode: 'train' or 'val'
        scaler: Optional sklearn scaler for glucose normalization
        c_in: Number of input channels (1=univariate, 2=multivariate)
    """
    def __init__(
        self, 
        pkl_path: str, 
        input_chunk_length=60, 
        output_chunk_length=12, 
        label_len=48, 
        mode='train', 
        scaler=None,
        c_in=1
    ):
        super().__init__()
        print(f"[GluformerDataset] Loading {pkl_path}...")
        raw_data = joblib.load(pkl_path)
        
        self.c_in = c_in
        self.data_list = self._convert_data(raw_data)
        
        self.input_len = input_chunk_length
        self.output_len = output_chunk_length
        self.label_len = label_len
        self.mode = mode
        
        # Apply Scaler to glucose
        if scaler:
            print(f"[GluformerDataset] Applying scaler to glucose...")
            for d in self.data_list:
                d['glucose'] = scaler.transform(d['glucose'].reshape(-1, 1)).flatten()
        
        self.windows = []
        self._prepare_windows()
    
    def _convert_data(self, raw_data) -> list:
        """Convert to list of {'glucose': arr, 'insulin': arr}."""
        if isinstance(raw_data, list) and len(raw_data) > 0:
            if isinstance(raw_data[0], dict):
                print(f"[GluformerDataset] Format: {len(raw_data)} dicts")
                return raw_data
            
            elif isinstance(raw_data[0], np.ndarray):
                print(f"[GluformerDataset] Format: {len(raw_data)} numpy arrays (univariate)")
                return [{'glucose': x.flatten().astype(np.float32), 
                         'insulin': np.zeros_like(x, dtype=np.float32)} for x in raw_data]
            else:
                print(f"[GluformerDataset] Format: unknown, converting")
                return [{'glucose': np.array(x).flatten().astype(np.float32), 
                         'insulin': np.zeros_like(np.array(x), dtype=np.float32)} for x in raw_data]
        else:
            raise ValueError(f"Unsupported format: {type(raw_data)}")

    def _prepare_windows(self):
        stride = 1
        total_len = self.input_len + self.output_len
        
        for i, series in enumerate(self.data_list):
            gl_len = len(series['glucose'])
            if gl_len < total_len:
                continue
            
            for start in range(0, gl_len - total_len + 1, stride):
                self.windows.append((i, start))
                
        print(f"[GluformerDataset] Prepared {len(self.windows)} windows from {len(self.data_list)} series.")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        series_idx, start = self.windows[idx]
        series = self.data_list[series_idx]
        
        # Slice glucose
        full_gl = series['glucose'][start : start + self.input_len + self.output_len]
        
        if self.c_in == 1:
            # UNIVARIATE: Only glucose
            x_enc = full_gl[:self.input_len].reshape(-1, 1).astype(np.float32)
            
            # Decoder input
            start_token = full_gl[self.input_len - self.label_len : self.input_len].reshape(-1, 1).astype(np.float32)
            zeros = np.zeros((self.output_len, 1), dtype=np.float32)
            x_dec = np.concatenate([start_token, zeros], axis=0)
        else:
            # MULTIVARIATE: Glucose + Insulin
            full_ins = series['insulin'][start : start + self.input_len + self.output_len]
            
            x_gl = full_gl[:self.input_len].reshape(-1, 1).astype(np.float32)
            x_ins = full_ins[:self.input_len].reshape(-1, 1).astype(np.float32)
            x_enc = np.concatenate([x_gl, x_ins], axis=1)
            
            start_token = x_enc[self.input_len - self.label_len : self.input_len]
            zeros = np.zeros((self.output_len, self.c_in), dtype=np.float32)
            x_dec = np.concatenate([start_token, zeros], axis=0)
        
        # Target: Future glucose
        y = full_gl[self.input_len:].reshape(-1, 1).astype(np.float32)
        
        # Static ID
        x_id = np.array([float(series_idx % 1000)], dtype=np.float32)
        
        return {
            'x_enc': torch.tensor(x_enc),
            'x_dec': torch.tensor(x_dec),
            'x_id': torch.tensor(x_id),
            'y': torch.tensor(y)
        }
