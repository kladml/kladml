
from pathlib import Path
import joblib
import numpy as np

def analyze_pkl(path: Path) -> dict:
    """Analyze a .pkl file and return metadata."""
    try:
        data = joblib.load(path)
    except Exception as e:
        return {
            "path": str(path),
            "size": path.stat().st_size,
            "type": "Error",
            "classification": "corrupt",
            "error": str(e)
        }
    
    result = {
        "path": str(path),
        "size": path.stat().st_size,
        "type": type(data).__name__,
    }
    
    # Analyze based on type
    if isinstance(data, list):
        result["num_items"] = len(data)
        
        if len(data) > 0:
            first = data[0]
            result["item_type"] = type(first).__name__
            
            if isinstance(first, np.ndarray):
                # List of arrays (time series)
                lengths = [len(arr) for arr in data]
                result["series_lengths"] = {
                    "min": min(lengths),
                    "max": max(lengths),
                    "mean": np.mean(lengths),
                    "total_samples": sum(lengths),
                }
                
                # Sample statistics
                # Use fewer samples for speed
                sample_count = min(10, len(data))
                all_values = np.concatenate([arr.flatten() for arr in data[:sample_count]])
                result["sample_stats"] = {
                    "min": float(np.min(all_values)),
                    "max": float(np.max(all_values)),
                    "mean": float(np.mean(all_values)),
                    "std": float(np.std(all_values)),
                }
                result["classification"] = "timeseries_list"
                
            elif hasattr(first, 'shape'):
                # Pandas DataFrame or similar
                result["item_shape"] = str(first.shape) if hasattr(first, 'shape') else "N/A"
                if hasattr(first, 'columns'):
                    result["columns"] = list(first.columns)[:10]
                result["classification"] = "dataframe_list"
    
    elif isinstance(data, np.ndarray):
        result["shape"] = data.shape
        result["dtype"] = str(data.dtype)
        result["sample_stats"] = {
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
        }
        result["classification"] = "numpy_array"
    
    elif isinstance(data, dict):
        result["keys"] = list(data.keys())[:20]
        result["num_keys"] = len(data)
        
        # Check if it's a scaler
        if 'mean_' in data or 'scale_' in data or 'mean' in data:
            result["classification"] = "scaler_coefficients"
            if 'mean_' in data:
                # Handle numpy scalars wrapped in arrays
                m = data['mean_']
                result["scaler_mean"] = float(m[0]) if hasattr(m, '__getitem__') and hasattr(m, 'shape') else float(m)
            if 'scale_' in data:
                s = data['scale_']
                result["scaler_scale"] = float(s[0]) if hasattr(s, '__getitem__') and hasattr(s, 'shape') else float(s)
        else:
            result["classification"] = "dictionary"
    
    elif hasattr(data, 'mean_') and hasattr(data, 'scale_'):
        # sklearn scaler object
        result["classification"] = "sklearn_scaler"
        result["scaler_mean"] = float(data.mean_[0])
        result["scaler_scale"] = float(data.scale_[0])
    
    else:
        result["classification"] = "unknown"
        if hasattr(data, 'shape'):
            result["shape"] = data.shape
    
    return result


def analyze_parquet(path: Path) -> dict:
    """Analyze a .parquet file using Polars."""
    import polars as pl
    
    try:
        # Lazy scan for speed
        lf = pl.scan_parquet(path)
        schema = lf.schema
        
        # Quick stats?
        # Just getting length via count
        total_rows = lf.select(pl.count()).collect().item()
        
        result = {
            "path": str(path),
            "size": path.stat().st_size,
            "type": "Parquet (Polars)",
            "classification": "tabular",
            "num_items": total_rows,
            "columns": list(schema.keys()),
            "schema": {k: str(v) for k, v in schema.items()}
        }
        
        return result
    except Exception as e:
        return {
            "path": str(path),
            "size": path.stat().st_size,
            "type": "Parquet (Error)",
            "classification": "corrupt",
            "error": str(e)
        }
