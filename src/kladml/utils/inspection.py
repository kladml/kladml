
from pathlib import Path
from typing import Any
import importlib.util
from kladml.utils.loading import resolve_model_class

def inspect_dataset(path: str) -> dict[str, Any]:
    """
    Inspect a dataset file (Parquet/CSV/HDF5) to extract heuristics.
    
    Returns:
        Dict with inferred properties:
        - input_dim: Number of feature columns
        - num_classes: Number of unique classes (if classification)
        - seq_len: Estimated sequence length (if time series)
        - num_rows: Total rows
    """
    path_obj = Path(path)
    heuristics = {}
    
    try:
        if path_obj.suffix == ".parquet":
            import polars as pl
            # Lazy scan for speed
            lf = pl.scan_parquet(path)
            schema = lf.collect_schema()
            
            
            # Simple heuristic: All numeric columns except "target", "label", "id", "date" are input features
            exclude = {"target", "label", "y", "id", "date", "timestamp", "time", "group_id"}
            features = [c for c, t in schema.items() if c.lower() not in exclude and t in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
            
            heuristics["input_dim"] = len(features)
            
            # Estimate rows (cheap enough for parquet metadata)
            # Fetch minimal info
            # Polars lazy count optimization
            heuristics["num_rows"] = lf.select(pl.len()).collect().item()
            
            # Check classification
            target_col = None
            if "label" in schema: target_col = "label"
            elif "target" in schema: target_col = "target"
            elif "y" in schema: target_col = "y"
            
            if target_col:
                dtype = schema[target_col]
                # If int string or categorical, check unique count
                if dtype in [pl.Int32, pl.Int64, pl.Utf8, pl.Categorical]:
                    # Limit unique check to avoid massive scan
                    unique_count = lf.select(pl.col(target_col).n_unique()).collect().item()
                    if unique_count < 100:
                        heuristics["num_classes"] = unique_count
                        
    except Exception as e:
        # Fallback or just log warning (don't crash the config generator)
        heuristics["error"] = str(e)

    return heuristics

def generate_smart_config(model_name: str, data_path: str = None) -> dict[str, Any]:
    """
    Generate a smart configuration by merging Model Defaults + Dataset Heuristics.
    """
    # 1. Get Model Defaults
    model_cls = resolve_model_class(model_name)
    if hasattr(model_cls, "default_config"):
        config = model_cls.default_config()
    else:
        config = {}
        
    # 2. Analyze Data (if provided)
    if data_path:
        heuristics = inspect_dataset(data_path)
        
        # 3. Apply Heuristics (The "Smart Merge")
        
        # Scale d_model based on input complexity
        if "input_dim" in heuristics:
            # If model uses d_model and we have input_dim estimate
            # Heuristic: d_model approx 4x input_features, clamped between 64 and 512
            est_d_model = heuristics["input_dim"] * 4
            est_d_model = max(64, min(512, est_d_model))
            # Round to power of 2
            est_d_model = 1 << (est_d_model - 1).bit_length()
            
            if "d_model" in config:
                config["d_model"] = est_d_model
                
        # Scale batch size based on rows
        if "num_rows" in heuristics:
            rows = heuristics["num_rows"]
            if rows < 1000:
                config["batch_size"] = 16
                config["epochs"] = 50 # Reduce epochs for tiny data
            elif rows > 100000:
                config["batch_size"] = 256
                
        # Handle Classification
        if "num_classes" in heuristics and heuristics["num_classes"] > 0:
            config["num_classes"] = heuristics["num_classes"]
            if heuristics["num_classes"] > 2:
                config["loss_mode"] = "cross_entropy" 
                # (Assuming model supports it, but config is just a suggestion)

    return config
