#!/usr/bin/env python3
"""
Convert PKL datasets to HDF5 format for lazy loading.

HDF5 Structure:
    dataset.h5
    ├── metadata/
    │   ├── num_series (int)
    │   ├── format_version (str)
    │   └── created_at (str)
    └── series/
        ├── 0/
        │   ├── glucose (float32 array)
        │   └── insulin (float32 array, optional)
        ├── 1/
        │   └── ...
        └── N/

Usage:
    python scripts/convert_pkl_to_hdf5.py input.pkl output.h5
    
Or via CLI:
    kladml data convert --input input.pkl --output output.h5 --format hdf5
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import joblib
import h5py


def convert_pkl_to_hdf5(
    input_path: str,
    output_path: str,
    compression: str = "gzip",
    compression_level: int = 4,
) -> dict:
    """
    Convert a PKL dataset to HDF5 format.
    
    Args:
        input_path: Path to input .pkl file
        output_path: Path to output .h5 file
        compression: Compression algorithm (gzip, lzf, or None)
        compression_level: Compression level (1-9 for gzip)
        
    Returns:
        dict with conversion statistics
    """
    print(f"[convert] Loading {input_path}...")
    raw_data = joblib.load(input_path)
    
    # Normalize to list of dicts format
    data_list = _normalize_data(raw_data)
    
    print(f"[convert] Found {len(data_list)} series")
    
    # Create HDF5 file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_points = 0
    
    with h5py.File(output_path, "w") as f:
        # Metadata group
        meta = f.create_group("metadata")
        meta.attrs["num_series"] = len(data_list)
        meta.attrs["format_version"] = "1.0"
        meta.attrs["created_at"] = datetime.now().isoformat()
        meta.attrs["source_file"] = str(input_path)
        
        # Series group
        series_grp = f.create_group("series")
        
        for i, item in enumerate(data_list):
            grp = series_grp.create_group(str(i))
            
            # Glucose (required)
            glucose = item["glucose"].astype(np.float32)
            grp.create_dataset(
                "glucose",
                data=glucose,
                compression=compression,
                compression_opts=compression_level if compression == "gzip" else None,
            )
            
            # Insulin (optional)
            if "insulin" in item and item["insulin"] is not None:
                insulin = item["insulin"].astype(np.float32)
                grp.create_dataset(
                    "insulin",
                    data=insulin,
                    compression=compression,
                    compression_opts=compression_level if compression == "gzip" else None,
                )
            
            total_points += len(glucose)
            
            if (i + 1) % 100 == 0:
                print(f"[convert] Processed {i + 1}/{len(data_list)} series...")
    
    # Get file sizes
    input_size = Path(input_path).stat().st_size
    output_size = output_path.stat().st_size
    ratio = output_size / input_size * 100
    
    stats = {
        "num_series": len(data_list),
        "total_points": total_points,
        "input_size_mb": input_size / 1024 / 1024,
        "output_size_mb": output_size / 1024 / 1024,
        "compression_ratio": ratio,
    }
    
    print(f"[convert] Done!")
    print(f"[convert] Series: {stats['num_series']}")
    print(f"[convert] Total data points: {stats['total_points']:,}")
    print(f"[convert] Input size: {stats['input_size_mb']:.2f} MB")
    print(f"[convert] Output size: {stats['output_size_mb']:.2f} MB ({ratio:.1f}%)")
    
    return stats


def _normalize_data(raw_data) -> list:
    """Normalize various input formats to list of dicts."""
    if isinstance(raw_data, list) and len(raw_data) > 0:
        if isinstance(raw_data[0], dict):
            # Already in correct format
            return raw_data
        
        elif isinstance(raw_data[0], np.ndarray):
            # List of numpy arrays (univariate)
            return [
                {
                    "glucose": x.flatten().astype(np.float32),
                    "insulin": np.zeros_like(x.flatten(), dtype=np.float32),
                }
                for x in raw_data
            ]
        else:
            # Unknown format, try to convert
            return [
                {
                    "glucose": np.array(x).flatten().astype(np.float32),
                    "insulin": np.zeros_like(np.array(x).flatten(), dtype=np.float32),
                }
                for x in raw_data
            ]
    else:
        raise ValueError(f"Unsupported data format: {type(raw_data)}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PKL datasets to HDF5 format for lazy loading"
    )
    parser.add_argument("input", help="Input .pkl file path")
    parser.add_argument("output", help="Output .h5 file path")
    parser.add_argument(
        "--compression",
        choices=["gzip", "lzf", "none"],
        default="gzip",
        help="Compression algorithm (default: gzip)",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=4,
        help="Compression level for gzip (1-9, default: 4)",
    )
    
    args = parser.parse_args()
    
    compression = None if args.compression == "none" else args.compression
    
    try:
        convert_pkl_to_hdf5(
            args.input,
            args.output,
            compression=compression,
            compression_level=args.level,
        )
    except Exception as e:
        print(f"[error] Conversion failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
