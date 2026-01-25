
"""
Script to merge CGM UM (Glucose only) and Hall (Glucose + Insulin + Covariates) datasets.

Output:
    data/datasets/merged/
        train.parquet
        val.parquet
        metadata.json

Schema:
    Parquet with columns:
    - id (str)
    - source (str)
    - glucose (List[float32])
    - insulin (List[float32])
"""

import polars as pl
import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import sys

# Config
OUTPUT_DIR = Path("data/datasets/merged_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def process_cgm_um():
    """Process CGM UM dataset (Glucose only)."""
    print("Processing CGM UM dataset...")
    path = "data/datasets/glucose/cgm_um_train.pkl"
    if not Path(path).exists():
        print(f"❌ {path} not found")
        return []
        
    data = joblib.load(path)
    processed = []
    
    for i, seq in enumerate(tqdm(data, desc="CGM UM")):
        # seq is numpy array of glucose values
        if len(seq) < 60:
            continue
            
        processed.append({
            "glucose": seq.astype(np.float32).tolist(),
            "insulin": np.zeros(len(seq), dtype=np.float32).tolist(),  # No insulin data
            "id": f"um_{i}",
            "source": "cgm_um"
        })
        
    return processed


def process_hall():
    """Process Hall dataset (Glucose + Insulin)."""
    print("Processing Hall dataset...")
    path = "data/datasets/hall/hall.csv"
    if not Path(path).exists():
        print(f"❌ {path} not found")
        return []
        
    # Read relevant columns using Polars
    # Schema: id, time, gl, insulin
    try:
        df = pl.read_csv(path, columns=['id', 'time', 'gl', 'insulin'])
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return []
    
    processed = []
    
    # Group by ID
    # Sort by time? Assuming numeric/ordered? Or convert time?
    # If time is string, cast to datetime.
    
    # We'll iter over groups. Polars specific pattern: partition_by("id", as_dict=True)
    # or df.group_by("id")
    
    # We sort by id then time first?
    # Assuming standard CSV structure.
    
    # Sort entire DF usually faster than per group
    # df = df.sort(["id", "time"]) # Need to parse time if string
    
    # Iterate groups
    # Using partition_by is RAM heavy if many groups.
    # Hall dataset isn't huge usually.
    
    # Efficient:
    # 1. Fill nulls (clean)
    # 2. Agg to lists
    
    grouped = (
        df
        .with_columns([
            pl.col("gl").fill_null(strategy="forward").fill_null(strategy="backward").alias("gl"), # Interpolate support in aggregation?
            pl.col("insulin").fill_null(0.0)
        ])
        .group_by("id")
        .agg([
            pl.col("gl"),
            pl.col("insulin"),
            pl.col("source").first().alias("source") if "source" in df.columns else pl.lit("hall").alias("source")
        ])
    )
    
    # Convert to list of dicts or just keep DF?
    # We can mix with cgm_um.
    # CGM UM is list of dicts.
    
    # Iterate rows
    for row in tqdm(grouped.iter_rows(named=True), desc="Hall", total=len(grouped)):
        gl = row['gl']
        ins = row['insulin']
        sid = str(row['id'])
        
        # Check length
        if len(gl) < 60:
            continue
            
        # Ensure float32 lists
        # Polars handles this usually, but to match exactly
        processed.append({
            "glucose": [float(x) for x in gl],
            "insulin": [float(x) for x in ins],
            "id": sid,
            "source": "hall"
        })
        
    return processed


def main():
    # 1. Load Data
    um_data = process_cgm_um()
    hall_data = process_hall()
    
    all_data = um_data + hall_data
    print(f"\nTotal sequences: {len(all_data)}")
    print(f"  - CGM UM: {len(um_data)}")
    print(f"  - Hall: {len(hall_data)}")
    
    if not all_data:
        print("No data found.")
        return

    # 2. Split Train/Val
    train_data, val_data = train_test_split(all_data, test_size=0.1, random_state=42, shuffle=True)
    
    # 3. Save as Parquet
    print(f"\nSaving to {OUTPUT_DIR}...")
    
    # Create DataFrames
    df_train = pl.DataFrame(train_data)
    df_val = pl.DataFrame(val_data)
    
    df_train.write_parquet(OUTPUT_DIR / "train.parquet")
    df_val.write_parquet(OUTPUT_DIR / "val.parquet")
    
    # 4. Save metadata
    import json
    meta = {
        "sources": ["cgm_um", "hall"],
        "num_train": len(train_data),
        "num_val": len(val_data),
        "features": ["glucose", "insulin"],
        "description": "Merged dataset v1 (UM + Hall). Insulin padded with 0 for UM."
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
        
    print("✅ Done!")


if __name__ == "__main__":
    try:
        main()
    except ImportError:
        print("Polars required. pip install polars")
