"""
Script to merge CGM UM (Glucose only) and Hall (Glucose + Insulin + Covariates) datasets.

Output:
    data/datasets/merged/
        train.pkl
        val.pkl
        metadata.json

Schema:
    List of dictionaries:
    [
        {
            "glucose": np.array([...]),  # mg/dL
            "insulin": np.array([...]),  # Units (0 if missing)
            "id": "subject_id",
            "source": "cgm_um" | "hall"
        },
        ...
    ]
"""

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Config
OUTPUT_DIR = Path("data/datasets/merged_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def process_cgm_um():
    """Process CGM UM dataset (Glucose only)."""
    print("Processing CGM UM dataset...")
    path = "data/datasets/glucose/cgm_um_train.pkl"
    if not os.path.exists(path):
        print(f"❌ {path} not found")
        return []
        
    data = joblib.load(path)
    processed = []
    
    for i, seq in enumerate(tqdm(data, desc="CGM UM")):
        # seq is numpy array of glucose values
        if len(seq) < 60:
            continue
            
        processed.append({
            "glucose": seq.astype(np.float32),
            "insulin": np.zeros_like(seq, dtype=np.float32),  # No insulin data
            "id": f"um_{i}",
            "source": "cgm_um"
        })
        
    return processed


def process_hall():
    """Process Hall dataset (Glucose + Insulin)."""
    print("Processing Hall dataset...")
    path = "data/datasets/hall/hall.csv"
    if not os.path.exists(path):
        print(f"❌ {path} not found")
        return []
        
    # Read relevant columns
    df = pd.read_csv(path, usecols=['id', 'time', 'gl', 'insulin'])
    
    # Convert time
    # Hall time format usually: "2015-01-01 12:00:00"
    # Or sometimes relative. Let's inspect.
    # Assuming ordered by time per ID for now.
    
    processed = []
    
    for subject_id, group in tqdm(df.groupby('id'), desc="Hall"):
        # Sort by time just in case
        # group = group.sort_values('time') 
        
        # Get values
        gl = group['gl'].values
        ins = group['insulin'].fillna(0).values
        
        # Handle NaN glucose
        # Linear interpolation
        gl = pd.Series(gl).interpolate(limit_direction='both').values
        
        if len(gl) < 60:
            continue
            
        processed.append({
            "glucose": gl.astype(np.float32),
            "insulin": ins.astype(np.float32),
            "id": str(subject_id),
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
    
    # 2. Split Train/Val
    train_data, val_data = train_test_split(all_data, test_size=0.1, random_state=42, shuffle=True)
    
    # 3. Save
    print(f"\nSaving to {OUTPUT_DIR}...")
    joblib.dump(train_data, OUTPUT_DIR / "train.pkl")
    joblib.dump(val_data, OUTPUT_DIR / "val.pkl")
    
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
    main()
