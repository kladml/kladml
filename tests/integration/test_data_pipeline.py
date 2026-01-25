
import pytest
import numpy as np
import pandas as pd
import joblib
import h5py
from pathlib import Path
import os

from kladml.data.converter import convert_pkl_to_hdf5, convert_pkl_to_parquet
from kladml.data.hdf5_dataset import HDF5GluformerDataset

@pytest.fixture
def dummy_pkl_data(tmp_path):
    """Create a dummy pickle file with list of dicts."""
    data = []
    for i in range(5):
        # Create series of length 100
        # Sine wave + noise
        t = np.linspace(0, 50, 100)
        glucose = 100 + 40 * np.sin(t)
        insulin = np.random.rand(100)
        
        data.append({
            "glucose": glucose,
            "insulin": insulin
        })
        
    path = tmp_path / "input_data.pkl"
    joblib.dump(data, path)
    return path

@pytest.fixture
def dummy_pkl_numpy_list(tmp_path):
    """Create a dummy pickle with list of numpy arrays (univariate)."""
    data = []
    for i in range(3):
        t = np.linspace(0, 50, 100)
        glucose = 100 + 40 * np.sin(t)
        data.append(glucose)
        
    path = tmp_path / "input_numpy.pkl"
    joblib.dump(data, path)
    return path

def test_conversion_hdf5_full_flow(dummy_pkl_data, tmp_path):
    """
    Test PKL -> HDF5 conversion and then HDF5Dataset Reading.
    """
    output_h5 = tmp_path / "output.h5"
    
    # 1. Run Conversion
    stats = convert_pkl_to_hdf5(
        input_path=str(dummy_pkl_data),
        output_path=str(output_h5),
        compression="gzip",
        compression_level=1
    )
    
    assert output_h5.exists()
    assert stats["num_series"] == 5
    assert stats["total_points"] == 500 # 5 series * 100 points
    
    with h5py.File(output_h5, 'r') as f:
        assert "metadata" in f
        assert f["metadata"].attrs["num_series"] == 5
        assert "series" in f
        assert "0" in f["series"]
        assert "glucose" in f["series"]["0"]
        assert "insulin" in f["series"]["0"]
        
    # 3. Test HDF5Dataset (Reader)
    dataset = HDF5GluformerDataset(
        h5_path=str(output_h5),
        input_chunk_length=20,
        output_chunk_length=5,
        label_len=10,
        c_in=2 # Multivariate
    )
    
    assert len(dataset) > 0
    # Length calculation:
    # Per series: 100 points. Window size = 20 + 5 = 25.
    # Windows per series = 100 - 25 + 1 = 76.
    # Total = 5 * 76 = 380.
    assert len(dataset) == 380
    
    # Test __getitem__
    item = dataset[0]
    assert "x_enc" in item
    assert "x_dec" in item
    assert "y" in item
    assert item["x_enc"].shape == (20, 2) # [SeqLen, Channels]
    assert item["y"].shape == (25-20, 1) # [PredLen, 1] (Target is usually univariate glucose)
    
    # Verify values match source roughly (check first point)
    # Re-load source
    source = joblib.load(dummy_pkl_data)
    first_glucose = source[0]["glucose"][0]
    
    # HDF5 dataset x_enc[0, 0] should be first glucose
    assert np.isclose(item["x_enc"][0, 0].item(), first_glucose, atol=1e-5)

def test_conversion_numpy_legacy(dummy_pkl_numpy_list, tmp_path):
    """Test backward compatibility with list of numpy arrays."""
    output_h5 = tmp_path / "numpy_out.h5"
    
    stats = convert_pkl_to_hdf5(
        input_path=str(dummy_pkl_numpy_list),
        output_path=str(output_h5)
    )
    
    assert stats["num_series"] == 3
    
    # Verify Dataset can read it as univariate
    dataset = HDF5GluformerDataset(
        h5_path=str(output_h5),
        c_in=1
    )
    
    item = dataset[0]
    assert item["x_enc"].shape == (60, 1) # Default input len 60

def test_parquet_conversion(dummy_pkl_data, tmp_path):
    """Test Parquet conversion."""
    output_parquet = tmp_path / "data.parquet"
    
    stats = convert_pkl_to_parquet(
        input_path=str(dummy_pkl_data),
        output_path=str(output_parquet)
    )
    
    assert output_parquet.exists()
    assert stats["num_series"] == 5
    
    # Verify with Polars or Pandas if installed
    try:
        df = pd.read_parquet(output_parquet)
        assert len(df) == 5
        assert "glucose" in df.columns
        assert "insulin" in df.columns
    except ImportError:
        pass
