
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch
from kladml.models.timeseries.transformer.gluformer.datamodule import GluformerDataModule

@pytest.fixture
def dummy_data_list():
    # List of 100 series, each length 100
    return [np.random.randn(100) for _ in range(10)]

def test_datamodule_setup_memory(dummy_data_list):
    dm = GluformerDataModule(
        train_path=dummy_data_list,
        val_path=None,
        seq_len=10,
        pred_len=5,
        label_len=5,
        batch_size=4
    )
    
    dm.setup()
    
    assert dm.train_dataset is not None
    assert len(dm.train_dataset) > 0
    assert dm.scaler.mean_ is not None # Scaler should be fit

def test_datamodule_dataloaders(dummy_data_list):
    dm = GluformerDataModule(
        train_path=dummy_data_list,
        batch_size=2
    )
    dm.setup()
    
    loader = dm.train_dataloader()
    assert isinstance(loader, torch.utils.data.DataLoader)
    
    batch = next(iter(loader))
    # Batch keys: x_enc, x_dec, x_id, y
    assert "x_enc" in batch
    assert "y" in batch
    assert batch["x_enc"].shape[0] == 2
    
def test_datamodule_val_loader_none(dummy_data_list):
    dm = GluformerDataModule(train_path=dummy_data_list, val_path=None)
    dm.setup()
    assert dm.val_dataloader() is None

@patch("h5py.File")
def test_datamodule_setup_hdf5(mock_h5):
    # Mock HDF5 interactions to test logic without real file
    # dm checks: isinstance(str) and endswith .h5
    
    dm = GluformerDataModule(
        train_path="fake.h5",
        seq_len=10
    )
    
    # Mock file context manager
    mock_file = MagicMock()
    mock_h5.return_value.__enter__.return_value = mock_file
    
    # Mock metadata
    mock_file.__contains__.return_value = True # 'metadata' in f
    mock_attrs = {"scaler_mean": 10.0, "scaler_scale": 2.0}
    mock_file.__getitem__.return_value.attrs = mock_attrs # f['metadata'].attrs
    
    # Mock Series keys
    # Wait, HDF5GluformerDataset needs real file usually or we mock dataset class
    # To avoid mocking HDF5GluformerDataset completely, we would need real file.
    # But for coverage of `setup`, we just need to verify it enters the if is_hdf5 block
    # and initializes HDF5GluformerDataset.
    
    with patch("kladml.models.timeseries.transformer.gluformer.datamodule.HDF5GluformerDataset") as mock_ds:
        dm.setup()
        
        # Verify scaler loaded from metadata
        assert dm.scaler.mean_[0] == 10.0
        assert dm.scaler.scale_[0] == 2.0
        
        # Verify dataset created
        mock_ds.assert_called_once()    
