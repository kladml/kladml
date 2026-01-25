
import pytest
import numpy as np
import warnings
import pandas as pd
from pathlib import Path

from kladml.models.timeseries.transformer.gluformer.model import GluformerModel
from kladml.utils.paths import ensure_data_structure

# Suppress PyTorch/Accelerator warnings for clean output
@pytest.fixture(autouse=True)
def ignore_warnings():
    warnings.simplefilter("ignore")

@pytest.fixture
def dummy_glucose_data(tmp_path):
    """Create a dummy glucose dataset pickle."""
    # Create 200 samples of glucose data (sine wave)
    t = np.linspace(0, 100, 200)
    glucose = 100 + 40 * np.sin(t) + np.random.normal(0, 5, 200)
    
    # Needs to be a dataframe usually? Or GluformerDataModule handles lists?
    # GluformerDataModule expects: .pkl with pandas DataFrame OR keys 'glucose', 'insulin'
    # Let's check DataModule logic implicitly via failure or success.
    # Looking at datamodule source (not visible but assumed), usually expects DF.
    
    df = pd.DataFrame({"glucose": glucose, "insulin": np.zeros_like(glucose)})
    
    path = tmp_path / "glucose_data.pkl"
    df.to_pickle(path)
    return path

def test_gluformer_full_lifecycle(dummy_glucose_data, tmp_path):
    """
    Test the full lifecycle of a Gluformer model:
    1. Initialize
    2. Train (1 epoch, minimal data)
    3. Predict
    4. Save/Load
    """
    # 0. Setup Workspace (needed for project logging)
    # We patch paths to use tmp_path
    with pytest.MonkeyPatch.context() as m:
        m.setattr("kladml.utils.paths.get_root_data_path", lambda: tmp_path / "data")
        ensure_data_structure()
        
        # 1. Initialize
        config = {
            "project_name": "test_proj",
            "experiment_name": "integration_test",
            "epochs": 1,
            "batch_size": 16,
            "d_model": 16, # Tiny model for speed
            "n_heads": 2,
            "e_layers": 1,
            "d_layers": 1,
            "seq_len": 12,
            "pred_len": 4,
            "label_len": 6,
            "device": "cpu" # Force CPU for test environment
        }
        
        model = GluformerModel(config=config)
        
        assert model.config["d_model"] == 16
        
        # 2. Train
        metrics = model.train(X_train=str(dummy_glucose_data), X_val=str(dummy_glucose_data))
        
        assert "train_loss" in metrics or "loss" in metrics
        assert model.is_trained
        
        # 3. Predict
        input_seq = [100.0] * 12 # seq_len
        pred = model.predict(input_seq)
        
        assert len(pred["forecast"]) == 4 # pred_len
        assert "risk_assessment" in pred
        
        # 4. Save & Load
        save_path = tmp_path / "saved_model.pt"
        model.save(str(save_path))
        
        # Create new instance and load
        new_model = GluformerModel(config=config)
        new_model.load(str(save_path))
        
        assert new_model.is_trained
        
        # Verify prediction match
        pred_new = new_model.predict(input_seq)
        assert np.allclose(pred["forecast"], pred_new["forecast"], atol=1e-5)

def test_gluformer_export(dummy_glucose_data, tmp_path):
    """Test TorchScript export."""
    with pytest.MonkeyPatch.context() as m:
        m.setattr("kladml.utils.paths.get_root_data_path", lambda: tmp_path / "data")
        ensure_data_structure()
        
        config = {
             "project_name": "test_export",
             "experiment_name": "export_test",
             "epochs": 0, # Skip training, just init
             "d_model": 16,
             "device": "cpu"
        }
        model = GluformerModel(config=config)
        # Manually build model since we skipped train
        model.build_model()
        # Mock scaler to avoid None error during export if it requires it
        from sklearn.preprocessing import StandardScaler
        model._scaler = StandardScaler()
        model._scaler.fit(np.array([[100], [180]]))
        
        export_path = tmp_path / "model.pt"
        
        # Should succeed
        model.export_model(str(export_path), format="torchscript")
        
        assert export_path.exists()
        assert export_path.stat().st_size > 0
