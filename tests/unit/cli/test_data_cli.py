
import pytest
from typer.testing import CliRunner
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
from kladml.cli.commands.data.core import app

runner = CliRunner()

# --- Inspect Tests ---


def test_inspect_pkl_list_of_arrays(tmp_path):
    """Test inspecting a PKL containing list of arrays (TimeSeries)."""
    p = tmp_path / "ts.pkl"
    p.touch()
    
    import numpy as np
    dummy_data = [np.zeros((10, 1)) for _ in range(5)]
    
    with patch("kladml.cli.commands.data.inspect.joblib.load") as mock_load:
        mock_load.return_value = dummy_data
        
        result = runner.invoke(app, ["inspect", str(p)])
        assert result.exit_code == 0
        assert "Type: timeseries_list" in result.stdout
        assert "Total Samples" in result.stdout

def test_inspect_pkl_numpy(tmp_path):
    """Test inspecting a PKL containing numpy array."""
    p = tmp_path / "arr.pkl"
    p.touch()
    
    import numpy as np
    dummy_data = np.zeros((10, 5))
    
    with patch("kladml.cli.commands.data.inspect.joblib.load") as mock_load:
        mock_load.return_value = dummy_data
        
        result = runner.invoke(app, ["inspect", str(p)])
        assert result.exit_code == 0
        assert "Type: numpy_array" in result.stdout
        assert "Shape" in result.stdout

def test_inspect_pkl_dict(tmp_path):
    """Test inspecting a PKL containing dict."""
    p = tmp_path / "dict.pkl"
    p.touch()
    
    dummy_data = {"key1": 1, "key2": 2}
    
    with patch("kladml.cli.commands.data.inspect.joblib.load") as mock_load:
        mock_load.return_value = dummy_data
        
        result = runner.invoke(app, ["inspect", str(p)])
        assert result.exit_code == 0
        assert "Type: dictionary" in result.stdout
        assert "key1" in result.stdout

def test_inspect_pkl_sklearn_scaler(tmp_path):
    """Test inspecting a sklearn-like scaler."""
    p = tmp_path / "scaler.pkl"
    p.touch()
    
    # Mock object with mean_ and scale_
    mock_scaler = MagicMock()
    mock_scaler.mean_ = [0.5]
    mock_scaler.scale_ = [2.0]
    
    with patch("kladml.cli.commands.data.inspect.joblib.load") as mock_load:
        mock_load.return_value = mock_scaler
        
        result = runner.invoke(app, ["inspect", str(p)])
        assert result.exit_code == 0
        assert "Type: sklearn_scaler" in result.stdout
        assert "0.5000" in result.stdout

def test_inspect_corrupt_file(tmp_path):
    """Test inspecting a corrupt file."""
    p = tmp_path / "bad.pkl"
    p.touch()
    
    with patch("kladml.cli.commands.data.inspect.joblib.load") as mock_load:
        mock_load.side_effect = ValueError("Corrupt data")
        
        result = runner.invoke(app, ["inspect", str(p)])
        
        # It catches exception in analyze_pkl and returns classification='corrupt'
        # core.py prints the table with Error row?
        # Wait, analyze_pkl catches exception and returns dict with 'error' key.
        # core.py prints table with error row.
        
        assert result.exit_code == 0 # It doesn't crash app
        assert "Type: corrupt" in result.stdout
        assert "Corrupt data" in result.stdout

def test_inspect_parquet(tmp_path):
    """Test inspecting a valid .parquet file."""
    p = tmp_path / "test.parquet"
    p.touch()
    
    # We need to mock pl.scan_parquet
    # inspect.py imports polars inside analyze_parquet
    
    with patch("kladml.cli.commands.data.inspect.analyze_parquet") as mock_analyze: # Still mock this for now, Parquet requires Polars installed
         # We can try to test logic if we mock polars module
         pass
         
    # Let's mock polars in sys.modules
    mock_pl = MagicMock()
    mock_lf = MagicMock()
    mock_pl.scan_parquet.return_value = mock_lf
    mock_lf.schema = {"col1": "Int64"}
    mock_lf.select.return_value.collect.return_value.item.return_value = 100
    
    with patch.dict("sys.modules", {"polars": mock_pl}):
        # We invoke app.
        result = runner.invoke(app, ["inspect", str(p)])
        assert result.exit_code == 0
        assert "Type: tabular" in result.stdout
        assert "col1" in result.stdout

# --- Summary Tests ---

def test_summary_directory(tmp_path):
    """Test summary command on directory."""
    (tmp_path / "f1.pkl").touch()
    
    import numpy as np
    dummy_data = np.zeros((5,))
    
    with patch("kladml.cli.commands.data.inspect.joblib.load") as mock_load:
        mock_load.return_value = dummy_data
        
        result = runner.invoke(app, ["summary", str(tmp_path)])
        assert result.exit_code == 0
        assert "Datasets in" in result.stdout
        assert "f1.pkl" in result.stdout

# --- Convert Tests ---

def test_convert_pkl_to_hdf5(tmp_path):
    """Test conversion command."""
    input_p = tmp_path / "in.pkl"
    input_p.touch()
    output_p = tmp_path / "out.h5"
    
    with patch("kladml.data.converter.convert_pkl_to_hdf5") as mock_convert:
        result = runner.invoke(app, [
            "convert", 
            "-i", str(input_p),
            "-o", str(output_p),
            "-f", "hdf5"
        ])
        assert result.exit_code == 0
        assert "Conversion successful" in result.stdout
        mock_convert.assert_called_once()


def test_convert_unsupported_format(tmp_path):
    p = tmp_path / "in.pkl"
    result = runner.invoke(app, ["convert", "-i", str(p), "-o", "out", "-f", "csv"])
    assert result.exit_code == 1
    assert "Supported formats" in result.stdout

