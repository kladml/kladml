
import pytest
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
from kladml.cli.run import app
from pathlib import Path

runner = CliRunner()

@pytest.fixture
def mock_subprocess():
    with patch("subprocess.Popen") as mock_popen, \
         patch("subprocess.run") as mock_run:
        yield mock_popen, mock_run

@pytest.fixture
def mock_shutil():
    with patch("shutil.which") as mock_which:
        yield mock_which

@pytest.fixture
def mock_fs(tmp_path):
    # Create dummy script
    script = tmp_path / "train.py"
    script.touch()
    return script

def test_run_local_docker_success(mock_subprocess, mock_shutil, mock_fs):
    mock_popen, mock_run = mock_subprocess
    mock_shutil.return_value = "/usr/bin/docker"
    
    # Mock Popen process
    process_mock = MagicMock()
    process_mock.stdout = ["Log line 1\n", "Log line 2\n"]
    process_mock.returncode = 0
    process_mock.wait.return_value = None
    mock_popen.return_value = process_mock
    
    result = runner.invoke(app, [
        "local", 
        str(mock_fs),
        "--runtime", "auto"
    ])
    
    assert result.exit_code == 0
    assert "Running with Docker" in result.stdout
    assert "Log line 1" in result.stdout
    
    # Verify docker run command structure
    args = mock_popen.call_args[0][0]
    assert args[0] == "docker"
    assert args[1] == "run"
    assert str(mock_fs) in args

def test_run_native_success(mock_subprocess, mock_fs):
    mock_popen, _ = mock_subprocess
    
    process_mock = MagicMock()
    process_mock.stdout = ["Native Log\n"]
    process_mock.returncode = 0
    mock_popen.return_value = process_mock

    # Mock tracker (patched at source because imported inside function)
    with patch("kladml.backends.get_tracker_backend") as mock_get_track:
        mock_track_instance = MagicMock()
        mock_get_track.return_value = mock_track_instance
        mock_track_instance.create_experiment.return_value = "123"
        
        result = runner.invoke(app, ["native", str(mock_fs), "--experiment", "default"])
        
        # Check output for debug
        if result.exit_code != 0:
            print(result.stdout)
            
        assert result.exit_code == 0
        assert "Running natively" in result.stdout
        # Ensure it calls python executable
        args = mock_popen.call_args[0][0]
        # args[0] should be python executable path
        assert "python" in str(args[0])
        assert str(mock_fs) in args

def test_run_script_not_found():
    result = runner.invoke(app, ["local", "missing.py"])
    assert result.exit_code == 1
    assert "Script not found" in result.stdout
