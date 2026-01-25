
import pytest
import typer
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
from kladml.cli.compare import compare_runs

runner = CliRunner()

from kladml.cli.main import app

runner = CliRunner()


@pytest.fixture
def mock_tracker():
    with patch("kladml.cli.compare.get_tracker_backend") as mock_get:
        mock = MagicMock()
        mock_get.return_value = mock
        yield mock

def test_compare_less_than_two_runs(mock_tracker):
    # Pass params via args
    result = runner.invoke(app, ["compare", "--runs", "run1"])
    assert result.exit_code == 0, f"Exit code {result.exit_code}, Output: {result.stdout}"
    assert "Warning: Comparing less than 2 runs" in result.stdout

def test_compare_run_not_found(mock_tracker):
    mock_tracker.get_run.return_value = None
    result = runner.invoke(app, ["compare", "--runs", "missing_run,other"])
    assert result.exit_code == 1, f"Output: {result.stdout}"
    assert "not found" in result.stdout

def test_compare_success(mock_tracker):
    # Mock data
    run1 = {"run_name": "R1", "metrics": {"loss": 0.5, "acc": 0.8}, "params": {"lr": 0.01}}
    run2 = {"run_name": "R2", "metrics": {"loss": 0.4, "acc": 0.85}, "params": {"lr": 0.02}}
    
    def get_run_side_effect(rid):
        if rid == "r1": return run1
        if rid == "r2": return run2
        return None
    
    mock_tracker.get_run.side_effect = get_run_side_effect
    
    result = runner.invoke(app, ["compare", "--runs", "r1,r2"])
    
    assert result.exit_code == 0, f"Output: {result.stdout}"
    assert "0.5000" in result.stdout
    assert "0.4000" in result.stdout

def test_compare_filter_metrics(mock_tracker):
    run1 = {"run_name": "R1", "metrics": {"loss": 0.5, "acc": 0.8}, "params": {}}
    mock_tracker.get_run.return_value = run1
    
    result = runner.invoke(app, ["compare", "--runs", "r1,r1", "--metrics", "loss"])
    
    assert result.exit_code == 0, f"Output: {result.stdout}"
    assert "loss" in result.stdout
    assert "acc" not in result.stdout
