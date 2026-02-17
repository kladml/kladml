"""
Tests for the evaluate CLI command.
"""

import pytest
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
from pathlib import Path
import torch

from kladml.cli.evaluate import app

runner = CliRunner()


@pytest.fixture
def mock_evaluator_registry():
    with patch("kladml.cli.evaluate.EvaluatorRegistry") as mock_reg:
        yield mock_reg


@pytest.fixture
def mock_evaluator():
    """Create a mock evaluator instance."""
    evaluator = MagicMock()
    evaluator.run.return_value = {"accuracy": 0.95, "f1": 0.92}
    return evaluator


@pytest.fixture
def dummy_model_path(tmp_path):
    """Create a dummy model file for testing."""
    model = torch.nn.Linear(4, 2)
    model_path = tmp_path / "model.pt"
    torch.save(model, model_path)
    return model_path


@pytest.fixture
def dummy_data_path(tmp_path):
    """Create dummy data file for testing."""
    data = (torch.randn(10, 4), torch.randint(0, 2, (10,)))
    data_path = tmp_path / "data.pt"
    torch.save(data, data_path)
    return data_path


def test_evaluate_help():
    """Test that --help works."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "evaluation" in result.stdout.lower() or "evaluate" in result.stdout.lower()


def test_evaluate_with_explicit_task(
    mock_evaluator_registry, mock_evaluator, dummy_model_path, dummy_data_path, tmp_path
):
    """Test evaluation with explicitly specified task."""
    mock_evaluator_class = MagicMock(return_value=mock_evaluator)
    mock_evaluator_registry.get_evaluator.return_value = mock_evaluator_class

    output_dir = tmp_path / "eval_output"

    result = runner.invoke(
        app,
        [
            "run",
            "--run-id",
            "test_run_001",
            "--model",
            str(dummy_model_path),
            "--data",
            str(dummy_data_path),
            "--task",
            "classification",
            "--output",
            str(output_dir),
        ],
    )

    # Check registry was queried
    mock_evaluator_registry.get_evaluator.assert_called_once()


def test_evaluate_regression_task(
    mock_evaluator_registry, mock_evaluator, dummy_model_path, dummy_data_path, tmp_path
):
    """Test evaluation with regression task."""
    mock_evaluator_class = MagicMock(return_value=mock_evaluator)
    mock_evaluator_registry.get_evaluator.return_value = mock_evaluator_class

    output_dir = tmp_path / "eval_output"

    result = runner.invoke(
        app,
        [
            "run",
            "--run-id",
            "test_regression",
            "--model",
            str(dummy_model_path),
            "--data",
            str(dummy_data_path),
            "--task",
            "regression",
            "--output",
            str(output_dir),
        ],
    )

    mock_evaluator_registry.get_evaluator.assert_called_once()


def test_evaluate_timeseries_task(
    mock_evaluator_registry, mock_evaluator, dummy_model_path, dummy_data_path, tmp_path
):
    """Test evaluation with timeseries_forecasting task."""
    mock_evaluator_class = MagicMock(return_value=mock_evaluator)
    mock_evaluator_registry.get_evaluator.return_value = mock_evaluator_class

    output_dir = tmp_path / "eval_output"

    result = runner.invoke(
        app,
        [
            "run",
            "--run-id",
            "test_timeseries",
            "--model",
            str(dummy_model_path),
            "--data",
            str(dummy_data_path),
            "--task",
            "timeseries_forecasting",
            "--output",
            str(output_dir),
        ],
    )

    mock_evaluator_registry.get_evaluator.assert_called_once()


def test_evaluate_invalid_task(mock_evaluator_registry, dummy_model_path, dummy_data_path, tmp_path):
    """Test evaluation with invalid task."""
    output_dir = tmp_path / "eval_output"

    result = runner.invoke(
        app,
        [
            "run",
            "--run-id",
            "test_invalid",
            "--model",
            str(dummy_model_path),
            "--data",
            str(dummy_data_path),
            "--task",
            "invalid_task_name",
            "--output",
            str(output_dir),
        ],
    )

    assert result.exit_code != 0


def test_evaluate_missing_required_args():
    """Test that missing required args fails."""
    result = runner.invoke(app, ["run"])
    assert result.exit_code != 0


def test_evaluate_auto_detect_task(
    mock_evaluator_registry, mock_evaluator, dummy_model_path, dummy_data_path, tmp_path
):
    """Test auto-detection of task (currently defaults to classification)."""
    mock_evaluator_class = MagicMock(return_value=mock_evaluator)
    mock_evaluator_registry.get_evaluator.return_value = mock_evaluator_class

    output_dir = tmp_path / "eval_output"

    result = runner.invoke(
        app,
        [
            "run",
            "--run-id",
            "test_auto",
            "--model",
            str(dummy_model_path),
            "--data",
            str(dummy_data_path),
            "--output",
            str(output_dir),
        ],
    )

    # Without --task, should default to classification
    mock_evaluator_registry.get_evaluator.assert_called_once()
