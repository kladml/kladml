"""Tests for export CLI command."""

import pytest
import json
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock

from kladml.cli.export import app, _load_metadata, _get_model_config

runner = CliRunner()


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_load_metadata_exists(self, tmp_path):
        """Test loading metadata from existing file."""
        metadata_file = tmp_path / "checkpoints" / "metadata.json"
        metadata_file.parent.mkdir(parents=True)
        metadata_file.write_text(json.dumps({"task": "timeseries", "seq_len": 60}))

        checkpoint_path = tmp_path / "checkpoints" / "best_model.pt"
        checkpoint_path.touch()

        result = _load_metadata(checkpoint_path)
        assert result["task"] == "timeseries"
        assert result["seq_len"] == 60

    def test_load_metadata_missing(self, tmp_path):
        """Test loading metadata when file doesn't exist."""
        checkpoint_path = tmp_path / "checkpoints" / "best_model.pt"
        checkpoint_path.parent.mkdir(parents=True)
        checkpoint_path.touch()

        result = _load_metadata(checkpoint_path)
        assert result == {}

    def test_load_metadata_invalid_json(self, tmp_path):
        """Test loading metadata with invalid JSON."""
        metadata_file = tmp_path / "checkpoints" / "metadata.json"
        metadata_file.parent.mkdir(parents=True)
        metadata_file.write_text("not valid json")

        checkpoint_path = tmp_path / "checkpoints" / "best_model.pt"
        checkpoint_path.touch()

        result = _load_metadata(checkpoint_path)
        assert result == {}

    def test_get_model_config_from_checkpoint(self):
        """Test config extraction from checkpoint data."""
        ckpt_data = {"config": {"seq_len": 100, "pred_len": 24}}
        result = _get_model_config(ckpt_data, {}, None)
        assert result["seq_len"] == 100
        assert result["pred_len"] == 24

    def test_get_model_config_from_metadata(self):
        """Test config override from metadata."""
        ckpt_data = {"config": {"seq_len": 100}}
        metadata = {"seq_len": 200, "enc_in": 10}
        result = _get_model_config(ckpt_data, metadata, None)
        assert result["seq_len"] == 200  # Overridden
        assert result["enc_in"] == 10

    def test_get_model_config_from_cli(self, tmp_path):
        """Test config override from CLI config file."""
        ckpt_data = {"config": {"seq_len": 100}}
        metadata = {"seq_len": 200}

        config_file = tmp_path / "config.yaml"
        config_file.write_text("seq_len: 300\npred_len: 48")

        result = _get_model_config(ckpt_data, metadata, config_file)
        assert result["seq_len"] == 300  # CLI wins
        assert result["pred_len"] == 48


class TestExportCLI:
    """Tests for export CLI command."""

    def test_export_help(self):
        """Test help output."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Export" in result.stdout

    def test_export_missing_checkpoint(self):
        """Test error when checkpoint doesn't exist."""
        result = runner.invoke(app, [
            "--checkpoint", "/nonexistent/path.pt",
            "--format", "torchscript"
        ])
        assert result.exit_code != 0
        assert "not found" in result.stdout.lower()

    def test_export_invalid_format(self, tmp_path):
        """Test error for unsupported format."""
        checkpoint = tmp_path / "model.pt"
        checkpoint.touch()

        result = runner.invoke(app, [
            "--checkpoint", str(checkpoint),
            "--format", "invalid_format"
        ])
        assert result.exit_code != 0


class TestScalerWrapper:
    """Tests for ScalerWrapper in torchscript exporter."""

    def test_scaler_wrapper_standard_scaler(self):
        """Test ScalerWrapper with StandardScaler-like object."""
        import torch
        from kladml.exporters.torchscript import ScalerWrapper

        # Mock StandardScaler
        class MockScaler:
            mean_ = [0.5, 1.0, 1.5]
            scale_ = [0.1, 0.2, 0.3]

        model = torch.nn.Linear(3, 2)
        wrapper = ScalerWrapper(model, MockScaler())

        # Check buffers registered
        assert hasattr(wrapper, 'mean')
        assert hasattr(wrapper, 'scale')
        assert wrapper.mean.shape == (3,)
        assert wrapper.scale.shape == (3,)

    def test_scaler_wrapper_minmax_scaler(self):
        """Test ScalerWrapper with MinMaxScaler-like object."""
        import torch
        from kladml.exporters.torchscript import ScalerWrapper

        # Mock MinMaxScaler
        class MockScaler:
            min_ = [0.0, 0.0]
            scale_ = [1.0, 1.0]

        model = torch.nn.Linear(2, 1)
        wrapper = ScalerWrapper(model, MockScaler())

        assert hasattr(wrapper, 'mean')
        assert hasattr(wrapper, 'scale')

    def test_scaler_wrapper_forward(self):
        """Test forward pass applies normalization."""
        import torch
        from kladml.exporters.torchscript import ScalerWrapper

        class MockScaler:
            mean_ = [1.0]
            scale_ = [2.0]

        # Identity model
        model = torch.nn.Identity()
        wrapper = ScalerWrapper(model, MockScaler())

        input_tensor = torch.tensor([[5.0]])
        output = wrapper(input_tensor)

        # Expected: (5.0 - 1.0) / 2.0 = 2.0
        expected = torch.tensor([[2.0]])
        assert torch.allclose(output, expected)


class TestTorchScriptExporter:
    """Tests for TorchScriptExporter."""

    def test_export_with_scaler(self, tmp_path):
        """Test export includes scaler when provided."""
        import torch
        from kladml.exporters.torchscript import TorchScriptExporter

        class MockScaler:
            mean_ = [0.0, 0.0]
            scale_ = [1.0, 1.0]

        model = torch.nn.Linear(2, 1)
        exporter = TorchScriptExporter()
        output_path = str(tmp_path / "model.pt")

        result = exporter.export(
            model,
            output_path,
            input_sample=torch.randn(1, 2),
            scaler=MockScaler()
        )

        assert Path(result).exists()

        # Load and verify
        loaded = torch.jit.load(result)
        assert loaded is not None

    def test_export_without_scaler(self, tmp_path):
        """Test export without scaler."""
        import torch
        from kladml.exporters.torchscript import TorchScriptExporter

        model = torch.nn.Linear(3, 2)
        exporter = TorchScriptExporter()
        output_path = str(tmp_path / "model.pt")

        result = exporter.export(
            model,
            output_path,
            input_sample=torch.randn(1, 3)
        )

        assert Path(result).exists()

    def test_validate_success(self, tmp_path):
        """Test validation passes for valid model."""
        import torch
        from kladml.exporters.torchscript import TorchScriptExporter

        model = torch.nn.Linear(2, 1)
        exporter = TorchScriptExporter()
        output_path = str(tmp_path / "model.pt")

        exporter.export(model, output_path, input_sample=torch.randn(1, 2))

        is_valid = exporter.validate(output_path, input_sample=torch.randn(1, 2))
        assert is_valid is True

    def test_validate_failure(self, tmp_path):
        """Test validation fails for corrupted model."""
        from kladml.exporters.torchscript import TorchScriptExporter

        # Write invalid data
        invalid_path = tmp_path / "invalid.pt"
        invalid_path.write_text("not a valid torchscript model")

        exporter = TorchScriptExporter()
        is_valid = exporter.validate(str(invalid_path))
        assert is_valid is False
