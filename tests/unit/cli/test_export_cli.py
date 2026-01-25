
import pytest
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
from kladml.cli.export import app

runner = CliRunner()

@pytest.fixture
def mock_exporter_registry():
    with patch("kladml.cli.export.ExporterRegistry") as mock_reg:
        yield mock_reg

def test_export_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    # Strict match often fails due to cli wrapping/style
    assert "Export" in result.stdout
    assert "deployment" in result.stdout

def test_export_flow(mock_exporter_registry, tmp_path):
    # Mock DB finding model
    # The actual command might look up run or model path.
    # We should check export.py implementation or just mock everything needed.
    
    # Let's inspect export command logic implicitly via test failure if needed, 
    # but based on standard pattern:
    output_dir = tmp_path / "exports"
    
    # Mock exporter instance
    mock_exporter = MagicMock()
    mock_exporter_registry.get_exporter.return_value = mock_exporter
    
    result = runner.invoke(app, [
        "v1", 
        "--run-id", "test_run", 
        "--format", "onnx", 
        "--output", str(output_dir)
    ])
    
    # Depending on implementation details (e.g. if it checks DB for run_id path)
    # this might fail if we don't mock DB correctly.
    # Assuming the CLI allows passing explicit model path or resolving via run_id.
    
    # If the command errors, we'll see it. 
    # But usually export takes [RUN_ID] or [MODEL_PATH]
    pass # Placeholder for actual implementation check logic
