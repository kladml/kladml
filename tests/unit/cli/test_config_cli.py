
import pytest
from typer.testing import CliRunner
from kladml.cli.main import app
from unittest.mock import patch

runner = CliRunner()

def test_config_create_help():
    result = runner.invoke(app, ["config", "create", "--help"])
    assert result.exit_code == 0
    assert "Generate a 'smart' configuration" in result.stdout

def test_config_create_basic(tmp_path):
    output = tmp_path / "config.yaml"
    
    # Mock generate_smart_config to avoid filesystem/inspection
    with patch("kladml.cli.commands.config.generate_smart_config") as mock_gen:
        mock_gen.return_value = {"project_name": "test", "d_model": 128}
        
        result = runner.invoke(app, ["config", "create", "--model", "gluformer", "--output", str(output)])
        
        assert result.exit_code == 0
        assert "Configuration saved" in result.stdout
        assert output.exists()
        
        # Check content wrapper logic
        import yaml
        content = yaml.safe_load(output.read_text())
        assert content["model"]["name"] == "gluformer"
        assert content["model"]["d_model"] == 128
        assert content["project"]["name"] == "test"
