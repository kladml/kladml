
from kladml.config.settings import KladMLSettings

class TestKladMLSettings:
    """Test functionality of Pydantic-based settings."""

    def test_defaults(self):
        """Test default values."""
        # We instantiate a fresh one to avoid pollution from global singleton
        settings = KladMLSettings()
        assert settings.api_url == "http://localhost:8001/api/v1"
        assert settings.debug is False
        assert settings.device == "auto"
        # Check computed defaults
        assert settings.database_url.startswith("sqlite:///")
        assert settings.mlflow_tracking_uri.startswith("sqlite:///")

    def test_env_override(self, monkeypatch):
        """Test environment variable overrides."""
        monkeypatch.setenv("KLADML_DEBUG", "true")
        monkeypatch.setenv("KLADML_DEVICE", "cuda")
        
        settings = KladMLSettings()
        assert settings.debug is True
        assert settings.device == "cuda"

    def test_yaml_loading(self, tmp_path, monkeypatch):
        """Test loading from YAML file."""
        # Create a dummy yaml
        config_data = """
debug: true
storage_bucket: "yaml-bucket"
executor: "local"
"""
        yaml_file = tmp_path / "kladml.yaml"
        yaml_file.write_text(config_data)

        # Switch to tmp dir so settings picks up kladml.yaml
        monkeypatch.chdir(tmp_path)
        
        settings = KladMLSettings()
        assert settings.debug is True
        assert settings.storage_bucket == "yaml-bucket"
        assert settings.executor == "local"

    def test_priority_order(self, tmp_path, monkeypatch):
        """Test that Env Vars > YAML > Defaults."""
        # YAML says debug=False
        yaml_file = tmp_path / "kladml.yaml"
        yaml_file.write_text("debug: false\ndevice: cpu")
        monkeypatch.chdir(tmp_path)

        # Env says debug=True
        monkeypatch.setenv("KLADML_DEBUG", "true")

        settings = KladMLSettings()
        # Env wins for debug
        assert settings.debug is True
        # YAML wins for device (default is auto)
        assert settings.device == "cpu"
