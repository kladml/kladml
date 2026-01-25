
import pytest
from kladml.utils.config_io import load_yaml_config
import yaml

def test_load_yaml_valid(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("foo: bar\nlist: [1, 2]")
    
    cfg = load_yaml_config(str(f))
    assert cfg["foo"] == "bar"
    assert cfg["list"] == [1, 2]

def test_load_yaml_empty(tmp_path):
    f = tmp_path / "empty.yaml"
    f.touch()
    
    cfg = load_yaml_config(str(f))
    assert cfg == {}

def test_load_yaml_missing():
    with pytest.raises(FileNotFoundError):
        load_yaml_config("nonexistent.yaml")

def test_load_yaml_invalid(tmp_path):
    f = tmp_path / "bad.yaml"
    # Tabs are forbidden in YAML
    f.write_text("key: value\n\t- tabbed list")
    
    with pytest.raises(yaml.YAMLError):
        load_yaml_config(str(f))
