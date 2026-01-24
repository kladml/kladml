"""
Tests for local data management.
"""

import pytest
import os
import shutil
import tempfile
from pathlib import Path
from typer.testing import CliRunner
from kladml.cli.main import app
from kladml.utils.paths import (
    ensure_data_structure, 
    resolve_dataset_path, 
    resolve_preprocessor_path,
    DATA_DIR, DATASETS_DIR, PREPROCESSORS_DIR
)

@pytest.fixture
def workspace():
    """Create a temp workspace."""
    cwd = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    os.chdir(temp_dir)
    yield Path(temp_dir)
    os.chdir(cwd)
    shutil.rmtree(temp_dir)

def test_ensure_data_structure(workspace):
    """Test directory creation."""
    root = ensure_data_structure()
    assert root.exists()
    assert (root / "datasets").exists()
    assert (root / "preprocessors").exists()
    assert (root / "registry").exists()
    assert (root / "projects").exists()

def test_resolve_paths(workspace):
    """Test path resolution logic."""
    ensure_data_structure()
    
    # 1. Absolute path -> returns as is
    abs_path = (workspace / "abs_data.csv").resolve()
    assert resolve_dataset_path(str(abs_path)) == abs_path
    
    # 2. Existing relative path -> returns resolved relative path
    rel_file = workspace / "local.csv"
    rel_file.touch()
    assert resolve_dataset_path("local.csv") == rel_file.resolve()
    
    # 3. Non-existent path -> maps to data/datasets/NAME
    expected = (workspace / DATA_DIR / DATASETS_DIR / "my_dataset").resolve()
    assert resolve_dataset_path("my_dataset").resolve() == expected
    
    # 4. Preprocessor logic (same)
    expected_prep = (workspace / DATA_DIR / PREPROCESSORS_DIR / "prep.py").resolve()
    assert resolve_preprocessor_path("prep.py").resolve() == expected_prep

def test_init_command(workspace):
    """Test 'kladml init' command."""
    runner = CliRunner()
    
    # Run init
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert "initialized at" in result.stdout
    assert (workspace / "data" / "datasets").exists()
    
    # Run again (should warn/info without force)
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert "already initialized" in result.stdout
    
    # Run with force
    result = runner.invoke(app, ["init", "--force"])
    assert result.exit_code == 0
    assert "initialized at" in result.stdout
