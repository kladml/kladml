import pytest
from typer.testing import CliRunner
from kladml.cli.main import app

runner = CliRunner()

import os
from unittest.mock import patch

@pytest.fixture(autouse=True)
def clean_db(tmp_path):
    """Ensure each test runs with a fresh empty database."""
    db_file = tmp_path / "test_kladml.db"
    
    # 1. Point to temp DB
    with patch.dict(os.environ, {"KLADML_DB_PATH": str(db_file)}):
        # 2. Reset global state in session.py to force new engine creation
        from kladml.db import session as db_session
        db_session._engine = None
        db_session._session_factory = None
        
        # 3. Initialize tables
        db_session.init_db()
        
        yield
        
        # 4. Cleanup
        db_session._engine.dispose()
        db_session._engine = None
        db_session._session_factory = None

def test_registry_register_and_list(tmp_path):
    # 1. Create dummy file
    dummy_path = tmp_path / "model.pkl"
    dummy_path.touch()
    
    # 2. Register
    result = runner.invoke(app, [
        "registry", "register",
        "--name", "test_model",
        "--path", str(dummy_path),
        "--tag", "v1",
        "--tag", "production"
    ])
    assert result.exit_code == 0
    assert "Registered artifact test_model" in result.stdout
    
    # 3. List
    result = runner.invoke(app, ["registry", "list"])
    assert result.exit_code == 0
    assert "test_model" in result.stdout
    assert "production" in result.stdout
    
    # 4. List with filter
    result = runner.invoke(app, ["registry", "list", "--tag", "v1"])
    assert "test_model" in result.stdout
    
    result = runner.invoke(app, ["registry", "list", "--tag", "missing"])
    assert "test_model" not in result.stdout

def test_registry_show():
    # Pre-populate DB via CLI or directly
    # Let's use CLI to test integration
    runner.invoke(app, ["registry", "register", "--name", "show_me", "--path", ".", "--version", "v2"])
    
    result = runner.invoke(app, ["registry", "show", "show_me"])
    assert result.exit_code == 0
    assert "Artifact Details: show_me" in result.stdout
    assert "Version: v2" in result.stdout

def test_registry_register_duplicate(tmp_path):
    p = tmp_path / "f.txt"
    p.touch()
    runner.invoke(app, ["registry", "register", "--name", "dup", "--path", str(p)])
    
    # Register same name/version should fail
    result = runner.invoke(app, ["registry", "register", "--name", "dup", "--path", str(p)])
    assert result.exit_code != 0
    assert "already exists" in result.stdout
