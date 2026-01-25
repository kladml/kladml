
import pytest
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
from kladml.cli.registry import app
from kladml.db.models import RegistryArtifact
from datetime import datetime

runner = CliRunner()

@pytest.fixture
def mock_session():
    with patch("kladml.cli.registry.get_session") as mock_get_session:
        session = MagicMock()
        mock_get_session.return_value.__enter__.return_value = session
        yield session

def test_registry_list_empty(mock_session):
    mock_session.exec.return_value.all.return_value = []
    
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "No artifacts found" in result.stdout

def test_registry_list_items(mock_session):
    # Mock artifacts
    art = RegistryArtifact(
        name="test-model",
        version="v1",
        artifact_type="model",
        path="/tmp/model.pt",
        tags=["best"],
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    art.id = 1
    
    mock_session.exec.return_value.all.return_value = [art]
    
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "test-model" in result.stdout
    assert "v1" in result.stdout
    assert "best" in result.stdout

def test_registry_add_success(mock_session, tmp_path):
    # Create dummy file
    f = tmp_path / "model.pt"
    f.touch()
    
    # Mock no duplicate
    mock_session.exec.return_value.first.return_value = None
    
    result = runner.invoke(app, [
        "add", 
        "--name", "new-model", 
        "--path", str(f), 
        "--version", "v1"
    ])
    
    assert result.exit_code == 0
    assert "Registered artifact new-model" in result.stdout
    
    # Verify DB add
    mock_session.add.assert_called_once()
    mock_session.commit.assert_called_once()

def test_registry_add_duplicate(mock_session, tmp_path):
    f = tmp_path / "model.pt"
    f.touch()
    
    # Mock duplicate
    existing = RegistryArtifact(name="new-model", id=99)
    mock_session.exec.return_value.first.return_value = existing
    
    result = runner.invoke(app, [
        "add", 
        "--name", "new-model", 
        "--path", str(f)
    ])
    
    assert result.exit_code == 1
    assert "already exists" in result.stdout
    
def test_registry_show(mock_session):
    art = RegistryArtifact(
        name="my-model", 
        version="v1", 
        path="/path",
        id=1,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    mock_session.exec.return_value.first.return_value = art
    
    result = runner.invoke(app, ["show", "my-model"])
    assert result.exit_code == 0
    assert "Artifact Details: my-model" in result.stdout
    assert "Path: /path" in result.stdout

def test_registry_show_not_found(mock_session):
    mock_session.exec.return_value.first.return_value = None
    result = runner.invoke(app, ["show", "missing"])
    assert result.exit_code == 1
    assert "not found" in result.stdout
