
import pytest
from typer.testing import CliRunner
from unittest.mock import MagicMock, patch
from kladml.cli.family import app
from kladml.interfaces import MetadataInterface

runner = CliRunner()

@patch("kladml.cli.family.metadata")
def test_create_family(mock_metadata):
    """Test family creation command."""
    # Setup mock
    mock_metadata.create_family.return_value = MagicMock()
    
    result = runner.invoke(app, ["create", "my-family", "--project", "my-proj", "-d", "desc"])
    
    assert result.exit_code == 0
    assert "Created family" in result.stdout
    mock_metadata.create_family.assert_called_with(name="my-family", project_name="my-proj", description="desc")

@patch("kladml.cli.family.metadata")
def test_create_family_error(mock_metadata):
    """Test error handling in creation."""
    mock_metadata.create_family.side_effect = ValueError("Project not found")
    
    result = runner.invoke(app, ["create", "my-family", "-p", "missing-proj"])
    
    assert result.exit_code == 1
    assert "Error" in result.stdout
    assert "Project not found" in result.stdout

@patch("kladml.cli.family.metadata")
def test_list_families(mock_metadata):
    """Test listing families."""
    # Mock family object
    fam = MagicMock()
    fam.name = "fam1"
    fam.project_name = "proj1"
    fam.experiment_names = ["exp1"]
    fam.description = "desc"
    
    mock_metadata.list_families.return_value = [fam]
    
    result = runner.invoke(app, ["list", "-p", "proj1"])
    
    assert result.exit_code == 0
    assert "fam1" in result.stdout
    assert "proj1" in result.stdout
    mock_metadata.list_families.assert_called_with(project_name="proj1")

@patch("kladml.cli.family.metadata")
def test_list_families_empty(mock_metadata):
    """Test listing empty results."""
    mock_metadata.list_families.return_value = []
    
    result = runner.invoke(app, ["list"])
    
    assert result.exit_code == 0
    assert "No families found" in result.stdout

@patch("kladml.cli.family.metadata")
def test_delete_family_confirm(mock_metadata):
    """Test deletion with confirmation."""
    fam = MagicMock()
    fam.experiment_names = []
    mock_metadata.get_family.return_value = fam
    
    # Input y to confirm
    result = runner.invoke(app, ["delete", "fam1", "-p", "proj1"], input="y\n")
    
    assert result.exit_code == 0
    assert "Deleted family" in result.stdout
    mock_metadata.delete_family.assert_called_with(name="fam1", project_name="proj1")

@patch("kladml.cli.family.metadata")
def test_delete_family_abort(mock_metadata):
    """Test deletion abort."""
    mock_metadata.get_family.return_value = MagicMock()
    
    # Input n to abort
    result = runner.invoke(app, ["delete", "fam1", "-p", "proj1"], input="n\n")
    
    assert result.exit_code == 0
    assert "Cancelled" in result.stdout
    mock_metadata.delete_family.assert_not_called()
