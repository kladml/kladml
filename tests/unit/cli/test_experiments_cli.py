
import pytest
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
from kladml.cli.experiments import app

runner = CliRunner()

@pytest.fixture
def mock_deps():
    with patch("kladml.cli.experiments.metadata") as mock_meta, \
         patch("kladml.cli.experiments.tracker") as mock_track:
        yield mock_meta, mock_track

def test_create_experiment_success(mock_deps):
    meta, track = mock_deps
    
    # Setup mocks
    meta.get_project.return_value = MagicMock(name="proj")
    meta.get_family.return_value = MagicMock(name="fam", experiment_names=[])
    track.create_experiment.return_value = "exp_id_123"
    
    result = runner.invoke(app, [
        "create",
        "--name", "new-exp",
        "--project", "test-proj",
        "--family", "test-fam"
    ])
    
    assert result.exit_code == 0
    assert "Created/Found" in result.stdout
    track.create_experiment.assert_called_with("new-exp")
    meta.add_experiment_to_family.assert_called_with("test-fam", "test-proj", "new-exp")

def test_create_experiment_duplicate(mock_deps):
    meta, track = mock_deps
    meta.get_project.return_value = MagicMock()
    # Simulate existing experiment in family
    meta.get_family.return_value = MagicMock(experiment_names=["existing-exp"])
    
    result = runner.invoke(app, [
        "create",
        "--name", "existing-exp",
        "--project", "test-proj"
    ])
    
    assert result.exit_code == 0
    assert "already exists" in result.stdout
    track.create_experiment.assert_not_called()

def test_list_experiments(mock_deps):
    meta, track = mock_deps
    
    fam1 = MagicMock(name="fam1")
    fam1.name = "fam1"
    fam1.experiment_names = ["exp1"]
    
    meta.list_families.return_value = [fam1]
    
    # Mock tracker response for exp1
    track.get_experiment_by_name.return_value = {
        "id": "1", "name": "exp1", "lifecycle_stage": "active"
    }
    track.search_runs.return_value = []
    
    result = runner.invoke(app, ["list", "--project", "test-proj"])
    
    assert result.exit_code == 0
    assert "fam1" in result.stdout
    assert "exp1" in result.stdout

def test_delete_experiment(mock_deps):
    meta, track = mock_deps
    
    fam = MagicMock()
    fam.experiment_names = ["exp1"]
    meta.get_family.return_value = fam
    
    result = runner.invoke(app, [
        "delete", 
        "exp1",
        "--project", "test-proj",
        "--force"
    ])
    
    assert result.exit_code == 0
    assert "Removed experiment" in result.stdout
    meta.remove_experiment_from_family.assert_called_once()

def test_list_runs(mock_deps):
    meta, track = mock_deps
    track.get_experiment_by_name.return_value = {"id": "123"}
    track.search_runs.return_value = [
        {"run_id": "r1", "run_name": "runA", "status": "FINISHED", "metrics": {"loss": 0.5}}
    ]
    
    result = runner.invoke(app, ["runs", "exp1"])
    
    assert result.exit_code == 0
    assert "runA" in result.stdout
    assert "0.5000" in result.stdout

def test_compare_experiments(mock_deps):
    meta, track = mock_deps
    
    track.get_experiment_by_name.side_effect = lambda n: {"id": n, "name": n}
    
    # Mock runs for exp1
    run1 = {"run_id": "r1", "run_name": "best1", "metrics": {"loss": 0.1}}
    run2 = {"run_id": "r2", "run_name": "bad1", "metrics": {"loss": 0.9}}
    
    # Mock runs for exp2
    run3 = {"run_id": "r3", "run_name": "best2", "metrics": {"loss": 0.05}}
    
    def side_effect_search(exp_id, **kwargs):
        if exp_id == "exp1": return [run1, run2]
        if exp_id == "exp2": return [run3]
        return []
        
    track.search_runs.side_effect = side_effect_search
    
    result = runner.invoke(app, ["compare", "exp1", "exp2", "--metric", "loss"])
    
    assert result.exit_code == 0
    assert "best1" in result.stdout
    assert "best2" in result.stdout
    # 0.05 is better than 0.1
