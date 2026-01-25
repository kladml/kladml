
import pytest
from unittest.mock import MagicMock, call, patch, ANY
from kladml.training.executor import LocalTrainingExecutor
from kladml.models.base import BaseModel

# Mock Helper
class MockModel(BaseModel):
    def train(self, X_train, X_val=None, **kwargs):
        if kwargs.get("fail", False):
            raise ValueError("Training Error")
        return {"loss": 0.1, "accuracy": 0.9}
    
    def predict(self, X): pass
    def evaluate(self, X): pass
    def save(self, path): pass
    def load(self, path): pass
    @property
    def ml_task(self): return "classification"

# Fixtures
@pytest.fixture
def mock_tracker():
    tracker = MagicMock()
    tracker.start_run.return_value = "run_123"
    return tracker

@pytest.fixture
def mock_publisher():
    return MagicMock()

@pytest.fixture
def executor(mock_tracker, mock_publisher):
    return LocalTrainingExecutor(
        model_class=MockModel,
        experiment_name="test_exp",
        config={"epochs": 10},
        tracker=mock_tracker,
        publisher=mock_publisher
    )

# Tests

def test_semantic_run_name(executor):
    """Test run name generation."""
    params = {"learning_rate": 0.01, "hidden_size": 256}
    name = executor._semantic_run_name(params, 1, 10)
    # Executor uses .0e for floats => 1e-02
    assert name == "MockModel_h256_lr1e-02_1of10"
    
    # Test abbreviations
    assert "h" in name
    assert "lr" in name

def test_generate_combinations():
    """Test grid search combination generation."""
    space = {
        "lr": [0.1, 0.01],
        "bs": [32]
    }
    combos = LocalTrainingExecutor._generate_combinations(space)
    assert len(combos) == 2
    assert {"lr": 0.1, "bs": 32} in combos
    assert {"lr": 0.01, "bs": 32} in combos

@patch("kladml.training.executor.resolve_dataset_path")
def test_execute_single_success(mock_resolve, executor, mock_tracker, mock_publisher):
    """Test successful single run execution."""
    mock_resolve.return_value = "/abs/path/data"
    
    run_id, metrics = executor.execute_single(
        data_path="data", 
        val_path="val",
        params={"lr": 0.01}
    )
    
    # Verify Path Resolution
    mock_resolve.assert_has_calls([call("data"), call("val")])
    
    # Verify Tracker
    mock_tracker.start_run.assert_called_once()
    mock_tracker.log_params.assert_called_once()
    # Logged params should include merged config
    logged_params = mock_tracker.log_params.call_args[0][0]
    assert logged_params["epochs"] == 10
    assert logged_params["lr"] == 0.01
    assert logged_params["data_path"] == "/abs/path/data"
    
    # Verify Metrics
    mock_tracker.log_metric.assert_any_call("loss", 0.1)
    
    # Verify Publisher
    mock_publisher.publish.assert_any_call("run_start", ANY)
    mock_publisher.publish.assert_any_call("run_complete", ANY)
    
    # Verify Result
    assert run_id == "run_123"
    assert metrics["loss"] == 0.1

@patch("kladml.training.executor.resolve_dataset_path")
def test_execute_single_failure(mock_resolve, executor, mock_tracker):
    """Test failed run execution."""
    mock_resolve.return_value = "/abs/path/data"
    
    # Pass 'fail' param which MockModel uses to raise error
    run_id, metrics = executor.execute_single(data_path="data", params={"fail": True})
    
    assert run_id is None
    assert metrics is None
    
    # Verify failure tracking
    mock_tracker.end_run.assert_called_with(status="FAILED")

@patch("kladml.training.executor.resolve_dataset_path")
def test_grid_search_flow(mock_resolve, executor, mock_tracker):
    """Test grid search execution logic."""
    mock_resolve.return_value = "/abs/path/data"
    
    # search space: 2 combos
    search_space = {"lr": [0.1, 0.01]}
    
    # Since execute_single is internal logic, we can mock _execute_single_run or let it run.
    # Let's let it run with MockModel.
    # We update MockModel to return different loss based on lr
    
    # But MockModel is class level. We can patch it inside test.
    
    class SmartMockModel(BaseModel):
        def train(self, X_train, X_val=None, **kwargs):
            # lower lr = lower loss
            lr = kwargs.get("lr", 0.1)
            return {"loss": lr} # 0.1 or 0.01
        def predict(self, X): pass
        def evaluate(self, X): pass
        def save(self, path): pass
        def load(self, path): pass
        @property
        def ml_task(self): return "reg"

    executor.model_class = SmartMockModel
    executor._comparison_metric = "loss"
    executor._lower_is_better = True
    
    run_ids = executor.execute_grid_search("data", search_space)
    
    assert len(run_ids) == 2
    assert executor.best_metrics["loss"] == 0.01
    
    # Verify tracker calls
    assert mock_tracker.start_run.call_count == 2
