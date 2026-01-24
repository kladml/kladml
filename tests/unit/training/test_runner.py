
import pytest
from unittest.mock import MagicMock
from kladml.training.runner import ExperimentRunner
from kladml.interfaces import StorageInterface, TrackerInterface, PublisherInterface

class MockModel:
    """Mock model class."""
    def __init__(self, config=None):
        self.config = config
        
    def train(self, X_train, y_train=None, **kwargs):
        return {"loss": 0.1, "accuracy": 0.9}
        
    def evaluate(self, X_test, **kwargs):
        return {"loss": 0.15, "accuracy": 0.85}
        
    def save(self, path):
        pass

class TestExperimentRunner:
    """Tests for ExperimentRunner."""
    
    @pytest.fixture
    def mocks(self):
        storage = MagicMock(spec=StorageInterface)
        tracker = MagicMock(spec=TrackerInterface)
        publisher = MagicMock(spec=PublisherInterface)
        tracker.start_run.return_value = "run_123"
        return storage, tracker, publisher

    def test_run_success(self, mocks):
        """Test successful run execution."""
        storage, tracker, publisher = mocks
        runner = ExperimentRunner(storage=storage, tracker=tracker, publisher=publisher)
        
        result = runner.run(
            model_class=MockModel,
            train_data="dummy_train",
            val_data="dummy_val",
            experiment_name="test_exp",
            model_config={"layers": 3},
            training_config={"epochs": 5}
        )
        
        # Verify tracker calls
        tracker.start_run.assert_called_once()
        tracker.log_params.assert_called_once()
        
        # Verify params logging
        params = tracker.log_params.call_args[0][0]
        assert params["model_class"] == "MockModel"
        assert params["layers"] == 3
        assert params["train_epochs"] == 5
        
        # Verify metrics logging
        tracker.log_metrics.assert_any_call({"loss": 0.1, "accuracy": 0.9}) # Train
        tracker.log_metrics.assert_any_call({"val_loss": 0.15, "val_accuracy": 0.85}) # Eval
        
        # Verify completion
        tracker.end_run.assert_called_with("FINISHED")
        
        assert result["status"] == "COMPLETED"
        assert result["run_id"] == "run_123"

    def test_run_failure(self, mocks):
        """Test run failure handling."""
        storage, tracker, publisher = mocks
        runner = ExperimentRunner(storage=storage, tracker=tracker, publisher=publisher)
        
        class BrokenModel:
             def __init__(self, config=None): pass
             def train(self, **kwargs): raise ValueError("Kaboom")
        
        with pytest.raises(ValueError, match="Kaboom"):
            runner.run(model_class=BrokenModel, train_data="data")
            
        tracker.end_run.assert_called_with("FAILED")
        publisher.publish_status.assert_called_with("run_123", "FAILED", "Kaboom")
