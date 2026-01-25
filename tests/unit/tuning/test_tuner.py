
import pytest
from pydantic import ValidationError
from unittest.mock import MagicMock
import optuna

# We will import these once implemented
from kladml.tuning.tuner import KladMLTuner, TunerConfig
from kladml.tuning.search_space import SearchSpace, FloatDistribution, IntDistribution, CategoricalDistribution

def test_search_space_definition():
    """Test defining a search space using Pydantic models."""
    # Define a valid search space
    space = SearchSpace(
        parameters={
            "learning_rate": FloatDistribution(low=1e-5, high=1e-2, log=True),
            "num_layers": IntDistribution(low=1, high=4),
            "optimizer": CategoricalDistribution(choices=["adam", "sgd"])
        }
    )
    
    assert "learning_rate" in space.parameters
    assert isinstance(space.parameters["learning_rate"], FloatDistribution)
    assert space.parameters["learning_rate"].log is True

def test_tuner_optimization_simple():
    """
    Test generic optimization logic (independent of Trainer).
    We optimize a simple quadratic function: y = (x - 2)^2.
    Minimum should be at x = 2.
    """
    # 1. Define Search Space
    space = SearchSpace(
        parameters={
            "x": FloatDistribution(low=-10.0, high=10.0)
        }
    )
    
    # 2. Define Objective
    def objective_fn(params):
        x = params["x"]
        return (x - 2) ** 2
        
    # 3. Setup Tuner
    config = TunerConfig(
        study_name="test_quadratic",
        n_trials=50,
        direction="minimize",
        storage="sqlite:///:memory:" # In-memory isolation
    )
    
    tuner = KladMLTuner(config)
    
    # 4. Run Optimization
    best_params = tuner.optimize(objective_fn, space)
    
    # 5. Verify
    assert "x" in best_params
    assert abs(best_params["x"] - 2.0) < 0.1 # Should be close to 2.0
    
    # Check if study exists
    assert len(tuner.study.trials) == 50

def test_tuner_integration_mock_trainer():
    """
    Simulate a KladML training scenario.
    """
    space = SearchSpace(parameters={"lr": FloatDistribution(low=0.01, high=0.1)})
    
    # Mock trainer function
    objective_mock = MagicMock(return_value=0.5) 
    
    config = TunerConfig(study_name="test_mock", n_trials=5, storage="sqlite:///:memory:")
    tuner = KladMLTuner(config)
    
    tuner.optimize(objective_mock, space)
    
    assert objective_mock.call_count == 5
    # Check that called with dict
    args, _ = objective_mock.call_args
    assert "lr" in args[0]
