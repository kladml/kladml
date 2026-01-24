"""
Tests for KladML Models
"""

import pytest
from abc import ABC

from kladml import (
    BaseModel,
    BasePreprocessor,
    TimeSeriesModel,
    ClassificationModel,
    MLTask,
)


class TestBaseModel:
    """Test BaseModel abstract class."""
    
    def test_is_abstract(self):
        assert issubclass(BaseModel, ABC)
    
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseModel()
    
    def test_config_initialization(self):
        # Create a concrete implementation
        class ConcreteModel(BaseModel):
            @property
            def ml_task(self):
                return MLTask.CLASSIFICATION
            
            def train(self, X_train, y_train=None, **kwargs):
                return {"loss": 0.1}
            
            def predict(self, X, **kwargs):
                return [0] * len(X)
            
            def evaluate(self, X_test, y_test=None, **kwargs):
                return {"accuracy": 0.9}
            
            def save(self, path):
                pass
            
            def load(self, path):
                pass
        
        model = ConcreteModel(config={"epochs": 10})
        assert model.config["epochs"] == 10
        assert model.is_trained == False


class TestTimeSeriesModel:
    """Test TimeSeriesModel base class."""
    
    def test_ml_task(self):
        # Create minimal implementation
        class MyForecaster(TimeSeriesModel):
            def train(self, X_train, y_train=None, **kwargs):
                return {}
            def predict(self, X, **kwargs):
                return []
            def evaluate(self, X_test, y_test=None, **kwargs):
                return {}
            def save(self, path):
                pass
            def load(self, path):
                pass
        
        model = MyForecaster()
        assert model.ml_task == MLTask.TIMESERIES_FORECASTING
    
    def test_default_config(self):
        class MyForecaster(TimeSeriesModel):
            def train(self, X_train, y_train=None, **kwargs):
                return {}
            def predict(self, X, **kwargs):
                return []
            def evaluate(self, X_test, y_test=None, **kwargs):
                return {}
            def save(self, path):
                pass
            def load(self, path):
                pass
        
        model = MyForecaster()
        assert model.window_size == 10
        assert model.forecast_horizon == 1


class TestBasePreprocessor:
    """Test BasePreprocessor abstract class."""
    
    def test_is_abstract(self):
        assert issubclass(BasePreprocessor, ABC)
    
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BasePreprocessor()
