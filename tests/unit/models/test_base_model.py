"""
Tests for KladML SDK base classes.
"""

import pytest
import numpy as np
from kladml.models.base import BaseModel
from kladml.data.preprocessor import BasePreprocessor


class DummyModel(BaseModel):
    """Concrete implementation for testing."""
    
    @property
    def ml_task(self):
        from kladml.tasks import MLTask
        return MLTask.CLASSIFICATION
    
    def train(self, X_train, y_train=None, X_val=None, y_val=None, **kwargs):
        self._is_trained = True
        if y_train is not None:
            self._mean = np.mean(y_train)
        else:
            self._mean = 0.0
        return {"loss": 0.1}
    
    def fit(self, X, y, **kwargs):
        """Alias for train - for backward compatibility."""
        return self.train(X, y, **kwargs)
    
    def predict(self, X, **kwargs):
        if not self.is_trained:
            raise ValueError("Model not fitted")
        return np.full(len(X), self._mean)
    
    def evaluate(self, X_test, y_test=None, **kwargs):
        return {"accuracy": 0.9}
    
    def save(self, path: str):
        pass
    
    def load(self, path: str):
        pass


class DummyPreprocessor(BasePreprocessor):
    """Concrete implementation for testing."""
    
    def fit(self, dataset):
        self._is_fitted = True
        self._mean = np.mean(dataset)
        self._std = np.std(dataset)
    
    def transform(self, dataset):
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted")
        return (dataset - self._mean) / self._std
    
    def save(self, path: str):
        pass
    
    def load(self, path: str):
        pass


class TestBaseModel:
    """Test BaseModel interface."""
    
    def test_init_with_config(self):
        """Test initialization with config."""
        config = {"hidden_size": 256}
        model = DummyModel(config=config)
        assert model.config == config
        assert model.is_trained is False
    
    def test_init_without_config(self):
        """Test initialization without config."""
        model = DummyModel()
        assert model.config == {}
    
    def test_fit_sets_fitted_flag(self):
        """Test that fit sets is_fitted."""
        model = DummyModel()
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        model.fit(X, y)
        assert model.is_trained is True
    
    def test_predict_after_fit(self):
        """Test predict returns correct shape."""
        model = DummyModel()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 2])
        model.fit(X, y)
        predictions = model.predict(X)
        assert len(predictions) == len(X)
    
    def test_get_params(self):
        """Test get_params returns config copy."""
        config = {"hidden_size": 256}
        model = DummyModel(config=config)
        params = model.get_params()
        assert params == config
        # Verify it's a copy
        params["hidden_size"] = 512
        assert model.config["hidden_size"] == 256
    
    def test_set_params(self):
        """Test set_params updates config."""
        model = DummyModel()
        result = model.set_params(hidden_size=256, dropout=0.1)
        assert model.config["hidden_size"] == 256
        assert model.config["dropout"] == 0.1
        assert result is model  # Returns self
    
    def test_api_version(self):
        """Test API version is set."""
        assert DummyModel.API_VERSION == 1


class TestBasePreprocessor:
    """Test BasePreprocessor interface."""
    
    def test_init_with_config(self):
        """Test initialization with config."""
        config = {"normalize": True}
        preprocessor = DummyPreprocessor(config=config)
        assert preprocessor.config == config
        assert preprocessor.is_fitted is False
    
    def test_fit_sets_fitted_flag(self):
        """Test that fit sets is_fitted."""
        preprocessor = DummyPreprocessor()
        data = np.array([1, 2, 3, 4, 5])
        preprocessor.fit(data)
        assert preprocessor.is_fitted is True
    
    def test_transform_after_fit(self):
        """Test transform after fit."""
        preprocessor = DummyPreprocessor()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        preprocessor.fit(data)
        transformed = preprocessor.transform(data)
        # Should be standardized (mean=0, std=1)
        assert np.abs(np.mean(transformed)) < 1e-10
        assert np.abs(np.std(transformed) - 1.0) < 1e-10
    
    def test_fit_transform(self):
        """Test fit_transform convenience method."""
        preprocessor = DummyPreprocessor()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        transformed = preprocessor.fit_transform(data)
        assert preprocessor.is_fitted is True
        assert np.abs(np.mean(transformed)) < 1e-10
    
    def test_api_version(self):
        """Test API version is set."""
        assert DummyPreprocessor.API_VERSION == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
