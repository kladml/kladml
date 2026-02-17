"""
Tests for the ModelRegistry.
"""

import pytest
from kladml.models.base import BaseModel
from kladml.tasks import MLTask

# Import registry - use try/except to handle cases where package not reinstalled
try:
    from kladml.models.registry import ModelRegistry
except ImportError:
    # If package not reinstalled, skip these tests
    pytest.skip("ModelRegistry not available - package needs reinstall", allow_module_level=True)


class DummyModel(BaseModel):
    """Dummy model for testing."""

    @property
    def ml_task(self) -> MLTask:
        return MLTask.TIMESERIES_FORECASTING

    def train(self, X_train, y_train=None, X_val=None, y_val=None, **kwargs):
        return {"loss": 0.1}

    def predict(self, X, **kwargs):
        return [0.5]

    def evaluate(self, X_test, y_test=None, **kwargs):
        return {"accuracy": 0.9}

    def save(self, path):
        pass

    def load(self, path):
        pass


class AnotherModel(BaseModel):
    """Another dummy model for classification."""

    @property
    def ml_task(self) -> MLTask:
        return MLTask.CLASSIFICATION

    def train(self, X_train, y_train=None, X_val=None, y_val=None, **kwargs):
        return {"loss": 0.2}

    def predict(self, X, **kwargs):
        return [1]

    def evaluate(self, X_test, y_test=None, **kwargs):
        return {"accuracy": 0.85}

    def save(self, path):
        pass

    def load(self, path):
        pass


class TestModelRegistry:
    """Tests for ModelRegistry class."""

    def setup_method(self):
        """Clear registry before each test."""
        ModelRegistry.clear()

    def test_register_with_decorator(self):
        """Test registering a model using the decorator syntax."""

        @ModelRegistry.register("dummy")
        class DecoratedModel(BaseModel):
            @property
            def ml_task(self):
                return MLTask.TIMESERIES_FORECASTING

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

        assert ModelRegistry.contains("dummy")
        assert ModelRegistry.get("dummy") == DecoratedModel

    def test_register_directly(self):
        """Test registering a model directly."""
        ModelRegistry.register("another", AnotherModel)
        assert ModelRegistry.contains("another")
        assert ModelRegistry.get("another") == AnotherModel

    def test_register_non_model_raises(self):
        """Test that registering a non-BaseModel class raises TypeError."""

        class NotAModel:
            pass

        with pytest.raises(TypeError, match="must be a subclass of BaseModel"):
            ModelRegistry.register("invalid", NotAModel)

    def test_get_nonexistent_raises(self):
        """Test that getting a non-existent model raises ValueError."""
        with pytest.raises(ValueError, match="not found in registry"):
            ModelRegistry.get("nonexistent")

    def test_get_case_insensitive(self):
        """Test that model lookup is case-insensitive."""
        ModelRegistry.register("MyModel", DummyModel)
        assert ModelRegistry.get("mymodel") == DummyModel
        assert ModelRegistry.get("MYMODEL") == DummyModel

    def test_list_empty(self):
        """Test listing when registry is empty."""
        assert ModelRegistry.list() == {}

    def test_list_with_models(self):
        """Test listing registered models."""
        ModelRegistry.register("dummy", DummyModel)
        ModelRegistry.register("another", AnotherModel)

        models = ModelRegistry.list()
        assert "dummy" in models
        assert "another" in models
        assert "timeseries_forecasting" in models["dummy"].lower()

    def test_list_by_task(self):
        """Test filtering models by ML task."""
        ModelRegistry.register("ts_model", DummyModel)
        ModelRegistry.register("clf_model", AnotherModel)

        ts_models = ModelRegistry.list_by_task(MLTask.TIMESERIES_FORECASTING)
        assert "ts_model" in ts_models
        assert "clf_model" not in ts_models

        clf_models = ModelRegistry.list_by_task(MLTask.CLASSIFICATION)
        assert "clf_model" in clf_models
        assert "ts_model" not in clf_models

    def test_contains(self):
        """Test contains check."""
        assert not ModelRegistry.contains("dummy")
        ModelRegistry.register("dummy", DummyModel)
        assert ModelRegistry.contains("dummy")
        assert ModelRegistry.contains("DUMMY")  # Case insensitive

    def test_clear(self):
        """Test clearing the registry."""
        ModelRegistry.register("dummy", DummyModel)
        assert ModelRegistry.count() == 1

        ModelRegistry.clear()
        assert ModelRegistry.count() == 0
        assert ModelRegistry.list() == {}

    def test_count(self):
        """Test counting registered models."""
        assert ModelRegistry.count() == 0

        ModelRegistry.register("dummy", DummyModel)
        assert ModelRegistry.count() == 1

        ModelRegistry.register("another", AnotherModel)
        assert ModelRegistry.count() == 2

    def test_overwrite_registration(self):
        """Test that re-registering overwrites the previous entry."""
        ModelRegistry.register("test", DummyModel)
        assert ModelRegistry.get("test") == DummyModel

        ModelRegistry.register("test", AnotherModel)
        assert ModelRegistry.get("test") == AnotherModel
