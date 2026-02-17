"""
Model Registry for KladML SDK.

Provides automatic architecture discovery and registration similar to
ExporterRegistry and EvaluatorRegistry patterns.

Usage:
    # Register a model with decorator
    @ModelRegistry.register("gluformer")
    class Gluformer(BaseModel):
        ...

    # Or register manually
    ModelRegistry.register("my_model", MyModelClass)

    # Discover and retrieve models
    ModelRegistry.get("gluformer")
    ModelRegistry.list()
    ModelRegistry.list_by_task(MLTask.TIMESERIES_FORECASTING)
"""

from typing import Optional
from kladml.models.base import BaseModel
from kladml.tasks import MLTask


class ModelRegistry:
    """
    Registry for model architectures.

    Provides a centralized way to discover, register, and retrieve
    model classes. Supports filtering by ML task type.

    Example:
        # List all registered models
        models = ModelRegistry.list()
        # {'gluformer': 'Gluformer - Time series forecasting transformer', ...}

        # Get a specific model class
        Gluformer = ModelRegistry.get("gluformer")

        # Filter by task
        ts_models = ModelRegistry.list_by_task(MLTask.TIMESERIES_FORECASTING)
    """
    _models: dict[str, type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str, model_cls: Optional[type[BaseModel]] = None):
        """
        Register a model class.

        Can be used as a decorator or called directly:

        @ModelRegistry.register("gluformer")
        class Gluformer(BaseModel):
            ...

        Or:

        ModelRegistry.register("gluformer", GluformerClass)

        Args:
            name: Unique identifier for the model (e.g., "gluformer")
            model_cls: Model class to register (optional if used as decorator)

        Returns:
            The model class (for decorator chaining) or None
        """
        def decorator(mcls: type[BaseModel]) -> type[BaseModel]:
            if not issubclass(mcls, BaseModel):
                raise TypeError(
                    f"Cannot register {mcls.__name__}: must be a subclass of BaseModel"
                )
            cls._models[name.lower()] = mcls
            return mcls

        if model_cls is not None:
            # Called directly: ModelRegistry.register("name", ModelClass)
            return decorator(model_cls)
        else:
            # Used as decorator: @ModelRegistry.register("name")
            return decorator

    @classmethod
    def get(cls, name: str) -> type[BaseModel]:
        """
        Get a registered model class by name.

        Args:
            name: Model identifier (case-insensitive)

        Returns:
            The model class

        Raises:
            ValueError: If model is not found
        """
        name_lower = name.lower()
        if name_lower not in cls._models:
            available = ", ".join(sorted(cls._models.keys()))
            raise ValueError(
                f"Model '{name}' not found in registry. "
                f"Available models: {available or 'none registered'}"
            )
        return cls._models[name_lower]

    @classmethod
    def list(cls) -> dict[str, str]:
        """
        List all registered models with descriptions.

        Returns:
            Dict mapping model names to their descriptions
        """
        result = {}
        for name, model_cls in cls._models.items():
            # Get description from docstring
            doc = model_cls.__doc__ or ""
            # First line of docstring as short description
            short_desc = doc.strip().split("\n")[0] if doc else model_cls.__name__

            # Include task type if available
            try:
                instance = model_cls(config={})
                task = instance.ml_task
                result[name] = f"{short_desc} [{task.value}]"
            except Exception:
                result[name] = short_desc

        return result

    @classmethod
    def list_by_task(cls, task: MLTask) -> dict[str, type[BaseModel]]:
        """
        List all models registered for a specific ML task.

        Args:
            task: The ML task to filter by

        Returns:
            Dict mapping model names to their classes
        """
        result = {}
        for name, model_cls in cls._models.items():
            try:
                instance = model_cls(config={})
                if instance.ml_task == task:
                    result[name] = model_cls
            except Exception:
                # Skip models that can't be instantiated with empty config
                pass
        return result

    @classmethod
    def contains(cls, name: str) -> bool:
        """
        Check if a model is registered.

        Args:
            name: Model identifier (case-insensitive)

        Returns:
            True if model is registered
        """
        return name.lower() in cls._models

    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered models.

        Mainly useful for testing.
        """
        cls._models.clear()

    @classmethod
    def count(cls) -> int:
        """
        Return the number of registered models.

        Returns:
            Number of registered models
        """
        return len(cls._models)
