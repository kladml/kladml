"""
KladML Models Package.

Provides model architectures for various ML tasks including:
- Time series forecasting (Gluformer, Transformers)
- Classification
- Vision models

Architecture Discovery:
    from kladml.models import ModelRegistry

    # List all available models
    models = ModelRegistry.list()

    # Get a specific model
    GluformerModel = ModelRegistry.get("gluformer")

    # Filter by task
    from kladml.tasks import MLTask
    ts_models = ModelRegistry.list_by_task(MLTask.TIMESERIES_FORECASTING)
"""

from kladml.models.base import BaseModel
from kladml.models.registry import ModelRegistry

__all__ = [
    "BaseModel",
    "ModelRegistry",
]
