"""
Gluformer Model Package

Native implementation of Gluformer for glucose forecasting.
"""

from kladml.models.timeseries.transformer.gluformer.model import GluformerModel
from kladml.models.timeseries.transformer.gluformer.architecture import Gluformer
from kladml.models.registry import ModelRegistry

# Register Gluformer in the model registry
ModelRegistry.register("gluformer", GluformerModel)

__all__ = [
    "GluformerModel",
    "Gluformer",
]
