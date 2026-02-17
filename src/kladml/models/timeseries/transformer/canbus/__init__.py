
from .model import CanBusModel
from .architecture import CanBusTransformer
from .dataset import CanBusDataset
from kladml.models.registry import ModelRegistry

# Register CanBus model in the model registry
ModelRegistry.register("canbus", CanBusModel)

__all__ = ["CanBusModel", "CanBusTransformer", "CanBusDataset"]
