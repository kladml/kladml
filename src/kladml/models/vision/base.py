from abc import abstractmethod
from typing import Any, Optional
from kladml.models.base import BaseModel
import torch.nn as nn

class ImageModel(BaseModel, nn.Module):
    """
    Abstract base class for Computer Vision models.
    Inherits from BaseModel (KladML) and nn.Module (PyTorch).
    """
    
    def __init__(self, config: Optional[dict[str, Any]] = None):
        BaseModel.__init__(self, config)
        nn.Module.__init__(self)
        
    @abstractmethod
    def forward(self, x):
        pass
