from typing import Any, Optional
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from kladml.models.vision.classification.base import ImageClassifier

class ResNet18(ImageClassifier):
    """
    ResNet-18 implementation for KladML.
    Wraps torchvision.models.resnet18.
    """
    
    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__(config)
        
        pretrained = self.config.get("pretrained", True)
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        
        self.backbone = resnet18(weights=weights)
        
        # Replace the last fc layer to match num_classes
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, self.num_classes)
        
    def forward(self, x):
        return self.backbone(x)
