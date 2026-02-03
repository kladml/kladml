import pytest
import torch
from kladml.models.vision.classification.resnet import ResNet18

def test_resnet18_instantiation():
    config = {"num_classes": 10, "pretrained": False}
    model = ResNet18(config)
    assert model.num_classes == 10
    
def test_resnet18_forward():
    config = {"num_classes": 10, "pretrained": False}
    model = ResNet18(config)
    
    # Batch of 2 RGB images 224x224
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    
    assert y.shape == (2, 10)
    
def test_resnet18_training_step():
    config = {"num_classes": 10, "pretrained": False}
    model = ResNet18(config)
    
    x = torch.randn(2, 3, 224, 224)
    y = torch.randint(0, 10, (2,))
    batch = (x, y)
    
    loss = model.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0 # Scalar
