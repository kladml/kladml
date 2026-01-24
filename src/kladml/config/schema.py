
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal

class TrainingConfig(BaseModel):
    """
    Configuration for the UniversalTrainer.
    """
    model_config = ConfigDict(extra="forbid") # Strict config validation

    max_epochs: int = Field(default=10, ge=1, description="Maximum number of epochs to train")
    accelerator: Literal["auto", "cpu", "gpu", "cuda", "mps"] = Field(
        default="auto", 
        description="Hardware accelerator to use"
    )
    devices: str | int = Field(default="auto", description="Number of devices or 'auto'")
    
    # Optimizer params (often passed to trainer or model)
    learning_rate: float = Field(default=1e-3, gt=0, description="Learning rate")
    batch_size: int = Field(default=32, ge=1, description="Batch size for dataloaders")
    
    # Advanced
    gradient_clip_val: float | None = Field(default=None, ge=0, description="Gradient clipping value")
    accumulate_grad_batches: int = Field(default=1, ge=1, description="Gradient accumulation steps")
    
    default_root_dir: str | None = Field(default=None, description="Default root directory for logs/checkpoints")

class ModelConfig(BaseModel):
    """
    Base configuration for KladML Models.
    Allows extra fields for flexibility in specific architectures.
    """
    model_config = ConfigDict(extra="allow")
    
    name: str | None = Field(default=None, description="Name of the model instance")
    seed: int | None = Field(default=None, description="Random seed for checking reproducibility")
