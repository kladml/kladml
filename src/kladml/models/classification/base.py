from abc import abstractmethod
from typing import Dict, Any, Optional
from kladml.models.base import BaseModel
from kladml.tasks import MLTask

class ClassificationModel(BaseModel):
    """
    Base class for Classification models.
    Supports Image, Text, and Tabular classification.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.num_classes = self.config.get("num_classes", 2)

    @property
    def ml_task(self) -> MLTask:
        # Default to generic Image Classification if not specified in config?
        # Ideally user should override this if they are doing Text Classification
        # Or we can make this abstract if ClassificationModel is too generic.
        # Let's default to IMAGE_CLASSIFICATION for now as it's common,
        # or require subclass to implement it if we want to be strict.
        # But this is a concrete base for users.
        return MLTask.IMAGE_CLASSIFICATION

    @abstractmethod
    def train(self, X_train: Any, y_train: Any = None, X_val: Any = None, y_val: Any = None, **kwargs) -> Dict[str, float]:
        pass
