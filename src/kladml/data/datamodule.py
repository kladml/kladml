
from abc import ABC, abstractmethod
from typing import Any, Optional

class BaseDataModule(ABC):
    """
    Abstract Base Class for DataModules.
    
    A DataModule encapsulates all steps needed to process data:
    1. Prepare (download, untar) - executed on 1 GPU/Node
    2. Setup (split, transform) - executed on every GPU
    3. Loaders (train, val, test) - returned to trainer
    """
    
    def __init__(self):
        pass

    def prepare_data(self) -> None:
        """
        Use this to download and prepare the data.
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Use this to split and/or transform the data.
        """
        pass

    def train_dataloader(self) -> Any:
        """Return the training dataloader."""
        return None

    def val_dataloader(self) -> Any:
        """Return the validation dataloader."""
        return None

    def test_dataloader(self) -> Any:
        """Return the test dataloader."""
        return None
