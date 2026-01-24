from abc import abstractmethod
from typing import Dict, Any, Optional
from kladml.models.base import BaseModel
from kladml.tasks import MLTask

class TimeSeriesModel(BaseModel):
    """
    Base class for Time Series Forecasting models.
    """
    
    @property
    def ml_task(self) -> MLTask:
        return MLTask.TIMESERIES_FORECASTING
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.window_size = self.config.get("window_size", 10)
        self.forecast_horizon = self.config.get("forecast_horizon", 1)

    @abstractmethod
    def train(self, X_train: Any, y_train: Any = None, X_val: Any = None, y_val: Any = None, **kwargs) -> Dict[str, float]:
        """
        Train the forecasting model.
        Expected input shape: (batch_size, sequence_length, features)
        """
        pass
