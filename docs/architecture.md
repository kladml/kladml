# Model Architecture

KladML provides standardized base classes that ensure your models are portable, reproducible, and easy to track.

---

## BaseModel

The core abstract class that all models inherit from.

```python
from kladml import BaseModel

class BaseModel(ABC):
    """Base class for all ML models."""
    
    @abstractmethod
    def train(self, X_train, y_train=None, X_val=None, y_val=None, **kwargs) -> Dict[str, Any]:
        """Train the model. Must return a metrics dictionary."""
        pass
    
    @abstractmethod
    def predict(self, X, **kwargs) -> Any:
        """Generate predictions."""
        pass
    
    @abstractmethod
    def evaluate(self, X_test, y_test=None, **kwargs) -> Dict[str, float]:
        """Evaluate the model. Must return a metrics dictionary."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model artifacts to directory."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load model artifacts from directory."""
        pass
    
    @property
    @abstractmethod
    def ml_task(self) -> MLTask:
        """Return the ML task type."""
        pass
```

---

## Specialized Model Classes

KladML provides pre-configured subclasses for common ML tasks:

### TimeSeriesModel

For forecasting and time-series analysis.

```python
from kladml import TimeSeriesModel, MLTask

class MyForecaster(TimeSeriesModel):
    
    @property
    def ml_task(self):
        return MLTask.TIMESERIES_FORECASTING
    
    def train(self, X_train, y_train=None, **kwargs):
        # Train on sequences
        return {"loss": 0.1}
    
    def predict(self, X, **kwargs):
        # Forecast future values
        return predictions
    
    # ... implement other methods
```

### ClassificationModel

For classification tasks.

```python
from kladml import ClassificationModel, MLTask

class MyClassifier(ClassificationModel):
    
    @property
    def ml_task(self):
        return MLTask.CLASSIFICATION
    
    def train(self, X_train, y_train=None, **kwargs):
        return {"accuracy": 0.95, "f1": 0.92}
    
    def predict(self, X, **kwargs):
        return class_labels
```

---

## MLTask Enum

Defines the problem type your model solves. This helps the framework visualize and evaluate results correctly.

```python
from kladml import MLTask

class MLTask(Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    TIMESERIES_FORECASTING = "timeseries_forecasting"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
```

---

## Method Contracts

### `train()`

**Input:**
- `X_train`: Training features
- `y_train`: Training labels (optional for unsupervised)
- `X_val`, `y_val`: Validation data (optional)
- `**kwargs`: Additional parameters

**Output:**
- `Dict[str, Any]`: Metrics dictionary (e.g., `{"loss": 0.1, "epochs": 10}`)

**Side effects:**
- Sets `self._is_trained = True`
- Model state is updated

---

### `predict()`

**Input:**
- `X`: Features to predict on
- `**kwargs`: Additional parameters

**Output:**
- Predictions (numpy array, list, tensor, etc.)

**Precondition:**
- Model must be trained (`self._is_trained == True`)

---

### `evaluate()`

**Input:**
- `X_test`: Test features
- `y_test`: Test labels (optional)
- `**kwargs`: Additional parameters

**Output:**
- `Dict[str, float]`: Metrics dictionary (e.g., `{"mae": 0.5, "rmse": 0.7}`)

---

### `save()` / `load()`

**Input:**
- `path: str`: Directory path for saving/loading

**Behavior:**
- Save all model artifacts (weights, config, metadata) to the directory
- Load should restore the model to trainable/predictable state

**Example:**

```python
def save(self, path: str):
    import json
    import numpy as np
    
    # Save weights
    np.save(f"{path}/weights.npy", self.weights)
    
    # Save config
    with open(f"{path}/config.json", "w") as f:
        json.dump({"hidden_size": 64}, f)

def load(self, path: str):
    import json
    import numpy as np
    
    self.weights = np.load(f"{path}/weights.npy")
    with open(f"{path}/config.json") as f:
        self.config = json.load(f)
    
    self._is_trained = True
```

---

## Design Philosophy

1. **Pure Python interfaces** - No heavy framework dependencies in the contract
2. **Separation of concerns** - Models handle math, `ExperimentRunner` handles infrastructure
3. **Portability** - Same model code runs locally, in Docker, or on Kubernetes
4. **Testability** - Easy to unit test each method independently
