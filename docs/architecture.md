# Model Architecture

KladML uses a standardized architecture to ensure models are portable, reproducible, and easy to track.

## BaseArchitecture

The core of the SDK is the `BaseArchitecture` abstract base class. It defines the contract that all models must follow.

### Key Methods

1.  **`train(...)`**:
    *   **Input**: `X_train`, `y_train`, `X_val`, `y_val`.
    *   **Output**: A dictionary of metrics (e.g., `{"loss": 0.1, "accuracy": 0.95}`).
    *   **Side Effects**: Sets `self._is_trained = True`.

2.  **`predict(...)`**:
    *   **Input**: `X` (features).
    *   **Output**: Predictions (numpy array, list, or tensor).

3.  **`evaluate(...)`**:
    *   **Input**: `X_test`, `y_test`.
    *   **Output**: Metrics dictionary.

4.  **`save(path)` / `load(path)`**:
    *   Handle persistence of model artifacts (weights, config) to a local directory.

### Why this design?

*   **Pure Python**: No heavy dependencies in the interface.
*   **Separation of Concerns**: The model focuses on *math*. The `ExperimentRunner` focuses on *infrastructure* (tracking, logging, storage).
*   **Portability**: A model written with `BaseArchitecture` can run on your laptop, in a Docker container, or on a Kubernetes cluster without code changes.

## MLTask

The `ml_task` property defines the problem type:

```python
@property
def ml_task(self):
    return MLTask.TIMESERIES_FORECASTING
```

Supported tasks:
*   `REGRESSION`
*   `CLASSIFICATION`
*   `TIMESERIES_FORECASTING`
*   `CLUSTERING`
*   `ANOMALY_DETECTION`

