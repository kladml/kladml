# Core Concepts

KladML is built around a "1 Interface, N Implementations" philosophy. This allows the same code to run on your laptop (using files) and in the cloud (using S3/MinIO and Databases) without changing a single line of logic.

## The Experiment Runner

The heart of the SDK is the `ExperimentRunner`. It orchestrates the training process:

1.  **Configures** the environment.
2.  **Initializes** your Model.
3.  **Trains** the model (passing data).
4.  **Evaluates** the model.
5.  **Saves** the artifacts.

```python
from kladml import ExperimentRunner, TimeSeriesModel

# Your model logic
class MyModel(TimeSeriesModel):
    ...

# Orchestration
runner = ExperimentRunner()
runner.run(model_class=MyModel, train_data=...)
```

## Interfaces

KladML defines 4 key interfaces. You typically don't need to touch these unless you are building custom backends.

| Interface | Purpose | Local Implementation | Enterprise Implementation |
|-----------|---------|----------------------|---------------------------|
| **StorageInterface** | Handle files & artifacts | `LocalStorage` (File System) | `S3Storage` (MinIO/AWS) |
| **ConfigInterface** | Manage settings & secrets | `YamlConfig` (kladml.yaml) | `K8sConfig` (Secrets) |
| **TrackerInterface** | Log metrics & params | `LocalTracker` (SQLite) | `MLflowServerTracker` |
| **PublisherInterface**| Real-time status updates | `ConsolePublisher` (Print) | `RedisPublisher` |

## Model Architecture

### BaseArchitecture

The standardized base class for all models. It enforces a contract:

- `train(...)`: Must return a metrics dictionary.
- `predict(...)`: Must return predictions.
- `save/load(...)`: Must handle persistence.

### MLTask

Enum defining what problem the model solves (e.g., `TIMESERIES_FORECASTING`, `CLASSIFICATION`). This helps the platform visualize results correctly.
