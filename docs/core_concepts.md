# Core Concepts

KladML is built on a "1 Interface, N Implementations" philosophy. This allows you to run the same code locally (with files and SQLite) and in production (with S3, databases, etc.) without changing your logic.

---

## Architecture Overview

KladML organizes your work in a structured hierarchy, starting from a **Workspace**:

```
Workspace
├── Projects (Logic)
│   └── Family
│       └── Experiment
│           └── Run
├── Datasets (Data versioning)
└── Configs (Configuration files)
```

- **Workspace**: Root of your local environment (`.kladml/` + `data/`).
- **Project**: High-level container (e.g., "CustomerChurn").
- **Family**: Grouping of related experiments (e.g., "DeepLearning").
- **Dataset**: First-class entity, synced to DB.
- **Run**: Individual training execution.

---

## Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Your Code                              │
│         (Model implementation, Training script)             │
├─────────────────────────────────────────────────────────────┤
│                   ExperimentRunner                          │
│         (Orchestrates training, tracking, storage)          │
├─────────────────────────────────────────────────────────────┤
│  StorageInterface  │  ConfigInterface  │  TrackerInterface  │
│                    │                   │                    │
│  (Abstraction)     │  (Abstraction)    │  (Abstraction)     │
├─────────────────────────────────────────────────────────────┤
│  LocalStorage      │  YamlConfig       │  LocalTracker      │
│  S3Storage         │  EnvConfig        │  MLflowTracker     │
│  (Implementation)  │  (Implementation) │  (Implementation)  │
└─────────────────────────────────────────────────────────────┘
```

---

## Interfaces

KladML defines 4 core interfaces. Each interface defines **what** a backend must do, not **how**.

### StorageInterface

Handles files and artifacts (models, data, outputs).

```python
from kladml.interfaces import StorageInterface

class StorageInterface(ABC):
    @abstractmethod
    def upload_file(self, local_path: str, bucket: str, key: str) -> str: ...
    
    @abstractmethod
    def download_file(self, bucket: str, key: str, local_path: str) -> str: ...
    
    @abstractmethod
    def list_files(self, bucket: str, prefix: str = "") -> List[str]: ...
```

**Default:** `LocalStorage` (filesystem)

---

### ConfigInterface

Manages configuration and settings.

```python
from kladml.interfaces import ConfigInterface

class ConfigInterface(ABC):
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any: ...
    
    @abstractmethod
    def set(self, key: str, value: Any) -> None: ...
```

**Default:** `YamlConfig` (reads from `kladml.yaml` + environment variables)

---

### TrackerInterface

Logs experiments, parameters, and metrics.

```python
from kladml.interfaces import TrackerInterface

class TrackerInterface(ABC):
    @abstractmethod
    def start_run(self, experiment_name: str, run_name: str = None) -> str: ...
    
    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None: ...
    
    @abstractmethod
    def log_metric(self, key: str, value: float, step: int = None) -> None: ...
    
    @abstractmethod
    def end_run(self, status: str = "FINISHED") -> None: ...
```

**Default:** `LocalTracker` (MLflow + SQLite)

---

### PublisherInterface

Publishes real-time updates during training.

```python
from kladml.interfaces import PublisherInterface

class PublisherInterface(ABC):
    @abstractmethod
    def publish(self, channel: str, message: Dict[str, Any]) -> None: ...
```

**Default:** `ConsolePublisher` (prints to stdout)

---

## The ExperimentRunner

The `ExperimentRunner` is the central orchestrator. It:

1. **Loads configuration** from `kladml.yaml`
2. **Starts a tracking run** (MLflow)
3. **Instantiates your model**
4. **Calls `train()`** and logs metrics
5. **Calls `evaluate()`** and logs results
6. **Saves artifacts** (model weights, config)

```python
from kladml import ExperimentRunner

runner = ExperimentRunner(
    storage=LocalStorage(),      # Optional: custom storage
    config=YamlConfig(),         # Optional: custom config
    tracker=LocalTracker(),      # Optional: custom tracker
)

result = runner.run(
    model_class=MyModel,
    train_data=(X_train, y_train),
    experiment_name="my-experiment",
)
```

---

## Implementing Custom Backends

Want to use S3 instead of local files? Implement the interface:

```python
from kladml.interfaces import StorageInterface
import boto3

class S3Storage(StorageInterface):
    def __init__(self, bucket_name: str):
        self.s3 = boto3.client('s3')
        self.bucket = bucket_name
    
    def upload_file(self, local_path: str, bucket: str, key: str) -> str:
        self.s3.upload_file(local_path, bucket, key)
        return f"s3://{bucket}/{key}"
    
    def download_file(self, bucket: str, key: str, local_path: str) -> str:
        self.s3.download_file(bucket, key, local_path)
        return local_path
    
    def list_files(self, bucket: str, prefix: str = "") -> List[str]:
        response = self.s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        return [obj['Key'] for obj in response.get('Contents', [])]

# Use it
runner = ExperimentRunner(storage=S3Storage("my-bucket"))
```

---

---

---

---

## Advanced Usage

### Lightweight Installation
If you are running KladML in a lightweight environment (e.g. CI/CD, Scripts, Notebooks) where you don't need the TUI:

- **Core (`pip install kladml`)**: Installs only essential ML logic and models.
- **Full CLI (`pip install "kladml[cli]"`)**: Includes the TUI and rich terminal formatting.

### Modular Design
KladML uses a modular interface-based design. This ensures your training code is decoupled from the underlying storage or tracking implementation, keeping your codebase clean and testable.

---

## Next Steps

- [Architecture](architecture.md) - Model contracts and design patterns
- [CLI Reference](cli.md) - Command-line interface
- [Roadmap](roadmap.md) - Planned features

---

## Training & Callbacks

Starting from v0.3.0, KladML standardizes the training lifecycle for all models.

### Standard Features
Every model inheriting from `BaseModel` automatically gets:

1.  **Structured Logging**: Training logs are saved to `data/projects/<project>/<experiment>/<run_id>/training.jsonl`.
2.  **Automatic Checkpointing**: 
    - `best_model.pth`: PyTorch state dict (weights + optimizer) for resuming training.
    - `best_model_jit.pt`: TorchScript artifact optimized for deployment.
3.  **Metrics Tracking**: Loss and validation metrics are tracked per epoch.

### Configurable Early Stopping
Early stopping is built-in but fully configurable via your `config.yaml` or run configuration:

```yaml
early_stopping:
  enabled: true       # Set to false to disable
  patience: 10        # Epochs to wait for improvement
  min_delta: 0.001    # Minimum change to count as improvement
```

If not specified, defaults are: enabled=True, patience=5.
