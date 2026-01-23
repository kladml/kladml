# KladML Architecture

This document describes the architectural decisions and conventions for KladML.

For the implementation plan, see [ROADMAP.md](./ROADMAP.md).

---

## Core Principles

1. **Universal MLOps**: Preprocessors, Architectures, Evaluators are reusable across datasets
2. **Decoupled Entities**: Components are independent and composable
3. **Pydantic Schemas**: All configs validated with Pydantic
4. **Registry Pattern**: Built-in + user-defined components, dynamically loaded
5. **Convention over Configuration**: Minimal config, maximum automation

---

## Directory Structure

```
data/
├── kladml.sqlite                         # Single local database

├── datasets/                             # GLOBAL - reusable
│   └── {dataset_name}/
│       ├── raw/
│       ├── processed/
│       │   ├── train.parquet
│       │   ├── val.parquet
│       │   ├── test.parquet
│       │   └── preprocessing_config.yaml
│       └── metadata.yaml

├── configs/                              # GLOBAL - experiment configs
│   └── {experiment_name}.yaml

├── projects/                             # Results per project
│   └── {project}/
│       └── {family}/
│           └── {experiment}/
│               └── {run_id}/
│                   ├── config.yaml       # Expanded config snapshot
│                   ├── training.jsonl
│                   ├── checkpoints/
│                   │   ├── best_model.pt
│                   │   └── last_model.pt
│                   ├── exports/
│                   │   └── model.onnx
│                   └── evaluations/
│                       └── eval_001/
│                           ├── metrics.json
│                           └── plots/

└── exports/                              # PRODUCTION models (promoted)
    └── {project}/
        └── {model_name}_v{version}.onnx
```

---

## Config System

### User Config (what you write)

```yaml
# data/configs/canbus_anomaly_v2.yaml

project: sentinella
family: canbus_anomaly
experiment: foundation_v2_nano
version: "2.0.0"

dataset: canbus/processed

architecture: TransformerAutoencoder
params:
  d_model: 64
  n_heads: 4

training:
  epochs: 15
  batch_size: 128

export:
  auto: true
  format: onnx

evaluation:
  auto: true
  evaluator: AnomalyEvaluator
  plots: [cdf, loglog]
```

### Expanded Config (auto-saved in run folder)

The system saves a fully resolved config with:
- All defaults filled in
- Absolute paths
- Timestamps
- SDK version

---

## Python API

### Recommended: Hybrid Approach

```python
from kladml import Experiment

# Load config, optionally override
exp = Experiment.from_config("data/configs/canbus_v2.yaml")
exp.config.training.epochs = 20  # Override if needed

# Run training + auto-export + auto-eval
run = exp.run()

# Or one-liner
run = Experiment.from_config("config.yaml").run()
```

### Explicit API (advanced)

```python
from kladml import Project, Dataset, Architecture

project = Project("sentinella")
dataset = Dataset.load("canbus/processed")
arch = Architecture.from_registry("TransformerAutoencoder", d_model=64)

run = project.train(architecture=arch, dataset=dataset, epochs=15)
run.evaluate(evaluator="AnomalyEvaluator")
run.export(format="onnx")
```

---

## Database Schema

### Core Tables

| Table | Purpose |
|-------|---------|
| `dataset` | Registered datasets (name, data_type, path) |
| `architecture` | Registered architectures (name, module_path, param_schema) |
| `preprocessor` | Registered preprocessors (name, input_types, output_schema) |
| `evaluator` | Registered evaluators (name, task_types, plot_types) |
| `project` | Projects container |
| `run` | Training runs (metrics, artifacts_path) |
| `evaluation` | Evaluation instances (run_id, evaluator_id, metrics) |

### Compatibility Matrix

- `Preprocessor.input_types` ∩ `Dataset.data_type`
- `Preprocessor.output_schema` == `Architecture.input_schema`
- `Architecture.model_type` ∈ `Evaluator.task_types`

---

## Registry System

Components are registered in DB and loaded dynamically:

```python
PREPROCESSOR_REGISTRY = {
    "TimeSeriesInterpolator": "kladml.data.transforms.TimeSeriesInterpolator",
    "StandardScaler": "kladml.data.transforms.StandardScaler",
}

ARCHITECTURE_REGISTRY = {
    "TransformerAutoencoder": "kladml.architectures.transformer.TransformerAutoencoder",
}

EVALUATOR_REGISTRY = {
    "AnomalyEvaluator": "kladml.evaluation.anomaly.AnomalyEvaluator",
}
```

### CLI Registration

```bash
kladml register preprocessor --name MyTransform --module my_pkg.Transform
kladml list preprocessors
```

---

## Naming Conventions

| Entity | Format | Example |
|--------|--------|---------|
| Run ID | `run_{NNN}_{YYYYMMDD}_{HHMM}` | `run_002_20260121_1345` |
| Evaluation | `eval_{NNN}` | `eval_001` |
| Export (dev) | `model.onnx` | N/A |
| Export (prod) | `{name}_v{version}.onnx` | `canbus_anomaly_v2.onnx` |

---

## Platform-Ready Design

KladML Core is designed to be importable as a library by both CLI and Platform.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     KladML Platform                          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  FastAPI + WebSocket                                     ││
│  │  - Auth, Multi-tenancy, Job Queue                        ││
│  │  - Real-time progress via WebSocket                      ││
│  └─────────────────────────────────────────────────────────┘│
│                           ↓ imports                          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  KladML Core                                             ││
│  │  - TrainingManager, Evaluator, QuickStart                ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **No global state**: All operations receive explicit context
2. **Dependency injection**: Components receive dependencies via constructor
3. **Return values over side effects**: Functions return Pydantic models, don't print
4. **Async-ready**: Core ops are async or easily wrappable
5. **Event-driven**: Emit events for progress/state changes

---

## Event System (Callbacks)

For real-time progress reporting (CLI rich output, Platform WebSocket):

```python
from abc import ABC, abstractmethod
from typing import Protocol

class TrainingEvent:
    """Base event emitted during training."""
    pass

class EpochCompleted(TrainingEvent):
    epoch: int
    total_epochs: int
    train_loss: float
    val_loss: float | None

class TrainingProgress(TrainingEvent):
    current: int
    total: int
    message: str

class TrainingCompleted(TrainingEvent):
    run_id: str
    final_metrics: dict

class TrainingFailed(TrainingEvent):
    error: str
    traceback: str

class EventHandler(Protocol):
    def on_event(self, event: TrainingEvent) -> None: ...

# CLI implementation
class RichProgressHandler(EventHandler):
    def on_event(self, event: TrainingEvent):
        if isinstance(event, TrainingProgress):
            self.progress_bar.update(event.current)

# Platform implementation
class WebSocketHandler(EventHandler):
    def on_event(self, event: TrainingEvent):
        await self.ws.send_json(event.model_dump())
```

### Usage in TrainingManager

```python
class TrainingManager:
    def __init__(self, event_handler: EventHandler | None = None):
        self.event_handler = event_handler or NoOpHandler()
    
    async def train(self, config: TrainConfig) -> TrainResult:
        for epoch in range(config.epochs):
            # ... training logic ...
            self.event_handler.on_event(EpochCompleted(
                epoch=epoch,
                total_epochs=config.epochs,
                train_loss=loss
            ))
```

---

## Cancellation Support

For stopping long-running operations gracefully:

```python
import asyncio

class CancellationToken:
    def __init__(self):
        self._cancelled = False
    
    def cancel(self):
        self._cancelled = True
    
    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

# Usage
async def train(config, cancel_token: CancellationToken | None = None):
    for epoch in range(epochs):
        if cancel_token and cancel_token.is_cancelled:
            # Save checkpoint before exiting
            save_checkpoint(model, f"cancelled_epoch_{epoch}.pt")
            raise TrainingCancelled(f"Cancelled at epoch {epoch}")
        # ... training logic ...
```

---

## Artifact Storage Abstraction

Artifacts (models, plots, configs) use an abstract storage backend:

```python
from abc import ABC, abstractmethod
from pathlib import Path

class ArtifactStore(ABC):
    @abstractmethod
    def save(self, key: str, data: bytes) -> str:
        """Save artifact, return URI."""
        pass
    
    @abstractmethod
    def load(self, key: str) -> bytes:
        """Load artifact by key."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        pass

# Local filesystem (CLI, development)
class FileSystemStore(ArtifactStore):
    def __init__(self, base_path: Path):
        self.base_path = base_path
    
    def save(self, key: str, data: bytes) -> str:
        path = self.base_path / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return f"file://{path.absolute()}"

# Cloudflare R2 (Platform)
class R2Store(ArtifactStore):
    def __init__(self, bucket: str, endpoint: str, credentials: R2Credentials):
        self.client = boto3.client('s3', endpoint_url=endpoint, ...)
    
    def save(self, key: str, data: bytes) -> str:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data)
        return f"r2://{self.bucket}/{key}"
```

### Configuration

```yaml
# CLI/Local
storage:
  backend: filesystem
  base_path: ./data

# Platform
storage:
  backend: r2
  bucket: kladml-artifacts
  endpoint: https://xxx.r2.cloudflarestorage.com
```

---

## Data Types

KladML supports multiple data types with automatic detection:

| Data Type | File Formats | Auto-Detection |
|-----------|--------------|----------------|
| `TABULAR` | CSV, Parquet | Numeric columns, no datetime index |
| `TIMESERIES` | CSV, Parquet | Has datetime column/index |
| `IMAGE` | JPG, PNG, folder | Image extensions or folder structure |
| `TEXT` | CSV, JSON, TXT | Text columns detected |

```python
from enum import Enum

class DataType(str, Enum):
    TABULAR = "tabular"
    TIMESERIES = "timeseries"
    IMAGE = "image"
    TEXT = "text"

def detect_data_type(path: Path) -> DataType:
    """Auto-detect data type from file/folder."""
    if path.is_dir():
        if any(p.suffix in ['.jpg', '.png'] for p in path.iterdir()):
            return DataType.IMAGE
    
    if path.suffix in ['.csv', '.parquet']:
        df = load_sample(path, n=100)
        if has_datetime_index(df):
            return DataType.TIMESERIES
        if has_text_columns(df):
            return DataType.TEXT
        return DataType.TABULAR
    
    raise ValueError(f"Cannot detect data type for {path}")
```

---

## Quickstart Pipeline Matrix

Pre-configured pipelines for common use cases:

| Data Type | Task | Preprocessors | Architecture | Evaluator |
|-----------|------|---------------|--------------|-----------|
| TABULAR | Classification | AutoCategorical, StandardScaler | XGBoostClassifier | ClassificationEvaluator |
| TABULAR | Regression | StandardScaler | XGBoostRegressor | RegressionEvaluator |
| TABULAR | Clustering | StandardScaler | KMeans | ClusteringEvaluator |
| TIMESERIES | Anomaly | Interpolate, Scale, Window | TransformerAutoencoder | AnomalyEvaluator |
| TIMESERIES | Forecast | Interpolate, Scale, Window | Gluformer | TimeSeriesEvaluator |
| IMAGE | Classification | Resize, Normalize, Augment | ResNet50Transfer | ClassificationEvaluator |
| TEXT | Classification | Tokenize, Pad | BERTClassifier | ClassificationEvaluator |

### Quickstart Flow

```bash
kladml quickstart --data my_data.csv

# Output:
📊 Analyzing data...
   Data type: TABULAR (5 columns, 1000 rows)
   
? What task do you want to perform?
  > Classification (detected 'label' column)
    Regression
    Clustering

? Which column is the target?
  > label

🔧 Selected pipeline:
   - Preprocessor: AutoCategorical + StandardScaler
   - Architecture: XGBoostClassifier
   - Evaluator: ClassificationEvaluator

🚀 Training...
✅ Complete! Results saved to data/projects/quickstart/run_001/
```

---

## Structured Logging

All logs include contextual information for debugging:

```python
import structlog

logger = structlog.get_logger()

# Context binding
log = logger.bind(
    user_id="user_123",      # Platform only
    run_id="run_001",
    project="sentinella"
)

log.info("Training started", epochs=15, batch_size=128)
# Output (JSON):
# {"event": "Training started", "epochs": 15, "batch_size": 128, 
#  "run_id": "run_001", "project": "sentinella", "timestamp": "..."}
```

### Log Levels

| Level | Usage |
|-------|-------|
| `DEBUG` | Internal details (tensor shapes, gradients) |
| `INFO` | User-relevant progress (epoch completed, export done) |
| `WARNING` | Non-fatal issues (missing optional config) |
| `ERROR` | Recoverable errors (checkpoint load failed, using default) |
| `CRITICAL` | Fatal errors (training cannot continue) |

---

## Hyperparameter Tuning (Optuna)

```python
from kladml.tuning import OptunaStudy, SearchSpace

class TransformerSearchSpace(SearchSpace):
    d_model: tuple[int, int] = (64, 256)
    n_heads: tuple[int, int] = (2, 8)
    n_layers: tuple[int, int] = (2, 6)
    learning_rate: tuple[float, float] = (1e-5, 1e-2)

study = OptunaStudy(
    config_path="config.yaml",
    search_space=TransformerSearchSpace(),
    n_trials=50,
    pruner="hyperband"
)

best_config = study.optimize()
best_config.save("best_config.yaml")
```

### CLI

```bash
kladml tune --config config.yaml --n-trials 50 --timeout 3600
```
