# KladML SDK

<p align="center">
  <strong>🚀 Enterprise-grade MLOps toolkit for Python</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/kladml/"><img src="https://img.shields.io/pypi/v/kladml.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/kladml/"><img src="https://img.shields.io/pypi/pyversions/kladml.svg" alt="Python"></a>
  <a href="https://github.com/kladml/kladml/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
</p>

---

**KladML** is a lightweight, extensible SDK for building production ML pipelines. It provides:

- 🔌 **Pluggable backends** - Swap between local filesystem and cloud storage seamlessly
- 📊 **Experiment tracking** - MLflow integration out of the box
- 🎯 **Type-safe interfaces** - Abstract contracts for all core services
- 💻 **CLI included** - Initialize projects, run experiments from terminal

## Installation

```bash
pip install kladml
```

This includes MLflow for experiment tracking out of the box.

## Quick Start

### 1. Initialize a Project

```bash
kladml init my-project
cd my-project
```

### 2. Create Your Model

```python
from kladml import TimeSeriesModel, MLTask

class MyForecaster(TimeSeriesModel):
    
    def train(self, X_train, y_train=None, **kwargs):
        # Your training logic
        return {"loss": 0.1}
    
    def predict(self, X, **kwargs):
        # Your prediction logic
        return predictions
    
    def evaluate(self, X_test, y_test=None, **kwargs):
        return {"mae": 0.5, "mse": 0.25}
    
    def save(self, path: str):
        # Save model state
        pass
    
    def load(self, path: str):
        # Load model state
        pass
```

### 3. Run Training

```python
from kladml import ExperimentRunner

runner = ExperimentRunner()
result = runner.run(
    model_class=MyForecaster,
    train_data=train_data,
    experiment_name="my-experiment",
)
print(f"Run ID: {result['run_id']}")
```

Or via CLI:
```bash
kladml run native train.py --experiment my-experiment
```

## Architecture

KladML uses **dependency injection** with abstract interfaces, allowing you to swap implementations:

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Code                                │
├─────────────────────────────────────────────────────────────┤
│                  ExperimentRunner                           │
├─────────────────────────────────────────────────────────────┤
│  StorageInterface  │  ConfigInterface  │  TrackerInterface  │
├─────────────────────────────────────────────────────────────┤
│  LocalStorage      │  YamlConfig       │  LocalTracker      │
│  (filesystem)      │  (kladml.yaml)    │  (MLflow+SQLite)   │
└─────────────────────────────────────────────────────────────┘
```

### Implement Custom Backends

```python
from kladml.interfaces import StorageInterface

class S3Storage(StorageInterface):
    """Custom S3 implementation for production."""
    
    def upload_file(self, local_path, bucket, key):
        # Your S3 logic
        ...

# Use it
runner = ExperimentRunner(storage=S3Storage())
```

## Interfaces

| Interface | Description | Default Implementation |
|-----------|-------------|------------------------|
| `StorageInterface` | Object storage (files, artifacts) | `LocalStorage` (filesystem) |
| `ConfigInterface` | Configuration management | `YamlConfig` (kladml.yaml + env) |
| `PublisherInterface` | Real-time metric publishing | `ConsolePublisher` (stdout) |
| `TrackerInterface` | Experiment tracking | `LocalTracker` (MLflow + SQLite) |

## Configuration

Create `kladml.yaml` in your project root:

```yaml
project:
  name: my-project
  version: 0.1.0

training:
  device: auto  # auto | cpu | cuda | mps

storage:
  artifacts_dir: ./artifacts

mlflow:
  tracking_uri: sqlite:///mlruns.db
```

Or use environment variables:

```bash
export KLADML_TRAINING_DEVICE=cuda
export KLADML_STORAGE_ARTIFACTS_DIR=/data/artifacts
```

## CLI Commands

```bash
kladml --help                    # Show all commands
kladml init <name>               # Initialize new project
kladml run native <script>       # Run training locally
kladml run local <script>        # Run in Docker container
```

## Development

```bash
# Clone and install in development mode
git clone https://github.com/kladml/kladml.git
cd kladml
pip install -e ".[dev,tracking]"

# Run tests
pytest

# Build package
python -m build
```

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  Made with ❤️ by the KladML Team
</p>
