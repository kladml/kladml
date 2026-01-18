<div align="center">

# KladML

**Build ML pipelines with pluggable backends. Simple. Modular. Yours.**

![PyPI - Version](https://img.shields.io/pypi/v/kladml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/kladml)
[![License](https://img.shields.io/github/license/kladml/kladml.svg)](https://github.com/kladml/kladml/blob/main/LICENSE)

`⭐ Star us on GitHub to support the project!`

</div>

---

## Why KladML?

| Feature | KladML | MLflow | ClearML |
|---------|--------|--------|---------|
| **Interface-based** | ✅ Pluggable | ❌ Hardcoded | ❌ Hardcoded |
| **Server required** | ❌ No | ⚠️ Optional | ✅ Yes |
| **Local-first** | ✅ SQLite default | ✅ Yes | ❌ No |
| **Learning curve** | 🟢 Minutes | 🟡 Days | 🔴 Weeks |
| **Hierarchy** | ✅ Project/Family/Exp | ❌ Exp/Run | ❌ Project/task |
| **Custom backends** | ✅ Easy | ⚠️ Complex | ❌ No |

---

## Installation

```bash
pip install kladml
```

## Quick Start

```bash
# Initialize workspace
kladml init

# Quick training (no database setup required)
kladml train quick \
    --config data/configs/my_config.yaml \
    --train data/datasets/train.pkl \
    --val data/datasets/val.pkl

# Evaluate a trained model
kladml eval run \
    --checkpoint path/to/best_model_jit.pt \
    --data data/datasets/test.pkl

# Resume interrupted training
kladml train quick \
    --config data/configs/my_config.yaml \
    --train data/datasets/train.pkl \
    --resume
```

### Create Your Model

```python
from kladml import TimeSeriesModel, ExperimentRunner

class MyForecaster(TimeSeriesModel):
    def train(self, X_train, y_train=None, **kwargs):
        # Your training logic
        return {"loss": 0.1}
    
    def predict(self, X, **kwargs):
        return predictions
    
    def evaluate(self, X_test, y_test=None, **kwargs):
        return {"mae": 0.5, "mse": 0.25}
    
    def save(self, path: str):
        pass
    
    def load(self, path: str):
        pass

# Run with experiment tracking
runner = ExperimentRunner()
result = runner.run(
    model_class=MyForecaster,
    train_data=train_data,
    experiment_name="my-experiment",
)
```

---

## Architecture

KladML uses **dependency injection** with abstract interfaces. Swap implementations without changing your code:

```
┌─────────────────────────────────────────────────────────────┐
│                      Your Code                              │
├─────────────────────────────────────────────────────────────┤
│                   ExperimentRunner                          │
├─────────────────────────────────────────────────────────────┤
│  StorageInterface  │  ConfigInterface  │  TrackerInterface  │
├─────────────────────────────────────────────────────────────┤
│  LocalStorage      │  YamlConfig       │  LocalTracker      │
│  S3Storage         │  EnvConfig        │  MLflowTracker     │
│  (your impl)       │  (your impl)      │  (your impl)       │
└─────────────────────────────────────────────────────────────┘
```

### Implement Custom Backends

```python
from kladml.interfaces import StorageInterface

class S3Storage(StorageInterface):
    """Custom S3 implementation."""
    
    def upload_file(self, local_path, bucket, key):
        # Your S3 logic
        ...

# Plug it in
runner = ExperimentRunner(storage=S3Storage())
```

---

## Interfaces

| Interface | Description | Default |
|-----------|-------------|---------|
| `StorageInterface` | Object storage (files, artifacts) | `LocalStorage` |
| `ConfigInterface` | Configuration management | `YamlConfig` |
| `PublisherInterface` | Real-time metric publishing | `ConsolePublisher` |
| `TrackerInterface` | Experiment tracking | `LocalTracker` (MLflow + SQLite) |

---

## Configuration

Create `kladml.yaml`:

```yaml
project:
  name: my-project
  version: 0.1.0

training:
  device: auto  # auto | cpu | cuda | mps

storage:
  artifacts_dir: ./artifacts
```

Or use environment variables:

```bash
export KLADML_TRAINING_DEVICE=cuda
export KLADML_STORAGE_ARTIFACTS_DIR=/data/artifacts
```

---

## CLI Commands

```bash
kladml --help                 # Show all commands
kladml init                   # Initialize workspace
kladml version                # Show version

# Training
kladml train quick ...        # Quick training (no DB setup)
kladml train single ...       # Full training with project/experiment

# Evaluation
kladml eval run ...           # Evaluate a model
kladml eval info              # Show available evaluators

# Data
kladml data inspect <path>    # Analyze a dataset
kladml data summary <dir>     # Summary of datasets in directory
kladml data convert ...       # Convert PKL -> HDF5

# Models
kladml models export ...      # Export to TorchScript

# Organization
kladml project list           # List all projects
kladml family list ...        # List families
kladml experiment list ...    # List experiments
```

---

## Contributing

PRs welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git clone https://github.com/kladml/kladml.git
cd kladml
pip install -e ".[dev]"
pytest
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**[Documentation](https://docs.klad.ml)** · **[PyPI](https://pypi.org/project/kladml/)** · **[GitHub](https://github.com/kladml/kladml)**

Made in 🇮🇹 by the KladML Team

</div>
