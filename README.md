<div align="center">

<img src="docs/assets/images/logo_full.png" alt="KladML" width="600"/>

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
| **Local-first** | ✅ Unified SQLite DB | ✅ Yes | ❌ No |
| **Learning curve** | 🟢 Minutes | 🟡 Days | 🔴 Weeks |
| **Hierarchy** | ✅ Workspace/Proj/Fam | ❌ Exp/Run | ❌ Project/task |
| **User Interface** | ✅ TUI (Terminal) | ⚠️ Web UI | ✅ Web UI |
| **Custom backends** | ✅ Easy | ⚠️ Complex | ❌ No |

---

## Installation

```bash
# Core (lightweight, no UI)
pip install kladml

# Full CLI (for terminal usage with TUI)
pip install -e ".[all]"

# For Vision support (optional)
pip install -e ".[vision]"
```

---

## Workflow

### 1. Initialize Workspace
```bash
kladml init
```
Creates the standard folder structure (`data/`, `registry/`, `projects/`).

### 2. Interactive Management (TUI)
```bash
kladml ui
```
Explore projects, runs, and datasets visually in your terminal.

### 3. Training
```bash
# Train using a config file
kladml train --config data/configs/my_config.yaml
```

---

## Supported Data Types

| Data Type | Pipeline |
|-----------|----------|
| **TABULAR** | XGBoost |
| **TIMESERIES** | Transformer/Gluformer |
| **IMAGE** | ResNet50 (Coming Soon) |

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
kladml compare --runs r1,r2   # Compare runs side-by-side

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
