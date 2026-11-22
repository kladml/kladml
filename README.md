# KladML

Training and experiment tracking for small teams: config-as-code, PyTorch through Hugging Face
Accelerate, MLflow for artefacts, one local SQLite database for runs, and a terminal UI to browse
them.

> **Status: experimental, unmaintained since February 2026.**
> KladML was written in early 2026 as the training harness behind an edge-AI project. That project
> shipped; this repository stopped there. It stays public as a record of the interfaces it settled
> on — storage, config, tracking and publishing, each behind an abstract class you can swap.

---

## What it does

- **Training** — one or many GPUs through Accelerate, with mixed precision (FP16/BF16) and gradient
  accumulation, without changing your model code.
- **Configuration** — hierarchical YAML validated with Pydantic, overridable from environment
  variables.
- **Tracking** — runs, metrics and artefacts in a local SQLite database; MLflow available as an
  optional tracker.
- **Data** — dataset inspection and conversion between PKL, Parquet and HDF5, with Polars
  underneath.
- **Terminal UI** — projects, runs and datasets without leaving the shell.

---

## Requirements

- **Python**: 3.10, 3.11, 3.12 (native support for modern type hints)
- **OS**: Linux, macOS, Windows

## Installation

```bash
# Core (lightweight, minimal dependencies)
pip install kladml

# Training + data engine (includes Polars, Torch)
pip install "kladml[train]"

# Full suite (tracking + TUI + dev)
pip install "kladml[all]"

# For developers: reproducible GPU setup
mamba env create -f environment.yml
mamba activate kladml
pip install -e .
```

---

## Workflow

### 1. Initialize a workspace

```bash
kladml init
```

Creates the standard structure (`data/configs/`, `data/projects/`, `data/datasets/`).

### 2. Browse it in the terminal

```bash
kladml ui
```

### 3. Train

```bash
# Single process, device detected (GPU/MPS/CPU)
kladml train --config data/configs/my_config.yaml

# Multi-GPU
kladml train --config ... --distributed --num-processes 2
```

KladML works with any PyTorch or scikit-learn model. Reference configurations live in `examples/`.

---

## Architecture

Dependency injection over abstract interfaces: swap an implementation without touching the runner.

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

### A custom backend

```python
from kladml.interfaces import StorageInterface

class S3Storage(StorageInterface):
    """Custom S3 implementation."""

    def upload_file(self, local_path, bucket, key):
        # Your S3 logic
        ...

runner = ExperimentRunner(storage=S3Storage())
```

---

## Interfaces

| Interface | Description | Default |
|-----------|-------------|---------|
| `StorageInterface` | Object storage (files, artefacts) | `LocalStorage` |
| `ConfigInterface` | Configuration management | `YamlConfig` |
| `PublisherInterface` | Real-time metric publishing | `ConsolePublisher` |
| `TrackerInterface` | Experiment tracking | `LocalTracker` (MLflow + SQLite) |

---

## Configuration

`kladml.yaml`:

```yaml
project:
  name: my-project
  version: 0.1.0

training:
  device: auto  # auto | cpu | cuda | mps

storage:
  artifacts_dir: ./data
```

Or environment variables:

```bash
export KLADML_TRAINING_DEVICE=cuda
export KLADML_STORAGE_ARTIFACTS_DIR=/data/artifacts
```

---

## CLI

```bash
kladml --help                 # all commands
kladml init                   # initialize workspace
kladml version                # version

# Training
kladml train quick ...        # quick run, no database setup
kladml train single ...       # full run with project/experiment

# Evaluation
kladml eval run ...           # evaluate a model
kladml eval info              # available evaluators
kladml compare --runs r1,r2   # compare runs side by side

# Data
kladml data inspect <path>    # analyse a dataset (Parquet/PKL)
kladml data summary <dir>     # summary of datasets
kladml data convert ...       # PKL -> Parquet/HDF5

# Models
kladml export ...             # export to ONNX

# Organisation
kladml project list
kladml family list ...
kladml experiment list ...
```

---

## Contributing

The project is not actively maintained, so issues and pull requests may go unanswered. If you want
to take it further, fork it: the interfaces are the part worth keeping, and they are small.

```bash
git clone https://github.com/kladml/kladml.git
cd kladml
pip install -e ".[dev]"
pytest
```

---

## License

MIT — see [LICENSE](LICENSE).

**[Documentation](https://docs.klad.ml)** · **[PyPI](https://pypi.org/project/kladml/)**
