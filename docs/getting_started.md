# Getting Started

This guide will get you up and running with KladML in under 5 minutes.

## Installation

```bash
pip install kladml
```

This installs:

- The `kladml` Python package
- MLflow for experiment tracking
- CLI commands (`kladml init`, `kladml run`, etc.)

### Verify Installation

```bash
kladml version
# KladML version 0.1.0
```

---

## Create Your First Project

```bash
kladml init my-project
cd my-project
```

This creates:

```
my-project/
├── kladml.yaml      # Project configuration
├── train.py         # Example training script
├── data/            # Dataset directory
├── models/          # Saved models
└── experiments/     # Experiment outputs
```

---

## Run Training

KladML provides a generic training command that works with any model architecture.

### 1. (Optional) Convert Data

For large datasets, convert to HDF5 for lazy loading:

```bash
kladml data convert \
    --input data/datasets/dataset.pkl \
    --output data/datasets/dataset.h5
```

### 2. Train a Model

Run a single training experiment:

```bash
kladml train single \
    --model gluformer \
    --data data/datasets/dataset.h5 \
    --project my-project \
    --experiment baseline
```

- `--model`: Name of the architecture (e.g., `gluformer`) or path to a Python file.
- `--data`: Path to your training data (PKL or HDF5).

### 3. Grid Search

Run a grid search over hyperparameters defined in a YAML config:

```bash
# config.yaml (see documentation for format)
# ... grid search params ...

kladml train grid \
    --model gluformer \
    --config config.yaml \
    --project my-project \
    --experiment hyperparam-tuning
```

---

## Create Custom Models

To add your own model architecture:

1. Create a file `my_model.py`
2. Inherit from `TimeSeriesModel` (or similar base class)
3. Implement `train`, `predict`, `evaluate`

```python
# my_model.py
from kladml import TimeSeriesModel, MLTask

class MyModel(TimeSeriesModel):
    # ... implementation ...
    pass
```

Then train it using the CLI:

```bash
kladml train single --model my_model.py --data ...
```

---

## View Experiments

KladML uses MLflow for tracking. View your experiments:

```bash
mlflow ui --backend-store-uri sqlite:///data/projects/mlflow.db
```

Open http://localhost:5000 in your browser.

---

## Next Steps

- [Core Concepts](core_concepts.md) - Understand interfaces and architecture
- [Architecture](architecture.md) - Deep dive into model contracts
- [CLI Reference](cli.md) - All available commands
