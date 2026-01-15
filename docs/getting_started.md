# Getting Started with KladML

KladML is a production-grade MLOps SDK designed to standardize how you train, track, and deploy machine learning models.

## Installation

```bash
pip install kladml
```

For full feature support (including MLflow tracking):

```bash
pip install "kladml[all]"
```

## Quick Start 🚀

### 1. Initialize a Project

Create a new project directory with the standard structure:

```bash
kladml init my-forecasting-project --template timeseries
cd my-forecasting-project
```

This creates:
- `kladml.yaml`: Configuration file.
- `train.py`: Example training script.
- `data/`, `models/`, `experiments/`: Organized folders.

### 2. Run Training Locally

You can run your script in two ways:

**A. Native (Fastest for Dev)**
Runs directly in your current Python environment (conda/venv). No Docker required.

```bash
kladml run native train.py --experiment quick-test
```

**B. Containerized (Best for Reproducibility)**
Runs inside a Docker or Podman container. Ensures your code runs in a clean, production-like environment.

```bash
# Auto-detects Docker or Podman
kladml run local train.py

# Force specific runtime
kladml run local train.py --runtime podman
```

### 3. Track Experiments

KladML automatically tracks parameters and metrics. By default, it uses a local SQLite database.

View your experiments with MLflow (if installed):

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

## Next Steps

- Learn about [Core Concepts](core_concepts.md)
- Explore the [Architecture](architecture.md)
