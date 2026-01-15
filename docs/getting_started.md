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

### Option 1: Native (Development)

Run directly with your local Python environment:

```bash
kladml run native train.py
```

!!! tip "Best for development"
    Native mode is fastest for iterating on your code. No Docker required.

### Option 2: Containerized (Reproducibility)

Run inside a Docker/Podman container:

```bash
kladml run local train.py
```

!!! note "GPU Support"
    For CUDA, use `--device cuda`. The container automatically uses the GPU image.
    ```bash
    kladml run local train.py --device cuda
    ```

---

## Create a Model

Here's a minimal example:

```python
from kladml import TimeSeriesModel

class MyForecaster(TimeSeriesModel):
    
    def train(self, X_train, y_train=None, **kwargs):
        """Train the model. Return metrics dict."""
        # Your training logic here
        self.weights = ...
        return {"loss": 0.1, "epochs": 10}
    
    def predict(self, X, **kwargs):
        """Generate predictions."""
        return self.weights @ X
    
    def evaluate(self, X_test, y_test=None, **kwargs):
        """Evaluate on test data. Return metrics dict."""
        predictions = self.predict(X_test)
        return {"mae": abs(predictions - y_test).mean()}
    
    def save(self, path: str):
        """Save model to directory."""
        import json
        with open(f"{path}/weights.json", "w") as f:
            json.dump(self.weights.tolist(), f)
    
    def load(self, path: str):
        """Load model from directory."""
        import json
        with open(f"{path}/weights.json") as f:
            self.weights = json.load(f)
```

---

## Use the ExperimentRunner

The `ExperimentRunner` handles:

- Creating MLflow runs
- Logging parameters and metrics
- Saving artifacts
- Managing the training lifecycle

```python
from kladml import ExperimentRunner

runner = ExperimentRunner()

result = runner.run(
    model_class=MyForecaster,
    train_data=(X_train, y_train),
    test_data=(X_test, y_test),
    experiment_name="my-experiment",
    params={"learning_rate": 0.01}
)

print(f"Run ID: {result['run_id']}")
print(f"Metrics: {result['metrics']}")
```

---

## View Experiments

KladML uses MLflow for tracking. View your experiments:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Open http://localhost:5000 in your browser.

---

## Next Steps

- [Core Concepts](core_concepts.md) - Understand interfaces and architecture
- [Architecture](architecture.md) - Deep dive into model contracts
- [CLI Reference](cli.md) - All available commands
