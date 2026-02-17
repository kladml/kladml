# Evaluation System

KladML provides a pluggable evaluation system for assessing model performance across different ML tasks.

---

## Overview

The evaluation system follows the same "Interface + Registry" pattern used throughout KladML:

```
BaseEvaluator (Abstract)
    ├── ClassificationEvaluator
    ├── RegressionEvaluator
    └── TimeSeriesEvaluator
         ↑
    EvaluatorRegistry (Discovery)
```

---

## Quick Start

### Using the CLI

```bash
# Evaluate a trained model
kladml eval run --run-id my_run --model best_model.pt --data test.pt --task classification

# Show available evaluators
kladml eval info
```

### Using Python API

```python
from pathlib import Path
from kladml.evaluation.registry import EvaluatorRegistry
from kladml.tasks import MLTask

# Get the appropriate evaluator for your task
evaluator_cls = EvaluatorRegistry.get_evaluator(MLTask.CLASSIFICATION)

# Instantiate and run
evaluator = evaluator_cls(
    run_dir=Path("./evaluations/run_001"),
    model_path=Path("model.pt"),
    data_path=Path("test_data.pt"),
    config={"num_classes": 3}
)

metrics = evaluator.run()
print(metrics)  # {"accuracy": 0.95, "f1": 0.92, ...}
```

---

## Available Evaluators

### ClassificationEvaluator

For classification tasks (binary and multiclass).

**Metrics:**
| Metric | Description |
|--------|-------------|
| `accuracy` | Overall classification accuracy |
| `precision` | Macro-averaged precision |
| `recall` | Macro-averaged recall |
| `f1` | Macro-averaged F1 score |
| `auroc` | Area under ROC curve (if probabilities available) |

**Plots:**
- Confusion Matrix heatmap

**Configuration:**
```python
config = {
    "num_classes": 3,        # Number of classes
    "compute_auroc": True,   # Enable AUROC computation
}
```

---

### RegressionEvaluator

For regression tasks (continuous values).

**Metrics:**
| Metric | Description |
|--------|-------------|
| `mae` | Mean Absolute Error |
| `mse` | Mean Squared Error |
| `rmse` | Root Mean Squared Error |
| `r2` | R-squared coefficient |

**Plots:**
- Prediction vs Actual scatter
- Residuals plot

---

### TimeSeriesEvaluator

For time series forecasting tasks.

**Metrics:**
| Metric | Description |
|--------|-------------|
| `mae` | Mean Absolute Error |
| `mse` | Mean Squared Error |
| `rmse` | Root Mean Squared Error |
| `mape` | Mean Absolute Percentage Error |
| `r2` | R-squared coefficient |

**Plots:**
- Prediction vs Actual (line plot over time)
- Prediction vs Actual (scatter)
- Residuals over time
- Error distribution histogram

**Configuration:**
```python
config = {
    "horizon": 12,  # Forecast horizon
}
```

---

## Evaluation Pipeline

Each evaluator follows a standard pipeline (Template Method pattern):

1. **load_model()** - Load the model from checkpoint
2. **load_data()** - Load test data
3. **inference()** - Run predictions
4. **compute_metrics()** - Calculate metrics
5. **save_plots()** - Generate visualization plots
6. **generate_report()** - Create Markdown report

---

## Output Structure

After running evaluation:

```
evaluations/run_001/
├── evaluation_report.md      # Human-readable report
├── evaluation_metrics.json   # Machine-readable metrics
└── plots/
    ├── confusion_matrix.png  # (Classification)
    ├── pred_vs_actual.png    # (Regression)
    ├── pred_vs_actual_line.png    # (TimeSeries)
    ├── pred_vs_actual_scatter.png # (TimeSeries)
    ├── residuals_over_time.png    # (TimeSeries)
    └── error_distribution.png     # (TimeSeries)
```

---

## Custom Evaluators

Create custom evaluators by subclassing `BaseEvaluator`:

```python
from kladml.evaluation.base import BaseEvaluator
from kladml.evaluation.registry import EvaluatorRegistry
from kladml.tasks import MLTask
from pathlib import Path
from typing import Any

class MyCustomEvaluator(BaseEvaluator):
    """Custom evaluator for specific domain tasks."""

    def load_model(self) -> Any:
        # Implement model loading
        pass

    def load_data(self) -> Any:
        # Implement data loading
        pass

    def inference(self, model, data) -> tuple:
        # Run inference and return (predictions, targets)
        pass

    def compute_metrics(self, predictions, targets) -> dict:
        # Return metric dictionary
        return {"custom_metric": 0.95}

    def save_plots(self, predictions, targets) -> None:
        # Generate and save plots
        pass

    def generate_report(self) -> str:
        # Return Markdown report
        return "# Custom Evaluation Report"

# Register it
EvaluatorRegistry.register(MLTask.OTHER, MyCustomEvaluator)
```

---

## Supported Tasks

| MLTask | Evaluator |
|--------|-----------|
| `CLASSIFICATION` | ClassificationEvaluator |
| `REGRESSION` | RegressionEvaluator |
| `TIMESERIES_FORECASTING` | TimeSeriesEvaluator |

Use the registry to discover available evaluators:

```python
from kladml.evaluation.registry import EvaluatorRegistry

# Get evaluator by enum or string
evaluator = EvaluatorRegistry.get_evaluator("classification")
evaluator = EvaluatorRegistry.get_evaluator(MLTask.REGRESSION)
```

---

## Data Format

Evaluators expect data in specific formats:

### Classification / Regression

```python
# Tuple of (X, y)
data = (torch.randn(100, 10), torch.randint(0, 3, (100,)))
# Or save/load as .pt file
torch.save(data, "test.pt")
```

### Time Series

```python
# Tuple of (X, y) where X has temporal dimension
data = (torch.randn(100, 60, 7), torch.randn(100, 12))
# X: (batch, seq_len, features)
# y: (batch, horizon)
```

---

## Integration with Training

After training, use the evaluator on test data:

```python
from kladml import ExperimentRunner
from kladml.evaluation import ClassificationEvaluator

# Train model
runner = ExperimentRunner()
runner.run(model_class=MyModel, train_data=train_data)

# Evaluate
evaluator = ClassificationEvaluator(
    run_dir=Path("./eval"),
    model_path=Path("./artifacts/best_model.pt"),
    data_path=Path("./test.pt"),
    config={"num_classes": 3}
)
metrics = evaluator.run()
```
