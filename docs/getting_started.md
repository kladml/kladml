# Getting Started

This guide will get you up and running with KladML in under 5 minutes.

## Installation

```bash
# Core library (lightweight, no UI)
pip install kladml

# Full CLI with Terminal UI
pip install "kladml[cli]"
```

### Verify Installation

```bash
kladml version
# KladML version X.X.X
```

---

## Option 1: Universal Quickstart (Recommended)

**Zero to Training in 60 Seconds** - Just point KladML at your data:

```bash
kladml quickstart --data my_data.csv
```

KladML will:

1. 📊 **Analyze** your data (detect type: tabular, timeseries, image, text)
2. 🎯 **Suggest** a task (classification, regression, anomaly, etc.)
3. 🔧 **Select** the best pipeline (preprocessors + architecture + evaluator)
4. 🚀 **Train** the model
5. 📈 **Evaluate** and generate a report

### Supported Data Types

| Data Type | Auto-Detection | Example Pipeline |
|-----------|----------------|------------------|
| TABULAR | Numeric CSV/Parquet | XGBoost + StandardScaler |
| TIMESERIES | Has datetime column | Transformer + Window |
| IMAGE | Folder of JPG/PNG | ResNet50 Transfer |
| TEXT | CSV with text columns | BERT Classifier |

---

## Option 2: Interactive TUI

Launch the Terminal User Interface for a guided experience:

```bash
kladml ui
```

---

## Option 3: Traditional Workflow

### Initialize a Project

```bash
kladml init
```

This creates the standard directory structure:

```
data/
├── kladml.sqlite        # Local database
├── configs/             # YAML configurations
├── datasets/            # Your data
└── projects/            # Training results
    └── {project}/
        └── {run_id}/
            ├── config.yaml
            ├── checkpoints/
            ├── exports/
            └── evaluations/
```

### Train with Config

```bash
kladml train --config data/configs/my_experiment.yaml
```

Example config:

```yaml
project: my-project
experiment: baseline_v1

dataset: my_data/processed
architecture: TransformerAutoencoder
params:
  d_model: 64
  n_heads: 4

training:
  epochs: 50
  batch_size: 128

export:
  auto: true
  format: onnx

evaluation:
  auto: true
  evaluator: AnomalyEvaluator
```

### Evaluate a Run

```bash
kladml eval --run run_001 --evaluator AnomalyEvaluator --plots cdf,loglog
```

### Compare Runs

```bash
kladml compare --runs run_001,run_002 --metric val_loss
```

---

## Hyperparameter Tuning

Use Optuna integration for automated hyperparameter search:

```bash
kladml tune --config config.yaml --n-trials 50 --timeout 3600
```

---

## Create Custom Models

```python
from kladml import BaseArchitecture, MLTask

class MyModel(BaseArchitecture):
    
    @property
    def ml_task(self):
        return MLTask.CLASSIFICATION
    
    def train(self, X_train, y_train=None, **kwargs):
        # Your training logic
        return {"accuracy": 0.95}
    
    def predict(self, X, **kwargs):
        return predictions
    
    def evaluate(self, X_test, y_test=None, **kwargs):
        return {"accuracy": 0.93, "f1": 0.91}
    
    def save(self, path: str):
        # Save model artifacts
        pass
    
    def load(self, path: str):
        # Load model artifacts
        pass
```

Register it:

```bash
kladml register architecture --name MyModel --module my_model.MyModel
```

Then use it:

```bash
kladml train --config config.yaml  # config references "MyModel"
```

---

## Next Steps

- [Core Concepts](core_concepts.md) - Understand interfaces and architecture
- [Architecture](architecture.md) - Deep dive into model contracts
- [Roadmap](roadmap.md) - Planned features
- [CLI Reference](cli.md) - All available commands
