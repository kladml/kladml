# KladML

**Build ML pipelines with pluggable backends. Simple. Modular. Yours.**

---

## What is KladML?

KladML is a lightweight, modular SDK for building production-ready machine learning pipelines. Unlike heavy MLOps platforms, KladML gives you:

- **Interface-based architecture** - Swap backends without changing code
- **Local-first** - No servers required, works offline with SQLite
- **Framework-agnostic** - Works with PyTorch, TensorFlow, scikit-learn, or any ML library
- **CLI included** - Initialize projects, run experiments from terminal

## Quick Install

```bash
pip install kladml
```

## Quick Start

```bash
# Create a project
kladml init my-project
cd my-project

# Run training
kladml run native train.py
```

## Why KladML?

| Feature | KladML | MLflow | ClearML |
|---------|--------|--------|---------|
| **Interface-based** | ✅ Pluggable | ❌ Hardcoded | ❌ Hardcoded |
| **Server required** | ❌ No | ⚠️ Optional | ✅ Yes |
| **Local-first** | ✅ SQLite default | ✅ Yes | ❌ No |
| **Learning curve** | 🟢 Minutes | 🟡 Days | 🔴 Weeks |
| **Custom backends** | ✅ Easy | ⚠️ Complex | ❌ No |

## Documentation

- 🚀 **[Getting Started](getting_started.md)** — Install, configure, and run your first experiment
- 🧠 **[Core Concepts](core_concepts.md)** — Understand interfaces, runners, and the architecture
- 🏗️ **[Architecture](architecture.md)** — Deep dive into model contracts and design patterns
- 📦 **[CLI Reference](cli.md)** — All available commands and options

## Example

```python
from kladml import TimeSeriesModel, ExperimentRunner

class MyForecaster(TimeSeriesModel):
    def train(self, X_train, y_train=None, **kwargs):
        # Your training logic
        return {"loss": 0.1}
    
    def predict(self, X, **kwargs):
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test=None, **kwargs):
        return {"mae": 0.5}
    
    def save(self, path: str):
        # Save model
        pass
    
    def load(self, path: str):
        # Load model
        pass

# Run with tracking
runner = ExperimentRunner()
runner.run(model_class=MyForecaster, train_data=data)
```

## Links

- [GitHub Repository](https://github.com/kladml/kladml)
- [PyPI Package](https://pypi.org/project/kladml/)
- [Report Issues](https://github.com/kladml/kladml/issues)
