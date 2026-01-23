# KladML

**Universal MLOps: Zero to Training in 60 Seconds**

---

## What is KladML?

KladML is a modular SDK for building production-ready machine learning pipelines. Unlike heavy MLOps frameworks, KladML gives you:

- **Universal Quickstart** - Auto-detect data type, suggest pipeline, train in one command
- **Interface-based architecture** - Swap backends without changing code
- **Local-first** - No servers required, works offline with SQLite
- **Extensible** - Register custom architectures, preprocessors, and evaluators
- **CLI included** - Initialize projects, run experiments from terminal

## Quick Install

```bash
# Core library
pip install kladml

# Full CLI with TUI
pip install "kladml[cli]"
```

## Quick Start

### Zero to Training in 60 Seconds

```bash
kladml quickstart --data my_data.csv

# Output:
# 📊 Analyzing data...
#    Data type: TABULAR (5 columns, 1000 rows)
#
# ? What task do you want to perform?
#   > Classification (detected 'label' column)
#
# 🔧 Selected: XGBoostClassifier + ClassificationEvaluator
# 🚀 Training...
# ✅ Complete! Results saved to data/projects/quickstart/run_001/
```

## Supported Data Types

| Data Type | Auto-Detection | Default Pipeline |
|-----------|----------------|------------------|
| **TABULAR** | Numeric CSV/Parquet | XGBoost |
| **TIMESERIES** | Has datetime column | Transformer/Gluformer |
| **IMAGE** | Folder with JPG/PNG | ResNet50 |
| **TEXT** | CSV with text columns | BERT |

## Why KladML?

| Feature | KladML | MLflow | ClearML |
|---------|--------|--------|---------|
| **Interface-based** | ✅ Pluggable | ❌ Hardcoded | ❌ Hardcoded |
| **Server required** | ❌ No | ⚠️ Optional | ✅ Yes |
| **Local-first** | ✅ SQLite default | ✅ Yes | ❌ No |
| **Learning curve** | 🟢 Minutes | 🟡 Days | 🔴 Weeks |
| **Universal Quickstart** | ✅ Yes | ❌ No | ❌ No |

## Documentation

- 🚀 **[Getting Started](getting_started.md)** — Install, configure, and run your first experiment
- 🧠 **[Core Concepts](core_concepts.md)** — Understand interfaces, runners, and the architecture
- 🏗️ **[Model Architecture](architecture.md)** — Deep dive into model contracts and design patterns
- 🗺️ **[Roadmap](roadmap.md)** — Planned features and what's coming next
- 📦 **[CLI Reference](cli.md)** — All available commands and options
- 🚢 **[Deployment](deployment.md)** — Export and deploy to edge devices

## Links

- [GitHub Repository](https://github.com/kladml/kladml)
- [PyPI Package](https://pypi.org/project/kladml/)
- [Report Issues](https://github.com/kladml/kladml/issues)
