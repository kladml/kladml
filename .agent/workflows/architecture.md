---
description: Architectural patterns and component implementation guides
---

# KladML Architecture Constraints & Guides

Use this guide when implementing new components (Architectures, Evaluators, Preprocessors) to ensure they fit the platform design.

## Core Design Patterns

### 1. Dependency Injection
**Rule:** Ensure all dependencies are passed via `__init__`. Do not instantiate heavy dependencies globally.

```python
# BAD
db = Database()
class MyComponent:
    def process(self):
        db.save(...)

# GOOD
class MyComponent:
    def __init__(self, db: DatabaseInterface):
        self.db = db
```

### 2. Registry Pattern
**Rule:** All new components must be registered to be discoverable by CLI and Config.

```python
# In src/kladml/architectures/__init__.py or similar
from .my_model import MyModel

ARCHITECTURE_REGISTRY = {
    "MyModel": MyModel
}
```

### 3. Pydantic Configuration
**Rule:** Every component must have a corresponding Pydantic config model if it has parameters.

```python
class MyModelConfig(BaseModel):
    d_model: int = 64
    dropout: float = 0.1
```

## Implementing Components

### Architecture
Must inherit from `BaseArchitecture` and implement:
- `process_batch(batch)` -> loss/output
- `configure_optimizers()`
- `export(path)`

### Evaluator
Must inherit from `BaseEvaluator` and implement:
- `evaluate(model, dataloader)` -> metrics dict
- `plot(results)` -> list of plot paths

## Directory Structure Enforcement

When adding files, strictly follow:
- `src/kladml/architectures/` - Neural network definitions
- `src/kladml/evaluation/` - Evaluators and metrics
- `src/kladml/data/` - Preprocessors and DataLoaders
- `src/kladml/interfaces/` - Abstract base classes (DO NOT CHANGE EXISTING ONES)

## Database Schema (SQLModel)
When modifying DB models in `src/kladml/db/models.py`, ensure matching Pydantic schemas are updated.

## Event System (Callbacks)
If implementing long-running tasks (Training, Export), emit standard events so the TUI/Platform can track progress.

```python
# Emit events
callback_handler.on_progress(current=..., total=...)
```
