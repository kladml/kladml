# KladML Framework Roadmap

## Philosophy
- **Core Library** (`kladml`): Zero CLI deps, usable as backend base for platform
- **CLI Extension** (`kladml[cli]`): Typer + Rich + Textual for local development
- **Same DB Models**: SDK and Platform share entity definitions (SQLModel/Pydantic)
- **Convention over Configuration**: Minimal config, maximum automation
- **Universal MLOps**: Preprocessors, Architectures, Evaluators are reusable across datasets/projects
- **Zero to Training**: Quickstart that does everything automatically

## Core Principles
- **Decoupled Entities**: Preprocessor, Architecture, Evaluator are independent and reusable
- **Compatibility Checking**: Auto-validate that components work together
- **Pydantic Schemas**: All configs use Pydantic for validation + JSON schema generation
- **Registry Pattern**: Built-in + user-defined components, all discoverable
- **Progressive Disclosure**: Simple by default, powerful when needed

---

## Phase 1: Database Unification (High Priority)
- [ ] Consolidate databases into single `data/kladml.sqlite`
- [ ] Create SQLModel models matching platform schema:
    - [ ] Dataset (with versions, data_type)
    - [ ] Architecture (with param_schema, input_schema, model_type)
    - [ ] Project
    - [ ] Run (metrics as JSON, artifacts_path)
    - [ ] Preprocessor (module_path, input_types, output_schema)
    - [ ] Evaluator (module_path, task_types, plot_types)
    - [ ] Evaluation (run_id, evaluator_id, metrics, artifacts_path)
    - [ ] ProjectDataset, ProjectArchitecture (link tables)
- [ ] Abstract DB backend (SQLite dev ↔ Postgres prod)
- [ ] Migrate existing runs to new schema
- [ ] Remove/archive legacy MLflow DB

## Phase 2: Registry System
- [ ] `PREPROCESSOR_REGISTRY`: Built-in transforms (Interpolate, Scale, Window)
- [ ] `ARCHITECTURE_REGISTRY`: Built-in models (Transformer, LSTM, Gluformer, XGBoost, ResNet)
- [ ] `EVALUATOR_REGISTRY`: Built-in evaluators (Anomaly, TimeSeries, Classification, Regression)
- [ ] Dynamic loading via `module_path`
- [ ] CLI: `kladml list preprocessors/architectures/evaluators`
- [ ] CLI: `kladml register preprocessor --name X --module path.to.Class`

## Phase 3: Quickstart System ⭐ (Differentiator)
**Goal: "Zero to Training in 60 Seconds"**

### Data Type Detection
- [ ] `detect_data_type(path)` → TABULAR, TIMESERIES, IMAGE, TEXT
- [ ] Schema inference from CSV/Parquet (columns, types, target detection)
- [ ] Image folder structure detection
- [ ] Text format detection

### Task Detection
- [ ] Heuristics: label column → Classification, no label → Anomaly/Clustering
- [ ] Interactive prompts when ambiguous
- [ ] CLI: `kladml quickstart --data X --task Y`

### Pre-configured Pipelines
- [ ] (TABULAR, Classification) → XGBoost + ClassificationEvaluator
- [ ] (TABULAR, Regression) → XGBoost + RegressionEvaluator
- [ ] (TIMESERIES, Anomaly) → TransformerAutoencoder + AnomalyEvaluator
- [ ] (TIMESERIES, Forecast) → Gluformer + TimeSeriesEvaluator
- [ ] (IMAGE, Classification) → ResNet50Transfer + ClassificationEvaluator

### Quickstart Flow
```
kladml quickstart --data my_data.csv
→ Analyze data
→ Detect type + suggest task
→ Select pipeline
→ Preprocess + Train + Evaluate + Export
→ Generate HTML report
```

## Phase 4: Preprocessing Pipeline System
- [ ] YAML pipeline definition format
- [ ] Preprocessor chaining (InterpolateOutput → ScalerInput → WindowOutput)
- [ ] Schema validation between steps
- [ ] CLI: `kladml data process --pipeline <path> --dataset <name>`
- [ ] Auto-split into train/val/test
- [ ] Save preprocessing config with dataset for reproducibility

## Phase 5: CLI Commands
- [ ] `kladml train --config <path>` (calls run_training)
- [ ] `kladml evaluate --run <run_id> [--evaluator X] [--plots cdf,loglog]`
- [ ] `kladml export --run <run_id> --format <onnx|torchscript>`
- [ ] `kladml list runs/datasets/preprocessors/evaluators`
- [ ] `kladml compare --runs run_001,run_002 --metric val_loss`
- [ ] `kladml quickstart --data <path> [--task <task>]`

## Phase 6: Modular Evaluation System
- [ ] BaseEvaluator with Pydantic config
- [ ] PlotGenerator base class with `required_inputs`
- [ ] Plot Registry: histogram, cdf, loglog, loss_curve, comparison
- [ ] Auto-generate metrics based on task type:
    - Regression: MAE, RMSE, R²
    - Classification: Accuracy, F1, AUC, Confusion Matrix
    - Anomaly: Threshold, FPR, TPR
    - Clustering: Silhouette, Davies-Bouldin
- [ ] Evaluation versioning (each eval in own folder: `evaluations/eval_001/`)
- [ ] Evaluator ↔ Architecture compatibility check (task_types match)

## Phase 7: Auto-Discovery & Compatibility
- [ ] Dataset schema inference from parquet/csv/images
- [ ] Compatibility matrix:
    - Preprocessor.input_types ∩ Dataset.data_type
    - Preprocessor.output_schema == Architecture.input_schema
    - Architecture.model_type ∈ Evaluator.task_types
- [ ] Suggested components: "For this dataset, try these architectures"
- [ ] Validation before training: "Architecture X is not compatible with Dataset Y"

## Phase 8: Terminal UI (Textual)
- [ ] `kladml ui` launcher
- [ ] Project browser (tree view)
- [ ] Run detail view (metrics, plots inline)
- [ ] One-click export/evaluate buttons
- [ ] Live training monitor
- [ ] Component browser (preprocessors, architectures, evaluators)
- [ ] Quickstart wizard (guided flow)

## Phase 9: Run Comparison
- [ ] `kladml compare --runs run_001,run_002 --metric val_loss`
- [ ] Overlay Loss Curves (one line per run)
- [ ] Overlay CDF Curves (one curve per run)
- [ ] Bar Chart comparison (final metrics side-by-side)
- [ ] Comparison saved to `data/comparisons/<comparison_id>/`

---

## Engineering Excellence

### Testing Strategy
- [ ] Unit tests for each component (preprocessor, architecture, evaluator)
- [ ] Integration tests for full pipelines (quickstart end-to-end)
- [ ] Property-based tests for data transformations (Hypothesis)
- [ ] CI/CD with GitHub Actions

### Error Handling
- [ ] Custom exceptions hierarchy (`KladMLError`, `ConfigError`, `DataError`)
- [ ] User-friendly error messages with suggestions
- [ ] Graceful degradation (continue if optional step fails)

### Observability
- [ ] Structured logging (JSON format)
- [ ] Progress bars for long operations (Rich)
- [ ] Debug mode with verbose output

### Performance
- [ ] Lazy loading of heavy components (torch, sklearn)
- [ ] Caching of preprocessed data
- [ ] Parallel preprocessing when possible

### Documentation
- [ ] API reference (auto-generated from docstrings)
- [ ] User guide with examples
- [ ] Architecture decision records (ADRs)

---

## Versioning Strategy

### Dataset Versioning
- [ ] Auto-increment version after each preprocessing (`v1`, `v2`, ...)
- [ ] `latest` symlink to most recent version
- [ ] Explicit version reference: `dataset@v1`

### Component Versioning
- [ ] Semantic versioning for architectures/preprocessors/evaluators
- [ ] Stored in registry DB

### Run Tracking
- [ ] Each run records: dataset@version, architecture@version, config hash
- [ ] Reproducibility info: Python version, package versions

---

## Phase 10: Hyperparameter Tuning (Optuna Integration)
- [ ] `OptunaStudy` wrapper class for architecture-aware HP optimization
- [ ] Define `SearchSpace` per architecture type:
    ```python
    class TransformerSearchSpace(BaseSearchSpace):
        d_model: Tuple[int, int] = (64, 256)
        n_heads: Tuple[int, int] = (2, 8)
        n_layers: Tuple[int, int] = (2, 6)
        learning_rate: Tuple[float, float] = (1e-5, 1e-2)
    ```
- [ ] Pruning strategies (Median, Hyperband)
- [ ] CLI: `kladml tune --config <path> --n-trials 50 --timeout 3600`
- [ ] Save best trial as new config file
- [ ] Visualization: optimization history, parameter importance
- [ ] Multi-objective optimization (accuracy vs latency)
- [ ] Database backend for distributed tuning (SQLite/Postgres)

---

## Platform-Ready Architecture

KladML è progettato per essere **riutilizzabile come backend** della KladML Platform.

### Design Pattern
```
┌─────────────────────────────────────────────────────────────┐
│                     KladML Platform                          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  FastAPI Endpoints                                       ││
│  │  - Auth, Multi-tenancy, Billing, Job Queue               ││
│  │  - /api/train, /api/evaluate, /api/quickstart           ││
│  └─────────────────────────────────────────────────────────┘│
│                           ↓ imports                          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  KladML Core (this repo)                                 ││
│  │  - TrainingManager, Evaluator, QuickStart                ││
│  │  - Same code used by CLI                                 ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     KladML CLI                               │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  kladml train / quickstart / compare                     ││
│  └─────────────────────────────────────────────────────────┘│
│                           ↓ imports                          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  KladML Core (this repo)                                 ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### API Design Principles
- [ ] **No global state**: All operations receive explicit context
- [ ] **Dependency injection**: Components receive dependencies via constructor
- [ ] **Return values over side effects**: Functions return results, don't print
- [ ] **Async-ready**: Core operations are async or easily wrappable
- [ ] **Serializable results**: All results are Pydantic models (JSON-ready)

### Example: Platform-Ready Training
```python
# CLI calls this directly
result = await training_manager.train(config)

# Platform wraps same call with authentication, job queue, webhooks
@app.post("/api/train")
async def train_endpoint(config: TrainConfig, user: User = Depends(get_user)):
    job = await job_queue.enqueue(training_manager.train, config, user_id=user.id)
    return {"job_id": job.id}
```

### Out of Scope (Platform Only)
- **Model Serving API**: REST/gRPC inference endpoints
- **Multi-tenancy**: User isolation, permissions
- **Billing/Usage**: Compute tracking, quotas
- **Web Frontend**: React dashboard

---

## Explicitly Out of Scope (for now)
- **Security/GDPR**: Open source project, users manage their own data
- **Multi-GPU/Distributed Training**: Single-node focus for simplicity

---

## Completed ✅
- [x] ONNX export with validation
- [x] run_training() auto-export
- [x] TorchExportMixin
- [x] Test suite for export
- [x] Requirements updated (torch, onnx, Python <3.13)
- [x] Mean + 3σ threshold formula
- [x] Multiple plot types (Histogram, CDF, LogLog, LogX)
- [x] Fixed run_id generation (now increments correctly: run_001, run_002...)
- [x] ARCHITECTURE.md with conventions and design decisions

