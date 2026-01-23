# KladML Framework Roadmap (Reorganized)

## Philosophy
- **Core Library** (`kladml`): Zero CLI deps, usable as backend base for platform
- **CLI Extension** (`kladml[cli]`): Typer + Rich + Textual for local development
- **Same DB Models**: SDK and Platform share entity definitions (SQLModel/Pydantic)
- **Convention over Configuration**: Minimal config, maximum automation
- **Universal MLOps**: Preprocessors, Architectures, Evaluators are reusable across datasets/projects

## Core Principles
- **Decoupled Entities**: Preprocessor, Architecture, Evaluator are independent and reusable
- **Compatibility Checking**: Auto-validate that components work together
- **Pydantic Schemas**: All configs use Pydantic for validation + JSON schema generation
- **Registry Pattern**: Built-in + user-defined components, all discoverable
- **Progressive Disclosure**: Simple by default, powerful when needed

---

## Phase 1: Database Unification (High Priority)
- [x] Consolidate databases into single `data/kladml.sqlite`
- [x] Create SQLModel models matching platform schema:
    - [x] Dataset (with versions, data_type)
    - [ ] Architecture (with param_schema, input_schema, model_type)
    - [x] Project
    - [x] Family (as grouping entity)
    - [x] Run (migrated to MLflow native tables in same DB)
    - [ ] Preprocessor (module_path, input_types, output_schema)
    - [ ] Evaluator (module_path, task_types, plot_types)
    - [ ] Evaluation (run_id, evaluator_id, metrics, artifacts_path)
    - [ ] ProjectDataset, ProjectArchitecture (link tables)
- [x] Abstract DB backend (SQLite dev ↔ Postgres prod)
- [x] Migrate existing runs to new schema
- [x] Remove/archive legacy MLflow DB

## Phase 2: Registry System
- [ ] `PREPROCESSOR_REGISTRY`: Built-in transforms (Interpolate, Scale, Window)
- [ ] `ARCHITECTURE_REGISTRY`: Built-in models (Transformer, LSTM, Gluformer, XGBoost, ResNet)
- [ ] `EVALUATOR_REGISTRY`: Built-in evaluators (Anomaly, TimeSeries, Classification, Regression)
- [ ] Dynamic loading via `module_path`
- [ ] CLI: `kladml list preprocessors/architectures/evaluators`
- [ ] CLI: `kladml register preprocessor --name X --module path.to.Class`

## Phase 3: Core CLI Commands (High Priority) ⭐
**Goal: Solid foundation for all workflows**

- [ ] `kladml train --config <path>` - Core training command
- [ ] `kladml evaluate --run <run_id> [--evaluator X] [--plots cdf,loglog]`
- [ ] `kladml export --run <run_id> --format <onnx|torchscript>`
- [ ] `kladml list runs/datasets/architectures/evaluators`
- [x] `kladml compare --runs run_001,run_002 --metric val_loss`
- [ ] Robust error handling with helpful messages
- [ ] Progress indicators (Rich progress bars)

## Phase 4: Preprocessing Pipeline System
- [ ] YAML pipeline definition format
- [ ] Preprocessor chaining (InterpolateOutput → ScalerInput → WindowOutput)
- [ ] Schema validation between steps
- [ ] CLI: `kladml data process --pipeline <path> --dataset <name>`
- [ ] Auto-split into train/val/test
- [ ] Save preprocessing config with dataset for reproducibility

## Phase 5: Modular Evaluation System
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

## Phase 6: Auto-Discovery & Compatibility
- [ ] Dataset schema inference from parquet/csv/images
- [ ] Compatibility matrix:
    - Preprocessor.input_types ∩ Dataset.data_type
    - Preprocessor.output_schema == Architecture.input_schema
    - Architecture.model_type ∈ Evaluator.task_types
- [ ] Suggested components: "For this dataset, try these architectures"
- [ ] Validation before training: "Architecture X is not compatible with Dataset Y"

## Phase 7: Terminal UI Improvements
- [x] `kladml ui` launcher (already exists)
- [ ] Project browser (tree view)
- [ ] Run detail view (metrics, plots inline)
- [ ] One-click export/evaluate buttons
- [ ] Live training monitor with real-time plots
- [ ] Component browser (preprocessors, architectures, evaluators)

## Phase 8: Smart Config Generator & Quickstart (Marketing)
**Goal: "Wow factor" for demos + onboarding**

### 8a. Config Generator (Priority)
- [ ] `kladml config generate --data <path>` - Smart config template generator
- [ ] Data type detection (`detect_data_type()`)
- [ ] Task suggestion heuristics
- [ ] Template selection based on (data_type, task) pair
- [ ] Interactive prompts for customization
- [ ] Output: Ready-to-use config YAML

### 8b. Quickstart (Nice to Have)
- [ ] `kladml quickstart --data <path>` - One-command demo
- [ ] Thin wrapper: detect → generate config → train
- [ ] Pre-configured pipelines for common cases:
    - (TABULAR, Classification) → XGBoost + ClassificationEvaluator
    - (TABULAR, Regression) → XGBoost + RegressionEvaluator
    - (TIMESERIES, Anomaly) → TransformerAutoencoder + AnomalyEvaluator
    - (TIMESERIES, Forecast) → Gluformer + TimeSeriesEvaluator
    - (IMAGE, Classification) → ResNet50Transfer + ClassificationEvaluator

**Important**: Quickstart does NOT duplicate logic, it just calls existing commands.

## Phase 9: Run Comparison
- [ ] `kladml compare --runs run_001,run_002 --metric val_loss`
- [ ] Overlay Loss Curves (one line per run)
- [ ] Overlay CDF Curves (one curve per run)
- [ ] Bar Chart comparison (final metrics side-by-side)
- [ ] Comparison saved to `data/comparisons/<comparison_id>/`

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
- [x] Terminal UI basic implementation (kladml ui)
- [x] CLI project/family/experiment management
