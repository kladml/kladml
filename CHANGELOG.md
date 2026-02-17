# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- **Evaluation:** TimeSeriesEvaluator for forecasting tasks with metrics (MAE, MSE, RMSE, MAPE, R2) and 4 diagnostic plots (line, scatter, residuals over time, error distribution).
- **Evaluation:** Registered RegressionEvaluator in EvaluatorRegistry (was implemented but not registered).
- **Evaluation:** Registered TimeSeriesEvaluator in EvaluatorRegistry for `MLTask.TIMESERIES_FORECASTING`.
- **Tests:** CLI tests for evaluate command with coverage for classification, regression, and timeseries tasks.
- **Docs:** New `evaluation.md` documentation page with evaluator usage guide and custom evaluator examples.
- **Docs:** Added evaluation documentation to mkdocs navigation.
- **Utils:** `detect_ml_task()` utility function for auto-detecting ML task from model or checkpoint metadata.
- **Checkpoint:** CheckpointManager now saves task type in metadata.json when model has `ml_task` property.
- **Export:** ScalerWrapper module for embedding sklearn scalers in TorchScript exports.
- **Export:** `--embed-scaler` CLI flag to include normalization in exported models.

### Changed
- **CLI:** `evaluate` command now auto-detects task from model or checkpoint metadata instead of defaulting to classification.
- **CLI:** `export` command now reads configuration from metadata.json and checkpoint config, with `--architecture` and `--config` override options.
- **Export:** TorchScriptExporter now supports optional scaler embedding via `scaler` parameter.

## [0.13.1] - 2026-01-26
### Added
- **Core:** Included `inspection.py` and `probabilistic.py` missed in v0.13.0.
- **Utils:** Smart config generator now fully supported with inspection utilities.
- **Metrics:** Probabilistic evaluation metrics (NLL, MSE) officially part of core.

## [0.13.0] - 2026-01-26
### Added
- **Gluformer Integration:** Full lifecycle support (Train, Predict, Export) via `UniversalTrainer`.
- **Integration Tests:** New test suite `tests/integration/` validating the complete pipeline (Model Lifecycle & Data Pipeline).
- **Data Pipeline:** Verified and hardened PKL->HDF5/Parquet pipeline with high test coverage (>80%).
- **CLI:** Enhanced `kladml config create` for smart dataset-aware configuration.

### Changed
- **Registry:** Renamed `kladml registry register` to `kladml registry add`.
- **Codebase Cleanliness:** Removed legacy `src/kladml/data/utils/` and moved scripts to `scripts/`.
- **Trainer:** Improved `UniversalTrainer` to support wrapper models (like Gluformer) natively.
- **Coverage:** Reached >80% coverage on Core, CLI, and Data modules.

## [0.12.1] - 2026-01-25
### Added
- **Smart Config Generator:** New `kladml config create` command that inspects datasets (Polars) and generates optimized initial configurations.
- **Model Defaults:** `BaseModel.default_config` interface for architectural defaults.


### Architecture
- **Operation 100%:** Achieved high test coverage on Core Logic (>80%) and Utilities (96%).
- **CLI Refactor:** Split monolithic `train.py` and `data.py` into modular command packages.
- **BaseModel:** Enhanced with native Observer Pattern for callbacks, enabling isolated testing.
- **Utilities:** Hardened `config_io`, `loading`, and `paths` modules against edge cases.

### Fixed
- **CallbackList:** Implemented full list behavior for robust event delegation.
- **Tuner Tests:** Fixed SQLite concurrency issues in test suite.
## [0.11.0] - 2026-01-25
### Added
- **Hyperparameter Tuning:** Added `kladml tune` command powered by Optuna.
- **Automated Pruning:** `UniversalTrainer` now supports `OptunaPruningCallback` to stop bad runs early.
- **CLI:** Support for tuning `search_space` definition in YAML configs.
- **Utils:** Refactored dynamic model loading into `kladml.utils.loading`.

## [0.10.3] - 2026-01-25
### Fixed
- Updated CLI examples to use more realistic paths (`data/datasets/canbus`).

## [0.10.2] - 2026-01-25
### Added
- **Declarative Pipelines:** Added `kladml data process` command to run YAML-based preprocessing pipelines.
- **Component Registry:** Auto-registration via `@register_component` decorator.
- **Standardized I/O:** Components now exchange `pl.DataFrame` structs for maximum interoperability.

## [0.10.1] - 2026-01-25
### Added
- **High Performance:** Added `torch.compile` support in `UniversalTrainer` (`compile: true` in config) for 30%+ training speedup on supported hardare.
- **Generic Layers:** Unified `MultiheadAttention` with native Flash Attention (SDPA) support in `src/kladml/models/layers`.
- **Data Engine:** Integrated **Polars** and **Parquet** support.
    - `kladml data convert --format parquet`: Convert legacy datasets to efficient Parquet.
    - `kladml data inspect`: Instant analysis of Parquet files using Polars engine.
    - Deprecated `.pkl` datasets in favor of Parquet.

### Changed
- **Data Pipeline:** Complete refactor from Pandas to **Polars** for all components (`J1939Parser`, `Cleaner`, `Resampler`, `Splitter`).
    - Faster processing (up to 50x)
    - Memory efficient
    - Strict schema validation
- **Dependencies:** Removed hard dependency on Pandas (it is now optional/transitive). Added `polars` and `pyarrow`.

## [0.9.0] - 2026-01-25

### Added
- **Modern Training Loop:** Integrated Hugging Face `Accelerate` for simplified mixed precision and distributed training.
- **CLI Integration:** Added `--distributed` and `--num-processes` flags to `kladml train` for seamless multi-GPU launching.
- **Training Features:** Added support for Gradient Accumulation (`gradient_accumulation_steps`) and Gradient Clipping in `TrainingConfig`.
- **Performance:** Native support for Mixed Precision (`fp16`, `bf16`).

## [0.8.0] - 2026-01-25

### Added
- **Evaluator System:** Modular evaluation with `ClassificationEvaluator` and `RegressionEvaluator`.
- **Dependencies:** Added `torchmetrics` and `matplotlib` for evaluation and plotting.
- **CLI:** New `kladml eval` command.

## [0.7.0] - 2026-01-24

### Added
- **Typed Configuration:** Migrated all configs to **Pydantic v2 Models** (`TrainingConfig`, `ModelConfig`) for strict validation and improved type safety.
- **Evaluator Hierarchy:** Introduced structured evaluator system under `src/kladml/evaluation/` (Classification, Regression, TimeSeries).
- **Test Coverage:** Maximized coverage on critical paths (RunID, Trainer, Checkpoints, Family CLI).
- **Artifact Logging:** Added support for logging artifacts and models in `LocalTracker`.

### Changed
- **Dependencies:** Dropped support for Python 3.9 (Minimum is now 3.10).

- **Trainer:** `UniversalTrainer` now accepts typed config objects.
- **Testing:** Reorganized test suite into `unit/{models,training,cli}` structure.

### Fixed
- **PyPI Display:** Fixed broken logo image on PyPI by using absolute GitHub URL.
- **CLI Imports:** Fixed `ImportError` in `test_family_cli` caused by missing interface exports.

## [0.6.0] - 2026-01-24

### Added
- **Vision Support:** Added optional `vision` dependency group (Torchvision, Albumentations) and backend support for Lazy Loading.
- **Callbacks Package:** Refactored monothilic callbacks into a modular package structure.
- **Data Module:** Introduced scalable DataModule hierarchy.

### Changed
- **Architecture Refactor:** Renamed `BaseArchitecture` to `BaseModel` and Unified model structure under `src/kladml/models/`.
- **Registry:** Standardized artifact storage location to `registry/`.
- **Tracking:** Improved default tracking directory structure.
- **Documentation:** Comprehensive updates to CLI and Architecture guides.

### Fixed
- **PyPI Compliance:** License and versioning improvements.

## [0.5.5] - 2026-01-23

### Fixed
- **Internal:** Minor fixes and dependency adjustments.

## [0.5.4] - 2026-01-23

### Fixed
- **PyPI Compliance:** Updated license configuration in `pyproject.toml` and correct version format.

## [0.5.3] - 2026-01-23

### Added
- **Unified Database:** Integrated KladML metadata and MLflow tracking into a single SQLite database (`~/.kladml/kladml.db`), simplifying configuration and backups.
- **Compare Command:** New `kladml compare` CLI command for side-by-side run comparison of metrics and parameters.
- **TUI Updates:** Added multi-select and comparison interface to the Terminal User Interface (`kladml ui`), enabling direct run analysis.
- **Migration Tool:** Added `scripts/migrate_mlruns_to_sqlite.py` to seamlessly migrate existing MLflow FileStore data to the unified DB.

### Changed
- **Pydantic V2:** Updated configuration settings to fully comply with Pydantic V2 standards, removing deprecation warnings.
- **Date Handling:** Standardized on `timezone.utc` for internal timestamps, resolving datetime warnings.
- **Dependencies:** Added `sqlmodel` as a core dependency for the unified database layer.

## [0.5.2] - 2026-01-23

### Added
- **Data Pipeline CLI:** New `kladml data process` command for running YAML-defined preprocessing pipelines.
- **Auto-Export:** Added `run_training()` method with automatic ONNX export after training.
- **Family Support:** Run ID generation now supports optional `family_name` for better organization.
- **Public Roadmap:** Added user-facing roadmap (`docs/roadmap.md`) with reorganized priorities.

### Changed
- **Documentation Reorganization:** Moved internal docs to `private_docs/`, cleaned all public docs from platform references.
- **Roadmap Priorities:** Repositioned "Quickstart" as Phase 8 (marketing feature), promoted "Core CLI Commands" to Phase 3.
- **Export Default:** Changed default export format from TorchScript to ONNX.
- **Project Path:** Updated default run path from `./projects` to `data/projects`.
- **Python Support:** Added Python 3.12 classifier to `pyproject.toml`.

### Improved
- **CLI Documentation:** Comprehensive update to `docs/cli.md` with all commands (quickstart, tune, compare, register).
- **Getting Started:** Complete rewrite focusing on quickstart workflow and practical examples.
- **Error Messages:** Better method documentation in `BaseArchitecture`.

### Fixed
- **Examples:** Added `examples/` to `.gitignore` (volant scripts to be integrated into CLI).

## [0.5.1] - 2026-01-18

### Fixed
- **Source Sync:** Included missing TUI source files (`app.py`) in the package distribution.

## [0.5.0] - 2026-01-18

### Added
- **TUI (Interactive Workspace):** New `kladml ui` command for exploring workspace visually (Projects, Families, Datasets, Configs).
- **Datasets Management:** Added `Dataset` entity to database, auto-sync from `data/datasets`, and TUI integration.
- **Configs View:** Added TUI support for viewing configuration files.
- **Dependency Split:** Core vs CLI split in `pyproject.toml` (install `kladml[cli]` for UI).

### Changed
- **Hierarchy:** Refined concept to "Workspace > Projects > Family > Experiment".
- **Refactor:** `ConsolePublisher` now robust to missing `rich` library.

### Fixed
- Run ID generation now correctly detects existing directories (fixing `run_001` duplicates).
- Fixed missing parameters in re-imported runs (fallback to `training.jsonl`).
- Resolved `datasets` table collision with MLflow (renamed to `local_datasets`).

## [0.4.0] - 2026-01-18

### Added
- **Family Entity:** Introduced `Family` layer between Project and Experiment for better grouping.
- **Metadata Interface:** Abstracted metadata storage (Project/Family/Experiment relationships).
- **Modular Evaluation:** New evaluation system with CLI support.

### Changed
- **Architecture:** Major refactor to support Family-based structure.
- **Tracker:** Added custom `run_id` support.

## [0.3.0] - 2026-01-15

### Added
- **Deployment Export:** Automatic export of `best_model_jit.pt` for deployment.
- **Standardized Callbacks:** Uniform training lifecycle events.
- **Uncertainty Visualization:** Guides for frontend integration.

## [0.2.0] - 2026-01-10

### Added
- **HDF5 Support:** Lazy loading for large datasets (`kladml data convert`).
- **Model Agnostic CLI:** Generic training commands (`kladml train`).
- **Local Data Management:** `kladml init` command.

## [0.1.1] - 2026-01-05

### Fixed
- CI/CD workflow fixes for PyPI publishing.
- Documentation link updates.

## [0.1.0] - 2024-01-14

### Added
- Initial release.
- Core Interfaces: Storage, Config, Publisher, Tracker.
- Basic Backends & ExperimentRunner.

### Added
- Initial release
- Core interfaces: `StorageInterface`, `ConfigInterface`, `PublisherInterface`, `TrackerInterface`
- Default backends: `LocalStorage`, `YamlConfig`, `ConsolePublisher`, `LocalTracker`
- `ExperimentRunner` for orchestrating ML experiments
- Base model classes: `BaseArchitecture`, `BasePreprocessor`, `TimeSeriesModel`, `ClassificationModel`
- CLI with commands: `init`, `version`, `run native/local`
- GitHub Actions workflow for PyPI publication
- Full documentation and examples
