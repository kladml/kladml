# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
