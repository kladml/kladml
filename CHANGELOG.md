# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
