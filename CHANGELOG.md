# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-01-14

### Added
- Initial release
- Core interfaces: `StorageInterface`, `ConfigInterface`, `PublisherInterface`, `TrackerInterface`
- Default backends: `LocalStorage`, `YamlConfig`, `ConsolePublisher`, `LocalTracker`
- `ExperimentRunner` for orchestrating ML experiments
- Base model classes: `BaseArchitecture`, `BasePreprocessor`, `TimeSeriesModel`, `ClassificationModel`
- CLI with commands: `init`, `version`, `run native/local`
- GitHub Actions workflow for PyPI publication
- Full documentation and examples
