# Roadmap

What's coming in future versions of KladML.

---

## Current Version: v0.5.x

### ✅ Available Now

- **ONNX Export** - Export models to ONNX format for edge deployment
- **TorchScript Export** - Self-contained deployment artifacts
- **Multiple Evaluators** - Anomaly, Classification, Regression evaluation
- **Plot Types** - Histogram, CDF, Log-Log, Loss Curves
- **Run Tracking** - Automatic run ID generation and tracking
- **Early Stopping** - Configurable patience and minimum delta
- **Registry System** - Architecture and component registration
- **Terminal UI (TUI)** - Interactive terminal interface (`kladml ui`)
- **CLI Commands** - train, evaluate, data, project, family, experiment

---

## v0.6.x - Core CLI Enhancements

Complete the core command-line workflow:

- [ ] **Enhanced Training** - `kladml train --config <path>` with better progress tracking
- [ ] **Run Evaluation** - `kladml evaluate --run <run_id> --evaluator <name>`
- [ ] **Model Export** - `kladml export --run <run_id> --format <onnx|torchscript>`
- [ ] **Run Comparison** - `kladml compare --runs run_001,run_002`
- [ ] **Component Listing** - `kladml list <runs|datasets|architectures>`
- [ ] **Better Error Messages** - Helpful suggestions when something goes wrong
- [ ] **Progress Indicators** - Real-time progress for long operations

---

## v0.7.x - Preprocessing Pipelines

Composable data preprocessing:

- [ ] **YAML Pipeline Definition** - Define preprocessing steps in config
- [ ] **Component Chaining** - Link preprocessors with schema validation
- [ ] **CLI Command** - `kladml data process --pipeline <config>`
- [ ] **Auto-split** - Train/val/test splitting with reproducibility

---

## v0.8.x - TUI Improvements

Enhancements to the existing Terminal UI:

- [ ] **Live Training Monitor** - Real-time loss curves in terminal
- [ ] **Inline Plots** - View evaluation plots directly in TUI
- [ ] **One-click Actions** - Export, evaluate, compare from UI
- [ ] **Improved Navigation** - Better keyboard shortcuts

---

## v0.9.x - Smart Config Generator

Intelligent configuration assistant:

- [ ] **Data Analysis** - Auto-detect data type (tabular, timeseries, image, text)
- [ ] **Task Suggestion** - Recommend task based on data characteristics
- [ ] **Config Templates** - Generate ready-to-use configuration files
- [ ] **CLI Command**:
  ```bash
  kladml config generate --data my_data.csv
  # Output: configs/my_data_template.yaml
  ```

---

## v0.10.x - Universal Quickstart (Optional)

One-command training for demos and onboarding:

- [ ] **Quickstart Command** - `kladml quickstart --data <path>`
- [ ] **Auto-workflow** - Detect → Generate config → Train → Evaluate
- [ ] Pre-configured pipelines for common cases

**Note**: This is a thin wrapper around existing commands, primarily for marketing and demos.

---

## v1.0.0 - Stable Release

First stable release with:

- [ ] Comprehensive documentation
- [ ] Full test coverage
- [ ] API stability guarantees
- [ ] Migration guides from previous versions

---

## Future Ideas

These are being considered but not yet planned:

| Feature | Description |
|---------|-------------|
| **Hyperparameter Tuning** | Optuna integration for automated HPO |
| **Distributed Training** | Multi-GPU support |
| **Remote Execution** | Run on cloud infrastructure |
| **Model Versioning** | Track model versions with metadata |
| **Experiment Templates** | Reusable experiment configurations |

---

## Contributing

Want to help shape the roadmap? 

- 💡 [Open a Feature Request](https://github.com/kladml/kladml/issues/new)
- 🐛 [Report a Bug](https://github.com/kladml/kladml/issues)
- 🤝 [Contribute Code](https://github.com/kladml/kladml/blob/main/CONTRIBUTING.md)

---

## Changelog

See [CHANGELOG.md](https://github.com/kladml/kladml/blob/main/CHANGELOG.md) for detailed version history.
