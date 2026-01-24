# CLI Reference

KladML includes a command-line interface for common tasks.

## Hierarchy Overview

KladML organizes work in a 4-level hierarchy:

```
Workspace > Projects > Family > Experiment > Run
```

- **Project**: Top-level container (e.g., `sentinella`)
- **Family**: Groups related experiments (e.g., `glucose_forecasting`)
- **Experiment**: A specific model/approach (e.g., `gluformer_v4`)
- **Run**: Single training execution with specific params

---

## Global Commands

### `kladml ui`

Launch the interactive Terminal User Interface (TUI). This provides a visual workspace to explore:
- Projects, Families, and Experiments (Tree View)
- Datasets (List and details)
- Configs (File viewer)
- Run Details (Parameters and Metrics)

```bash
kladml ui
```

### `kladml version`

Show the installed version.

```bash
kladml version
```

### `kladml --help`

Show all available commands.

---

## Project Commands

### `kladml project create`

Create a new project.

```bash
kladml project create <name> [--description TEXT]
```

### `kladml project list`

List all projects.

```bash
kladml project list
```

### `kladml project show`

Show project details.

```bash
kladml project show <name>
```

### `kladml project delete`

Delete a project.

```bash
kladml project delete <name> [--force]
```

---

## Family Commands

Families group related experiments within a project.

### `kladml family create`

Create a new family under a project.

```bash
kladml family create -p <project> -n <name> [-d DESCRIPTION]
```

**Example:**

```bash
kladml family create -p sentinella -n glucose_forecasting -d "Blood glucose prediction models"
```

### `kladml family list`

List families in a project.

```bash
kladml family list -p <project>
```

### `kladml family delete`

Delete a family.

```bash
kladml family delete <name> -p <project> [--force]
```

---

## Experiment Commands

### `kladml experiment create`

Create a new experiment under a family.

```bash
kladml experiment create -p <project> -f <family> -n <name>
```

**Example:**

```bash
kladml experiment create -p sentinella -f glucose_forecasting -n gluformer_v4
```

### `kladml experiment list`

List experiments (grouped by family).

```bash
kladml experiment list -p <project> [-f <family>]
```

### `kladml experiment runs`

List runs in an experiment.

```bash
kladml experiment runs <experiment-name> [--max N]
```

---

## Run Commands

### `kladml run native`

Run a training script using your local Python environment.

```bash
kladml run native <script> [OPTIONS]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--experiment`, `-e` | `default` | Experiment name for tracking |

**Example:**

```bash
kladml run native train.py --experiment baseline
```

---

### `kladml run local`

Run a training script inside a Docker/Podman container.

```bash
kladml run local <script> [OPTIONS]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `script` | Path to the Python script to run |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--experiment`, `-e` | `default` | Experiment name for tracking |

**Example:**

```bash
kladml run native train.py --experiment baseline
```

---

### `kladml run local`

Run a training script inside a Docker/Podman container.

```bash
kladml run local <script> [OPTIONS]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `script` | Path to the Python script to run |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--device`, `-d` | `auto` | Device to use: `auto`, `cpu`, `cuda`, `mps` |
| `--runtime`, `-r` | `auto` | Container runtime: `auto`, `docker`, `podman` |
| `--image`, `-i` | (auto) | Custom Docker image to use |

**Examples:**

```bash
# Auto-detect runtime and device
kladml run local train.py

# Force CUDA and Docker
kladml run local train.py --device cuda --runtime docker

# Use custom image
kladml run local train.py --image my-registry/my-image:latest
```

**Default Images:**

| Device | Image |
|--------|-------|
| `cpu` | `ghcr.io/kladml/worker:cpu` |
| `cuda` | `ghcr.io/kladml/worker:cuda12` |
| `mps` | `ghcr.io/kladml/worker:cpu` (fallback) |

---


---

## Training Commands

### `kladml train quick`

**Recommended** - Quick training without database setup.

```bash
kladml train quick [OPTIONS]
```

**Options:**

| Option | Required | Description |
|--------|----------|-------------|
| `--config`, `-c` | Yes | Path to YAML config file |
| `--train`, `-t` | Yes | Path to training data (`.pkl` or `.h5`) |
| `--val`, `-v` | No | Path to validation data |
| `--model`, `-m` | No | Model name (default: `gluformer`) |
| `--device`, `-d` | No | Device: `auto`, `cpu`, `cuda`, `mps` |
| `--resume`, `-r` | No | Resume from latest checkpoint |

**Examples:**

```bash
# Basic training
kladml train quick -c data/configs/my_config.yaml -t train.pkl -v val.pkl

# Resume interrupted training
kladml train quick -c data/configs/my_config.yaml -t train.pkl --resume
```

---

### `kladml train single`

Full training with project and experiment tracking (requires database setup).

```bash
kladml train single [OPTIONS]
```

**Options:**

| Option | Required | Description |
|--------|----------|-------------|
| `--model`, `-m` | Yes | Model architecture name (e.g., `gluformer`) |
| `--data`, `-d` | Yes | Path to training data |
| `--val` | No | Path to validation data |
| `--project`, `-p` | Yes | Project name |
| `--family`, `-f` | No | Family name (default: `default`) |
| `--experiment`, `-e` | Yes | Experiment name |
| `--config`, `-c` | No | Path to YAML config file |

**Example:**

```bash
kladml train single --model gluformer --data train.h5 --project sentinella --experiment v1
```

---

### `kladml train grid`

Run a grid search over hyperparameters.

```bash
kladml train grid [OPTIONS]
```

The configuration file must define lists of values for grid search.

**Example:**

```bash
kladml train grid --model gluformer --config grid.yaml --project sentinella --experiment tuning
```

---

## Evaluation Commands

### `kladml eval run`

Evaluate a trained model on test data.

```bash
kladml eval run [OPTIONS]
```

**Options:**

| Option | Required | Description |
|--------|----------|-------------|
| `--checkpoint` | Yes | Path to model checkpoint (`.pt` file) |
| `--data` | Yes | Path to test data |
| `--model` | No | Model type (default: auto-detect) |
| `--output` | No | Output directory for results |
| `--device` | No | Device: `auto`, `cpu`, `cuda` |

**Example:**

```bash
kladml eval run --checkpoint best_model_jit.pt --data test.pkl --output eval_results/
```

**Output includes:**
- Metrics (MAE, RMSE, MAPE, Coverage)
- Plots (predictions, error distribution, scatter)
- JSON metrics file and markdown report

### `kladml eval info`

Show available evaluators for each model type.

```bash
kladml eval info
```

---

## Data Commands

### `kladml data inspect`

Analyze a `.pkl` dataset file.

```bash
kladml data inspect <path>
```

### `kladml data convert`

Convert a dataset to HDF5 format for lazy loading.

```bash
kladml data convert [OPTIONS]
```

**Options:**

| Option | Required | Description |
|--------|----------|-------------|
| `--input`, `-i` | Yes | Input `.pkl` file path |
| `--output`, `-o` | Yes | Output `.h5` file path |
| `--compression` | No | Compression (gzip, lzf). Default: gzip |

**Example:**

```bash
kladml data convert -i train.pkl -o train.h5
```

---

## Environment Variables


KladML respects these environment variables:

| Variable | Description |
|----------|-------------|
| `KLADML_TRAINING_DEVICE` | Override default device (`cpu`, `cuda`, `mps`) |
| `KLADML_STORAGE_ARTIFACTS_DIR` | Directory for saving artifacts |
| `KLADML_EXPERIMENT` | Default experiment name |

---



---

## Hyperparameter Tuning

### `kladml tune`

Run automated hyperparameter tuning using Optuna.

```bash
kladml tune [OPTIONS]
```

**Options:**

| Option | Required | Description |
|--------|----------|-------------|
| `--config`, `-c` | Yes | Path to YAML config file |
| `--n-trials`, `-n` | No | Number of trials (default: 50) |
| `--timeout` | No | Maximum tuning time in seconds |
| `--pruner` | No | Pruning strategy: `median`, `hyperband` (default: `median`) |
| `--study-name` | No | Name for the Optuna study |
| `--storage` | No | Database URL for distributed tuning |

**Examples:**

```bash
# Basic tuning
kladml tune --config config.yaml --n-trials 50

# With timeout
kladml tune --config config.yaml --n-trials 100 --timeout 3600

# Distributed tuning with shared database
kladml tune --config config.yaml --storage sqlite:///optuna.db --study-name my-study
```

**Output:**
- Best configuration saved to `best_config.yaml`
- Optimization history plot
- Parameter importance plot

---

## Run Comparison

### `kladml compare`

Compare multiple training runs visually.

```bash
kladml compare --runs <run_ids> [OPTIONS]
```

**Options:**

| Option | Required | Description |
|--------|----------|-------------|
| `--runs`, `-r` | Yes | Comma-separated list of run IDs |
| `--metric`, `-m` | No | Metric to compare (default: `val_loss`) |
| `--output`, `-o` | No | Output directory for comparison plots |

**Examples:**

```bash
# Compare two runs
kladml compare --runs run_001,run_002 --metric val_loss

# Compare multiple runs
kladml compare --runs run_001,run_002,run_003 --metric accuracy --output comparisons/
```

**Output:**
- Tabular comparison of Metrics (all logged metrics)
- Tabular comparison of Parameters (hyperparameters, config)
- Side-by-side view for direct analysis

---

## Component Registration

### `kladml register`

Register custom components (architectures, preprocessors, evaluators).

```bash
kladml register <component_type> [OPTIONS]
```

**Component Types:**

| Type | Description |
|------|-------------|
| `architecture` | Custom model architecture |
| `preprocessor` | Custom data preprocessor |
| `evaluator` | Custom evaluator |

**Options:**

| Option | Required | Description |
|--------|----------|-------------|
| `--name`, `-n` | Yes | Name for the component |
| `--module`, `-m` | Yes | Python module path (e.g., `my_pkg.MyClass`) |
| `--description` | No | Description of the component |

**Examples:**

```bash
# Register a custom architecture
kladml register architecture --name MyTransformer --module my_models.MyTransformer

# Register a custom preprocessor
kladml register preprocessor --name MyScaler --module my_transforms.MyScaler

# Register a custom evaluator
kladml register evaluator --name MyEvaluator --module my_eval.MyEvaluator
```

### `kladml list`

List registered components.

```bash
kladml list <component_type>
```

**Examples:**

```bash
kladml list architectures
kladml list preprocessors
kladml list evaluators
kladml list datasets
kladml list runs
```
