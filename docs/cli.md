# CLI Reference

KladML includes a command-line interface for common tasks.

## Global Commands

### `kladml version`

Show the installed version.

```bash
kladml version
# KladML version 0.1.0
```

### `kladml --help`

Show all available commands.

```bash
kladml --help
```

---

## Project Commands

### `kladml init`

Initialize a new KladML project.

```bash
kladml init <project-name> [OPTIONS]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `project-name` | Name of the project directory to create |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--template`, `-t` | `default` | Project template to use |

**Example:**

```bash
kladml init my-forecaster
cd my-forecaster
```

---

## Run Commands

### `kladml run native`

Run a training script using your local Python environment.

```bash
kladml run native <script> [OPTIONS]
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
