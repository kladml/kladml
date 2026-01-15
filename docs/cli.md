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

## Environment Variables

KladML respects these environment variables:

| Variable | Description |
|----------|-------------|
| `KLADML_TRAINING_DEVICE` | Override default device (`cpu`, `cuda`, `mps`) |
| `KLADML_STORAGE_ARTIFACTS_DIR` | Directory for saving artifacts |
| `KLADML_EXPERIMENT` | Default experiment name |
