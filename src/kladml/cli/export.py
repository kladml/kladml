"""
Export CLI command.
Uses the extensible ExporterRegistry to support multiple formats.
"""

import json
import typer
from pathlib import Path
from rich.console import Console
from typing import Optional

from kladml.exporters import ExporterRegistry

app = typer.Typer(help="Export models to deployment formats")
console = Console()


def _load_metadata(checkpoint_path: Path) -> dict:
    """Load metadata.json from checkpoint directory if available."""
    metadata_path = checkpoint_path.parent / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _get_model_config(checkpoint_data: dict, metadata: dict, config_path: Optional[Path]) -> dict:
    """
    Merge config from multiple sources (priority: CLI config > checkpoint config > metadata).

    Returns:
        Merged configuration dictionary
    """
    cfg = {}

    # Base config from checkpoint (saved during training)
    if checkpoint_data.get("config"):
        cfg.update(checkpoint_data["config"])

    # Override with metadata
    if metadata:
        for key in ["seq_len", "pred_len", "label_len", "enc_in", "dec_in", "d_model"]:
            if key in metadata:
                cfg[key] = metadata[key]

    # CLI config takes highest priority
    if config_path and config_path.exists():
        import yaml
        with open(config_path) as f:
            cli_cfg = yaml.safe_load(f) or {}
            cfg.update(cli_cfg)

    return cfg


def _instantiate_model(cfg: dict, architecture: Optional[str] = None):
    """
    Instantiate model based on architecture hint or config.

    Args:
        cfg: Model configuration
        architecture: Optional architecture name (e.g., "gluformer", "canbus")

    Returns:
        Tuple of (model_wrapper, inner_model)
    """
    # Default architecture if not specified
    arch = architecture or cfg.get("architecture", "gluformer").lower()

    if arch == "gluformer":
        from kladml.models.timeseries.transformer.gluformer.model import GluformerModel
        wrapper = GluformerModel(config=cfg)
        wrapper._build_model()
        return wrapper, wrapper._model
    elif arch == "canbus":
        from kladml.models.timeseries.transformer.canbus.model import CanbusModel
        wrapper = CanbusModel(config=cfg)
        wrapper._build_model()
        return wrapper, wrapper._model
    else:
        raise ValueError(f"Unknown architecture: {arch}. Supported: gluformer, canbus")


@app.callback(invoke_without_command=True)
def export(
    ctx: typer.Context,
    checkpoint: Path = typer.Option(..., "--checkpoint", "-c", help="Path to model checkpoint (.pt/.pth)"),
    format: str = typer.Option("torchscript", "--format", "-f", help="Export format (onnx, torchscript)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path (default: derived from checkpoint)"),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config file (if needed)"),
    architecture: Optional[str] = typer.Option(None, "--architecture", "-a", help="Model architecture (gluformer, canbus)"),
    embed_scaler: bool = typer.Option(False, "--embed-scaler", help="Embed scaler from checkpoint into exported model"),
):
    """
    Export a trained model to a deployment format.

    The command automatically detects model configuration from:
    1. metadata.json in checkpoint directory
    2. config saved inside the checkpoint
    3. Optional --config YAML file override

    Use --embed-scaler to include the data normalization step in the exported model.
    """
    if ctx.invoked_subcommand is not None:
        return

    try:
        # Validate format
        try:
            exporter_cls = ExporterRegistry.get(format)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)

        exporter = exporter_cls()
        console.print(f"[bold cyan]Exporting to {format.upper()}...[/bold cyan]")

        # Determine output path
        if output is None:
            ext = "onnx" if format == "onnx" else "pt"
            output = checkpoint.with_suffix(f".{ext}")

        import torch

        if not checkpoint.exists():
            console.print(f"[red]Checkpoint not found: {checkpoint}[/red]")
            raise typer.Exit(1)

        # Load checkpoint
        # SECURITY: weights_only=False required for some scalers, use caution
        ckpt_data = torch.load(checkpoint, map_location="cpu", weights_only=False)

        # Load metadata from checkpoint directory
        metadata = _load_metadata(checkpoint)

        # Get merged config
        cfg = _get_model_config(ckpt_data, metadata, config)

        # Set defaults for required fields
        cfg.setdefault("seq_len", 60)
        cfg.setdefault("pred_len", 12)
        cfg.setdefault("label_len", 48)
        cfg.setdefault("enc_in", 7)

        # Extract scaler if present and requested
        scaler = None
        if embed_scaler and isinstance(ckpt_data, dict) and "scaler" in ckpt_data:
            scaler = ckpt_data["scaler"]
            console.print(f"[dim]Found scaler: {type(scaler).__name__}[/dim]")

        # Check if checkpoint is already a model instance
        if isinstance(ckpt_data, torch.nn.Module):
            model = ckpt_data
            console.print("[dim]Loaded model directly from checkpoint[/dim]")
        else:
            # Instantiate model with detected configuration
            try:
                wrapper, model = _instantiate_model(cfg, architecture)
                console.print(f"[dim]Instantiated {wrapper.__class__.__name__}[/dim]")
            except Exception as e:
                console.print(f"[red]Failed to instantiate model: {e}[/red]")
                console.print("[yellow]Tip: Use --architecture to specify model type[/yellow]")
                raise typer.Exit(1)

            # Load weights
            if "model_state_dict" in ckpt_data:
                model.load_state_dict(ckpt_data["model_state_dict"])
                console.print("[dim]Loaded model_state_dict from checkpoint[/dim]")
            elif isinstance(ckpt_data, dict):
                # Try loading as raw state dict
                try:
                    model.load_state_dict(ckpt_data)
                    console.print("[dim]Loaded state dict directly from checkpoint[/dim]")
                except Exception as e:
                    console.print(f"[red]Could not load weights: {e}[/red]")
                    raise typer.Exit(1)

        # Prepare input sample for tracing
        seq_len = cfg.get("seq_len", 60)
        enc_in = cfg.get("enc_in", 7)
        dummy_input = torch.randn(1, seq_len, enc_in)

        # Execute Export
        result_path = exporter.export(model, str(output), input_sample=dummy_input, scaler=scaler)

        if exporter.validate(result_path, input_sample=dummy_input):
            console.print(f"[bold green]Exported successfully to {result_path}[/bold green]")
            if scaler is not None:
                console.print("[dim]Scaler embedded in exported model[/dim]")
        else:
            console.print(f"[bold red]Export verification failed[/bold red]")

    except Exception as e:
        console.print(f"[red]Export Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)
