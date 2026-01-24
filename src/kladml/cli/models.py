"""
Model management CLI commands.
"""

import typer
import logging
from pathlib import Path
from rich.console import Console

app = typer.Typer(help="Manage and export models")
console = Console()
logger = logging.getLogger(__name__)

@app.command("export")
def export_model(
    checkpoint_path: Path = typer.Option(..., "--checkpoint", "-c", help="Path to model checkpoint (.pt/.pth)"),
    output_path: Path = typer.Option(..., "--output", "-o", help="Output path for TorchScript model (.pt)"),
    config_path: Path = typer.Option(None, "--config", help="Path to config file (if not in checkpoint)"),
):
    """
    Export a trained model to TorchScript for deployment.
    
    Wraps the model to accept simplified input and embeds scaler stats.
    """
    # Lazy imports for faster CLI startup
    import torch
    from kladml.models.timeseries.transformer.gluformer.model import GluformerModel
    from kladml.training.checkpoint import CheckpointManager
    
    try:
        console.print(f"[bold cyan]Exporting model from:[/bold cyan] {checkpoint_path}")
        
        # 1. Load Checkpoint
        if not checkpoint_path.exists():
            console.print(f"[red]Checkpoint not found:[/red] {checkpoint_path}")
            raise typer.Exit(1)
            
        device = "cpu" # Force CPU for export
        # SECURITY: weights_only=False required to load sklearn scaler from checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # 2. Extract Config & Scaler
        # If config is not in checkpoint, try to infer or use provided
        # GluformerModel typically reloads config from file if not passed?
        # Ideally checkpoint has config. If not, we rely on defaults or external config.
        # Checkpoint dictionary structure: {'model_state_dict': ..., 'scaler': ...}
        # It DOES NOT usually have 'config'.
        
        # For now, we assume default Gluformer config matches training.
        # TODO: Ideally save config in checkpoint or metadata.json.
        # We will try to load metadata.json from parent folder if exists.
        
        project_dir = checkpoint_path.parent.parent # .../checkpoints/<run_id>/ -> .../checkpoints/
        # Wait, structure is run_id/checkpoints/ -> parent is run_id.
        run_dir = checkpoint_path.parent.parent
        # No, structure is .../checkpoints/run_id/ -> parent is run_id?
        # New structure: .../run_id/checkpoints/ -> parent is run_id.
        run_dir = checkpoint_path.parent.parent
        
        # Instantiate Model
        # Load config if provided
        model_config = {
            "device": "cpu",
            "seq_len": 60,
            "pred_len": 12,
            "label_len": 48
            # Defaults
        }
        
        if config_path and config_path.exists():
            console.print(f"[cyan]Loading config from {config_path}[/cyan]")
            import yaml
            with open(config_path) as f:
                loaded_config = yaml.safe_load(f)
                model_config.update(loaded_config)
        
        # We need the architecture.
        # Assuming GluformerModel for now (since export is specialized).
        
        model_wrapper = GluformerModel(config=model_config)
        model_wrapper._build_model()
        model_wrapper._model.load_state_dict(checkpoint["model_state_dict"])
        model_wrapper._is_trained = True
        
        # 3. Export using deployment utility
        from kladml.models.timeseries.transformer.gluformer.deployment import export_to_torchscript
        
        console.print("[cyan]Exporting to TorchScript...[/cyan]")
        export_to_torchscript(
            model=model_wrapper._model,
            output_path=str(output_path),
            scaler=checkpoint.get("scaler"),
            seq_len=model_wrapper.seq_len,
            pred_len=model_wrapper.pred_len,
            label_len=model_wrapper.label_len
        )
        
        console.print(f"[bold green]✓ Export successful![/bold green]")
        console.print(f"Output: {output_path}")

    except Exception as e:
        console.print(f"[red]Export failed:[/red] {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)
