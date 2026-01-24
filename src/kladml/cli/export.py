"""
Export CLI command.
Uses the extensible ExporterRegistry to support multiple formats.
"""

import typer
from pathlib import Path
from rich.console import Console
from typing import Optional

from kladml.exporters import ExporterRegistry

app = typer.Typer(help="Export models to deployment formats")
console = Console()



@app.callback(invoke_without_command=True)
def export(
    ctx: typer.Context,
    checkpoint: Path = typer.Option(..., "--checkpoint", "-c", help="Path to model checkpoint (.pt/.pth)"),
    format: str = typer.Option("torchscript", "--format", "-f", help="Export format (onnx, torchscript)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path (default: derived from checkpoint)"),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config file (if needed)"),
):
    """
    Export a trained model to a deployment format.
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
            
        # Load Model (Generic Logic)
        # TODO: This needs to be robust. Ideally we use ExperimentRunner or Model.load()
        # For now reusing the logic from legacy models.py but simplified
        import torch
        
        if not checkpoint.exists():
             console.print(f"[red]Checkpoint not found: {checkpoint}[/red]")
             raise typer.Exit(1)
             
        # Load checkpoint
        # SECURITY: weights_only=False required for some scalers, use caution
        ckpt_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        
        # We need the model definitions
        # Assuming we can instantiate the model class if we know it via config or meta
        # For this prototype, we will assume standard Gluformer/Transformer if not specified
        # This part is the tricky bit: restoring the python object.
        
        # AUTO-DETECTION Hack (Temporary until ModelRegistry is full)
        # If config is present, use it.
        
        model_wrapper = None
        
        # TRY: Load using our TimeSeriesModel base if we can guess the class
        # (For now, we just pass the raw nn.Module if saved in checkpoint['model_state_dict']?? No usually it's dict)
        
        # Temporary: Assume we export TimeSeriesModel from the checkpoint
        # Real solution: Checkpoint should contain 'architecture_name'.
        
        console.print("[yellow]Warning: Model instantiation validation is skipped in this generic export (Experimental).[/yellow]")
        
        # 1. Instantiate dummy model (User needs to provide code if it's custom)
        # For now, let's try to export just the state dict if we had the class?
        # WAIT. We can't export state_dict to ONNX/JIT without the class structure.
        
        # Fallback: Check if checkpoint is actually a full model (not recommended but possible)
        if isinstance(ckpt_data, torch.nn.Module):
             model = ckpt_data
        else:
             # It's a dict. We NEED the class.
             # Let's try to import the standard ones
             from kladml.models.timeseries.transformer.gluformer.model import GluformerModel
             
             # Try to build default config
             cfg = {
                 "seq_len": 60, "pred_len": 12, "label_len": 48 # Defaults
             }
             if config:
                 import yaml
                 with open(config) as f:
                     cfg.update(yaml.safe_load(f))
             
             wrapper = GluformerModel(config=cfg)
             wrapper._build_model() # Creates wrapper._model
             if "model_state_dict" in ckpt_data:
                 wrapper._model.load_state_dict(ckpt_data["model_state_dict"])
             else:
                 console.print("[red]Could not find 'model_state_dict' in checkpoint.[/red]")
                 # It might be the raw state dict
                 try:
                    wrapper._model.load_state_dict(ckpt_data)
                 except:
                    pass
                    
             model = wrapper._model
             
             # Prepare input sample for tracing
             # shape: (1, seq_len, n_features)
             # n_features depends on config
             d_model = cfg.get("d_model", 512) 
             # Wait, transformer input features are usually enc_in
             enc_in = cfg.get("enc_in", 7)
             
             dummy_input = torch.randn(1, cfg["seq_len"], enc_in)
             
             # Execute Export
             result_path = exporter.export(model, str(output), input_sample=dummy_input)
             
             if exporter.validate(result_path, input_sample=dummy_input):
                 console.print(f"[bold green]✓ Exported successfully to {result_path}[/bold green]")
             else:
                 console.print(f"[bold red]✗ Export verified failed[/bold red]")

    except Exception as e:
        console.print(f"[red]Export Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)
