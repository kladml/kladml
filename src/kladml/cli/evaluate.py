"""
Evaluation CLI for KladML.

Commands for running model evaluation.
"""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    help="Model evaluation commands",
    no_args_is_help=True
)

console = Console()


@app.command("run")
def run_evaluation(
    checkpoint: Path = typer.Option(
        ..., 
        "--checkpoint", "-c",
        help="Path to model checkpoint (.pt or .pth)"
    ),
    data: Path = typer.Option(
        ..., 
        "--data", "-d",
        help="Path to evaluation dataset (.h5 or .pkl)"
    ),
    model: str = typer.Option(
        "gluformer", 
        "--model", "-m",
        help="Model type (gluformer)"
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="Device for inference (cpu, cuda, mps)"
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output directory (default: checkpoint parent)"
    ),
):
    """
    Run model evaluation and generate report.
    
    Evaluates the model on the provided dataset and generates:
    - evaluation_report.md: Comprehensive Markdown report
    - evaluation.log: Execution log
    - evaluation_metrics.json: Metrics in JSON format
    - plots/: Directory with visualization plots
    
    Example:
        kladml eval run -c best_model_jit.pt -d test.h5 -m gluformer
    """
    # Validate inputs
    if not checkpoint.exists():
        console.print(f"[red]Error: Checkpoint not found: {checkpoint}[/red]")
        raise typer.Exit(1)
    
    if not data.exists():
        console.print(f"[red]Error: Data not found: {data}[/red]")
        raise typer.Exit(1)
    
    # Determine output directory
    run_dir = output_dir or checkpoint.parent
    
    console.print(f"\n[bold blue]KladML Evaluation[/bold blue]")
    console.print(f"  Model: {model}")
    console.print(f"  Checkpoint: {checkpoint}")
    console.print(f"  Data: {data}")
    console.print(f"  Output: {run_dir}")
    console.print(f"  Device: {device}")
    console.print()
    
    # Select evaluator based on model type
    if model.lower() == "gluformer":
        from kladml.models.timeseries.transformer.gluformer.evaluator import GluformerEvaluator
        evaluator = GluformerEvaluator(
            run_dir=run_dir,
            model_path=checkpoint,
            data_path=data,
            device=device
        )
    else:
        console.print(f"[red]Error: Unknown model type: {model}[/red]")
        console.print("Supported models: gluformer")
        raise typer.Exit(1)
    
    # Run evaluation
    console.print("[bold]Running evaluation...[/bold]\n")
    
    try:
        metrics = evaluator.run()
    except Exception as e:
        console.print(f"[red]Evaluation failed: {e}[/red]")
        raise typer.Exit(1)
    
    # Display results
    console.print("\n[bold green]✓ Evaluation Complete![/bold green]\n")
    
    # Create metrics table
    table = Table(title="Metrics Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    for name, value in metrics.items():
        table.add_row(name, f"{value:.4f}")
    
    console.print(table)
    
    # Show output paths
    console.print(f"\n[bold]Output Files:[/bold]")
    console.print(f"  📄 Report: {run_dir / 'evaluation_report.md'}")
    console.print(f"  📊 Metrics: {run_dir / 'evaluation_metrics.json'}")
    console.print(f"  📝 Log: {run_dir / 'evaluation.log'}")
    console.print(f"  📈 Plots: {run_dir / 'plots/'}")


@app.command("info")
def info():
    """
    Show information about available evaluators.
    """
    console.print("\n[bold]Available Evaluators[/bold]\n")
    
    table = Table()
    table.add_column("Model", style="cyan")
    table.add_column("Evaluator", style="green")
    table.add_column("Description")
    
    table.add_row(
        "gluformer", 
        "GluformerEvaluator",
        "Probabilistic glucose forecasting with uncertainty metrics"
    )
    
    console.print(table)
    console.print()
