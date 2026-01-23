
import typer
from typing import List, Optional
from rich.console import Console
from rich.table import Table
from rich import box
from kladml.backends import get_tracker_backend

# app = typer.Typer(help="Compare runs") - Managed by main
console = Console()

def compare_runs(
    runs: str = typer.Option(..., "--runs", "-r", help="Comma-separated list of Run IDs"),
    metrics: Optional[str] = typer.Option(None, "--metrics", "-m", help="Filter metrics (comma-separated)"),
) -> None:
    """
    Compare multiple runs side-by-side.
    
    Displays metrics and parameters for the specified runs.
    
    Example:
        kladml compare --runs run_id_1,run_id_2
    """
    tracker = get_tracker_backend()
    run_ids = [r.strip() for r in runs.split(",") if r.strip()]
    
    if len(run_ids) < 2:
        console.print("[yellow]Warning: Comparing less than 2 runs.[/yellow]")
    
    data = []
    headers = ["Metric/Param"]
    
    # Fetch data
    fetched_runs = []
    for rid in run_ids:
        run_data = tracker.get_run(rid)
        if not run_data:
            console.print(f"[red]Error: Run '{rid}' not found.[/red]")
            raise typer.Exit(code=1)
        headers.append(f"{run_data.get('run_name', rid)}\n[dim]{rid[:8]}[/dim]")
        fetched_runs.append(run_data)
        
    # Collect all keys
    all_metrics = set()
    all_params = set()
    
    for r in fetched_runs:
        all_metrics.update(r.get("metrics", {}).keys())
        all_params.update(r.get("params", {}).keys())
        
    # Filter metrics if requested
    if metrics:
        wanted = [m.strip() for m in metrics.split(",")]
        all_metrics = {m for m in all_metrics if m in wanted}
    
    # Build Table
    table = Table(title="Run Comparison", box=box.ROUNDED)
    for h in headers:
        table.add_column(h)
        
    # Section: Metrics
    table.add_row("[bold cyan]Metrics[/bold cyan]", *["" for _ in run_ids])
    
    for m in sorted(all_metrics):
        row = [m]
        for r in fetched_runs:
            val = r.get("metrics", {}).get(m, "-")
            if isinstance(val, float):
                val = f"{val:.4f}"
            row.append(str(val))
        table.add_row(*row)
        
    # Section: Params
    table.add_row("[bold cyan]Parameters[/bold cyan]", *["" for _ in run_ids], end_section=True)
    
    for p in sorted(all_params):
        row = [p]
        for r in fetched_runs:
            val = r.get("params", {}).get(p, "-")
            row.append(str(val))
        table.add_row(*row)
        
    console.print(table)


