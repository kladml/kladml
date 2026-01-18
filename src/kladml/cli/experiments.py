"""
Experiment management CLI commands for KladML.

Uses TrackerInterface for experiment/run management.
"""

import typer
from typing import Optional, List
from rich.console import Console
from rich.table import Table

from kladml.backends.local_tracker import LocalTracker
from kladml.interfaces.tracker import TrackerInterface
from kladml.backends import get_metadata_backend

app = typer.Typer(help="Manage KladML experiments")
console = Console()
metadata = get_metadata_backend()

# Instantiate tracker (DI would be better in a larger app)
tracker: TrackerInterface = LocalTracker()


@app.command("create")
def create_experiment(
    name: str = typer.Option(..., "--name", "-n", help="Experiment name"),
    project: str = typer.Option(..., "--project", "-p", help="Parent project name"),
    family: str = typer.Option("default", "--family", "-f", help="Family name (default: 'default')"),
    description: Optional[str] = typer.Option(None, "--description", "-d", help="Experiment description"),
) -> None:
    """
    Create a new experiment under a project and family.
    
    Example:
        kladml experiment create -p sentinella -f glucose_forecasting -n gluformer_v4
    """
    try:
        # Check if project exists
        proj = metadata.get_project(project)
        if not proj:
            console.print(f"[red]Error:[/red] Project '{project}' not found")
            raise typer.Exit(code=1)
        
        # Check/Create Family
        fam = metadata.get_family(family, project)
        if not fam:
            console.print(f"[yellow]Family '{family}' does not exist. Creating it...[/yellow]")
            fam = metadata.create_family(family, project, description="Default family")
        
        # Check if experiment already linked
        if fam.experiment_names and name in fam.experiment_names:
            console.print(f"[yellow]Experiment '{name}' already exists in family '{family}'[/yellow]")
            return

        # Create via Tracker interface (MLflow)
        try:
            exp_id = tracker.create_experiment(name)
            console.print(f"Created/Found MLflow experiment: {name} (id: {exp_id})")
        except Exception as e:
            console.print(f"[red]Error creating experiment:[/red] {e}")
            raise typer.Exit(code=1)
        
        # Link to family
        metadata.add_experiment_to_family(family, project, name)
        
        console.print(f"[green]✓[/green] Created experiment '{name}' in family '{family}' (project '{project}')")
        
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


@app.command("list")
def list_experiments(
    project: str = typer.Option(..., "--project", "-p", help="Project name"),
    family: Optional[str] = typer.Option(None, "--family", "-f", help="Filter by family"),
) -> None:
    """
    List all experiments in a project (grouped by family).
    
    Example:
        kladml experiment list -p sentinella
    """
    try:
        if family:
            fam_list = [metadata.get_family(family, project)]
            if not fam_list[0]:
                console.print(f"[red]Family '{family}' not found in project '{project}'[/red]")
                return
        else:
            fam_list = metadata.list_families(project)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)
        
    if not fam_list:
        console.print(f"[yellow]No families/experiments found in project '{project}'[/yellow]")
        return
    
    table = Table(title=f"Experiments in '{project}'")
    table.add_column("Family", style="cyan")
    table.add_column("Experiment", style="bold")
    table.add_column("ID", style="dim")
    table.add_column("Runs", justify="right")
    table.add_column("Status")
    
    for fam in fam_list:
        experiment_names = fam.experiment_names or []
        
        if not experiment_names:
            table.add_row(fam.name, "[dim]-[/dim]", "-", "-", "-")
            continue
            
        for name in experiment_names:
            exp = tracker.get_experiment_by_name(name)
            if exp:
                runs = tracker.search_runs(exp["id"], max_results=1000)
                table.add_row(
                    fam.name,
                    exp["name"],
                    exp["id"],
                    str(len(runs)),
                    exp.get("lifecycle_stage", "active"),
                )
            else:
                table.add_row(fam.name, name, "-", "0", "[red]not found[/red]")
    
    console.print(table)


@app.command("delete")
def delete_experiment(
    name: str = typer.Argument(..., help="Experiment name to unlink"),
    project: str = typer.Option(..., "--project", "-p", help="Parent project name"),
    family: str = typer.Option("default", "--family", "-f", help="Family name"),
    force: bool = typer.Option(False, "--force", "-y", help="Skip confirmation"),
) -> None:
    """
    Unlink an experiment from a family.
    """
    if not force:
        confirm = typer.confirm(f"Are you sure you want to remove experiment '{name}' from family '{family}'?")
        if not confirm:
            raise typer.Abort()

    try:
        # Check existence
        fam = metadata.get_family(family, project)
        if not fam:
            console.print(f"[red]Error:[/red] Family '{family}' not found")
            raise typer.Exit(code=1)
            
        if not fam.experiment_names or name not in fam.experiment_names:
            console.print(f"[yellow]Experiment '{name}' not found in family '{family}'[/yellow]")
            return

        # Remove from Family (Metadata)
        metadata.remove_experiment_from_family(family, project, name)
        
        # Note: We do NOT delete the MLflow experiment itself, as it might contain data.
        # We just unlink it from the hierarchy.
        
        console.print(f"[green]✓[/green] Removed experiment '{name}' from family '{family}'")
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


@app.command("runs")
def list_runs(
    experiment: str = typer.Argument(..., help="Experiment name"),
    max_results: int = typer.Option(20, "--max", "-m", help="Maximum results"),
) -> None:
    """
    List runs in an experiment.
    
    Example:
        kladml experiment runs baseline
    """
    exp = tracker.get_experiment_by_name(experiment)
    if not exp:
        console.print(f"[red]Error:[/red] Experiment '{experiment}' not found")
        return
        
    runs = tracker.search_runs(exp["id"], max_results=max_results)
    
    if not runs:
        console.print(f"[yellow]No runs found in experiment '{experiment}'[/yellow]")
        return
    
    table = Table(title=f"Runs in '{experiment}'")
    table.add_column("Run ID", style="dim")
    table.add_column("Name", style="bold")
    table.add_column("Status")
    table.add_column("Metrics")
    
    for run in runs:
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in list(run["metrics"].items())[:3])
        status_style = "green" if run["status"] == "FINISHED" else "yellow"
        
        table.add_row(
            run["run_id"][:8],
            run.get("run_name", "-"),
            f"[{status_style}]{run['status']}[/{status_style}]",
            metrics_str or "-",
        )
    
    console.print(table)


@app.command("compare")
def compare_experiments(
    exp1: str = typer.Argument(..., help="First experiment name"),
    exp2: str = typer.Argument(..., help="Second experiment name"),
    metric: str = typer.Option("loss", "--metric", "-m", help="Metric to compare"),
) -> None:
    """
    Compare two experiments by their best runs.
    """
    def get_best(exp_name: str, metric_name: str):
        exp = tracker.get_experiment_by_name(exp_name)
        if not exp:
            return None, None
            
        runs = tracker.search_runs(exp["id"], max_results=100)
        best_run = None
        best_value = None
        
        for run in runs:
            # Check float metrics
            value = run["metrics"].get(metric_name)
            if value is not None:
                if best_value is None or value < best_value:
                    best_value = value
                    best_run = run
        return best_run, best_value
    
    best1, val1 = get_best(exp1, metric)
    best2, val2 = get_best(exp2, metric)
    
    console.print(f"\n[bold]Comparison: {exp1} vs {exp2}[/bold]")
    console.print(f"Metric: {metric}\n")
    
    table = Table()
    table.add_column("Experiment", style="bold")
    table.add_column("Best Run")
    table.add_column(f"Best {metric}", justify="right")
    
    table.add_row(
        exp1,
        best1["run_name"] if best1 else "-",
        f"{val1:.4f}" if val1 is not None else "-",
    )
    table.add_row(
        exp2,
        best2["run_name"] if best2 else "-",
        f"{val2:.4f}" if val2 is not None else "-",
    )
    
    console.print(table)

if __name__ == "__main__":
    app()
