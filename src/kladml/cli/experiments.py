"""
Experiment management CLI commands for KladML.

Uses TrackerInterface for experiment/run management.
"""

import typer
from typing import Optional, List
from rich.console import Console
from rich.table import Table

from kladml.db import Project, init_db
from kladml.db.session import session_scope
from kladml.backends.local_tracker import LocalTracker
from kladml.interfaces.tracker import TrackerInterface

app = typer.Typer(help="Manage KladML experiments")
console = Console()

# Instantiate tracker (DI would be better in a larger app)
tracker: TrackerInterface = LocalTracker()


@app.command("create")
def create_experiment(
    name: str = typer.Option(..., "--name", "-n", help="Experiment name"),
    project: str = typer.Option(..., "--project", "-p", help="Parent project name"),
    description: Optional[str] = typer.Option(None, "--description", "-d", help="Experiment description"),
) -> None:
    """
    Create a new experiment under a project.
    
    Example:
        kladml experiment create -p my-project -n baseline
    """
    init_db()
    
    # Find parent project
    with session_scope() as session:
        parent = session.query(Project).filter_by(name=project).first()
        if not parent:
            console.print(f"[red]Error:[/red] Project '{project}' not found")
            raise typer.Exit(code=1)
        
        # Check if already linked
        if parent.experiment_names and name in parent.experiment_names:
            console.print(f"[yellow]Experiment '{name}' already exists in project '{project}'[/yellow]")
            return
        
        # Create via Tracker interface
        try:
            exp_id = tracker.create_experiment(name)
            console.print(f"Created/Found experiment: {name} (id: {exp_id})")
        except Exception as e:
            console.print(f"[red]Error creating experiment:[/red] {e}")
            raise typer.Exit(code=1)
        
        # Link to project
        parent.add_experiment(name)
    
    console.print(f"[green]✓[/green] Created experiment '{name}' in project '{project}'")


@app.command("list")
def list_experiments(
    project: str = typer.Option(..., "--project", "-p", help="Project name"),
) -> None:
    """
    List all experiments in a project.
    
    Example:
        kladml experiment list -p my-project
    """
    init_db()
    
    with session_scope() as session:
        parent = session.query(Project).filter_by(name=project).first()
        if not parent:
            console.print(f"[red]Error:[/red] Project '{project}' not found")
            raise typer.Exit(code=1)
        
        experiment_names = parent.experiment_names or []
        
        if not experiment_names:
            console.print(f"[yellow]No experiments in project '{project}'[/yellow]")
            return
        
        # Get details via Tracker
        table = Table(title=f"Experiments in '{project}'")
        table.add_column("Name", style="bold")
        table.add_column("ID", style="dim")
        table.add_column("Runs", justify="right")
        table.add_column("Status")
        
        found_count = 0
        for name in experiment_names:
            exp = tracker.get_experiment_by_name(name)
            if exp:
                runs = tracker.search_runs(exp["id"], max_results=0)  # Just need count really, but API might limit
                # Better way: search_runs returns list, just take len
                # To be efficient we might need a count method, but for now lists are fine for local
                run_list = tracker.search_runs(exp["id"], max_results=1000)
                
                table.add_row(
                    exp["name"],
                    exp["id"],
                    str(len(run_list)),
                    exp.get("lifecycle_stage", "active"),
                )
                found_count += 1
            else:
                table.add_row(name, "-", "0", "[red]not found[/red]")
        
        console.print(table)


@app.command("delete")
def delete_experiment(
    name: str = typer.Argument(..., help="Experiment name to delete"),
    project: str = typer.Option(..., "--project", "-p", help="Parent project name"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """
    Unlink an experiment from a project.
    
    Example:
        kladml experiment delete baseline -p my-project
    """
    init_db()
    
    with session_scope() as session:
        parent = session.query(Project).filter_by(name=project).first()
        if not parent:
            console.print(f"[red]Error:[/red] Project '{project}' not found")
            raise typer.Exit(code=1)
        
        if not parent.experiment_names or name not in parent.experiment_names:
            console.print(f"[red]Error:[/red] Experiment '{name}' not found in project '{project}'")
            raise typer.Exit(code=1)
        
        if not force:
            console.print(f"[yellow]Warning:[/yellow] This will unlink experiment '{name}' from project '{project}'")
            confirm = typer.confirm("Are you sure?")
            if not confirm:
                raise typer.Exit(code=0)
        
        parent.remove_experiment(name)
    
    console.print(f"[green]✓[/green] Unlinked experiment '{name}' from project '{project}'")


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
