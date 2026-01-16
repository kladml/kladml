"""
Train command for KladML CLI.

Uses TrackerInterface for MLflow interaction.
"""

import typer
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console

from kladml.db import Project, init_db
from kladml.db.session import session_scope
from kladml.training.executor import LocalTrainingExecutor
from kladml.backends.local_tracker import LocalTracker
from kladml.interfaces.tracker import TrackerInterface

app = typer.Typer(help="Train models")
console = Console()

# Instantiate tracker
tracker: TrackerInterface = LocalTracker()


def _load_model_class_from_path(model_path: str):
    """Dynamically load a model class."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    spec = importlib.util.spec_from_file_location("user_model", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {model_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules["user_model"] = module
    spec.loader.exec_module(module)
    
    from kladml.base import BaseArchitecture
    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type) 
            and issubclass(obj, BaseArchitecture) 
            and obj is not BaseArchitecture
        ):
            return obj
    
    raise ValueError(f"No model class found in {model_path}.")


def _resolve_model_class(model_identifier: str):
    """
    Resolve model class from identifier (name or path).
    
    Args:
        model_identifier: Model name (e.g. "gluformer") or path to .py file
        
    Returns:
        Model class
    """
    # 1. Try loading as file path
    if model_identifier.endswith(".py") or Path(model_identifier).exists():
        return _load_model_class_from_path(model_identifier)
        
    # 2. Try loading as architecture name
    try:
        # Import module: kladml.architectures.{name}
        module_path = f"kladml.architectures.{model_identifier}"
        try:
            module = importlib.import_module(module_path)
        except ImportError:
             raise ValueError(f"Architecture '{model_identifier}' not found in kladml.architectures")

        from kladml.base import BaseArchitecture
        
        # Check module's __init__ for a subclass of BaseArchitecture
        for name in dir(module):
            obj = getattr(module, name)
            if (
                isinstance(obj, type) 
                and issubclass(obj, BaseArchitecture) 
                and obj is not BaseArchitecture
            ):
                return obj
                
        # If not found in __init__, try .model submodule
        try:
            model_submodule = importlib.import_module(f"{module_path}.model")
            for name in dir(model_submodule):
                obj = getattr(model_submodule, name)
                if (
                    isinstance(obj, type) 
                    and issubclass(obj, BaseArchitecture) 
                    and obj is not BaseArchitecture
                ):
                    return obj
        except ImportError:
            pass
            
        raise ValueError(f"No BaseArchitecture subclass found in {module_path}")
        
    except Exception as e:
        raise ValueError(f"Could not load model '{model_identifier}': {e}")


def _load_yaml_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


@app.command("single")
def train_single(
    model: str = typer.Option(..., "--model", "-m", help="Model name (e.g. 'gluformer') or path to .py file"),
    data: str = typer.Option(..., "--data", "-d", help="Path to training data"),
    val_data: Optional[str] = typer.Option(None, "--val", "-v", help="Path to validation data"),
    project: str = typer.Option(..., "--project", "-p", help="Project name"),
    experiment: str = typer.Option(..., "--experiment", "-e", help="Experiment name"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to YAML config"),
) -> None:
    """Run a single training."""
    init_db()
    
    console.print(f"[bold]Training: {model}[/bold]")
    console.print(f"Data: {data}")
    console.print(f"Project: {project} / Experiment: {experiment}")
    
    try:
        model_class = _resolve_model_class(model)
        console.print(f"Loaded model: [green]{model_class.__name__}[/green]")
    except Exception as e:
        console.print(f"[red]Error loading model:[/red] {e}")
        raise typer.Exit(code=1)
    
    train_config = _load_yaml_config(config) if config else {}
    
    with session_scope() as session:
        proj = session.query(Project).filter_by(name=project).first()
        if not proj:
            console.print(f"[yellow]Creating project '{project}'...[/yellow]")
            proj = Project(name=project)
            session.add(proj)
            session.flush()
        
        # Create/Get experiment via Tracker
        tracker.create_experiment(experiment)
        
        # Link to project
        proj.add_experiment(experiment)
    
    # Execute training (inject tracker)
    executor = LocalTrainingExecutor(
        model_class=model_class,
        experiment_name=experiment,
        config=train_config,
        tracker=tracker,  # Pass tracker interface
    )
    
    console.print("\n[bold]Starting training...[/bold]\n")
    
    run_id, metrics = executor.execute_single(data_path=data, val_path=val_data)
    
    if run_id:
        console.print(f"\n[green]✓ Training complete![/green]")
        console.print(f"Run ID: {run_id}")
        if metrics:
            console.print(f"Metrics: {metrics}")
    else:
        console.print(f"\n[red]✗ Training failed[/red]")
        raise typer.Exit(code=1)


@app.command("grid")
def train_grid_search(
    model: str = typer.Option(..., "--model", "-m", help="Model name or path to .py file"),
    data: str = typer.Option(..., "--data", "-d", help="Path to training data"),
    project: str = typer.Option(..., "--project", "-p", help="Project name"),
    experiment: str = typer.Option(..., "--experiment", "-e", help="Experiment name"),
    grid_config: str = typer.Option(..., "--grid", "-g", help="Path to grid search YAML config"),
) -> None:
    """Run grid search training."""
    init_db()
    
    console.print(f"[bold]Grid Search Training: {model}[/bold]")
    
    try:
        model_class = _resolve_model_class(model)
        console.print(f"Loaded model: [green]{model_class.__name__}[/green]")
    except Exception as e:
        console.print(f"[red]Error loading model:[/red] {e}")
        raise typer.Exit(code=1)
    
    try:
        config = _load_yaml_config(grid_config)
        search_space = config.get("search_space", {})
        if not search_space:
            console.print("[red]Error:[/red] No 'search_space' found in grid config")
            raise typer.Exit(code=1)
            
        n_combos = 1
        for values in search_space.values():
            n_combos *= len(values)
        console.print(f"Search space: {len(search_space)} params, {n_combos} combinations")
    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        raise typer.Exit(code=1)
    
    with session_scope() as session:
        proj = session.query(Project).filter_by(name=project).first()
        if not proj:
            console.print(f"[yellow]Creating project '{project}'...[/yellow]")
            proj = Project(name=project)
            session.add(proj)
            session.flush()
        
        tracker.create_experiment(experiment)
        proj.add_experiment(experiment)
    
    # Execute grid search (inject tracker)
    executor = LocalTrainingExecutor(
        model_class=model_class,
        experiment_name=experiment,
        config=config,
        tracker=tracker,  # Pass tracker interface
    )
    
    console.print(f"\n[bold]Starting grid search ({n_combos} runs)...[/bold]\n")
    
    run_ids = executor.execute_grid_search(
        data_path=data,
        search_space=search_space,
    )
    
    console.print(f"\n[green]✓ Grid search complete![/green]")
    console.print(f"Successful runs: {len(run_ids)}/{n_combos}")
    
    if executor.best_run_id:
        console.print(f"\n[bold]Best run:[/bold] {executor.best_run_id}")
        if executor.best_metrics:
            console.print(f"Best metrics: {executor.best_metrics}")


if __name__ == "__main__":
    app()
