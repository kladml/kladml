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

# NOTE: Heavy imports (db, training, backends) are done inside functions
# for faster CLI startup time

app = typer.Typer(help="Train models")
console = Console()


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
    
    from kladml.models.base import BaseModel
    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type) 
            and issubclass(obj, BaseModel) 
            and obj is not BaseModel
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
        # Import module: kladml.models.{name}
        module_path = f"kladml.models.{model_identifier}"
        try:
            module = importlib.import_module(module_path)
        except ImportError:
             raise ValueError(f"Model '{model_identifier}' not found in kladml.models")

        from kladml.models.base import BaseModel
        
        # Check module's __init__ for a subclass of BaseModel
        for name in dir(module):
            obj = getattr(module, name)
            if (
                isinstance(obj, type) 
                and issubclass(obj, BaseModel) 
                and obj is not BaseModel
            ):
                return obj
                
        # If not found in __init__, try .model submodule
        try:
            model_submodule = importlib.import_module(f"{module_path}.model")
            for name in dir(model_submodule):
                obj = getattr(model_submodule, name)
                if (
                    isinstance(obj, type) 
                    and issubclass(obj, BaseModel) 
                    and obj is not BaseModel
                ):
                    return obj
        except ImportError:
            pass
            
        raise ValueError(f"No BaseModel subclass found in {module_path}")
        
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
    family: str = typer.Option("default", "--family", "-f", help="Family name (default: 'default')"),
    experiment: str = typer.Option(..., "--experiment", "-e", help="Experiment name"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to YAML config"),
) -> None:
    """Run a single training."""
    # Lazy imports for faster CLI startup
    from kladml.training.executor import LocalTrainingExecutor
    from kladml.backends.local_tracker import LocalTracker
    from kladml.interfaces.tracker import TrackerInterface
    from kladml.backends import get_metadata_backend
    
    tracker: TrackerInterface = LocalTracker()
    metadata = get_metadata_backend()
    
    console.print(f"[bold]Training: {model}[/bold]")
    console.print(f"Data: {data}")
    console.print(f"Project: {project} / Family: {family} / Experiment: {experiment}")
    
    try:
        model_class = _resolve_model_class(model)
        console.print(f"Loaded model: [green]{model_class.__name__}[/green]")
    except Exception as e:
        console.print(f"[red]Error loading model:[/red] {e}")
        raise typer.Exit(code=1)
    
    train_config = _load_yaml_config(config) if config else {}
    train_config["family_name"] = family
    
    # Setup Metadata (Project/Family/Experiment links)
    try:
        # Get or Create Project
        proj = metadata.get_project(project)
        if not proj:
            console.print(f"[yellow]Creating project '{project}'...[/yellow]")
            proj = metadata.create_project(project)
            
        # Get or Create Family
        fam = metadata.get_family(family, project)
        if not fam:
            console.print(f"[yellow]Creating family '{family}'...[/yellow]")
            fam = metadata.create_family(family, project, description="Created by training CLI")
            
        # Create/Get experiment via Tracker
        tracker.create_experiment(experiment)
        
        # Link to family (if not already linked)
        if experiment not in fam.experiment_names:
            metadata.add_experiment_to_family(family, project, experiment)
            
    except Exception as e:
        console.print(f"[red]Metadata error:[/red] {e}")
        raise typer.Exit(code=1)
    
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
    family: str = typer.Option("default", "--family", "-f", help="Family name (default: 'default')"),
    experiment: str = typer.Option(..., "--experiment", "-e", help="Experiment name"),
    grid_config: str = typer.Option(..., "--grid", "-g", help="Path to grid search YAML config"),
) -> None:
    """Run grid search training."""
    # Lazy imports for faster CLI startup
    from kladml.training.executor import LocalTrainingExecutor
    from kladml.backends.local_tracker import LocalTracker
    from kladml.interfaces.tracker import TrackerInterface
    from kladml.backends import get_metadata_backend
    
    tracker: TrackerInterface = LocalTracker()
    metadata = get_metadata_backend()
    
    console.print(f"[bold]Grid Search Training: {model}[/bold]")
    
    try:
        model_class = _resolve_model_class(model)
        console.print(f"Loaded model: [green]{model_class.__name__}[/green]")
    except Exception as e:
        console.print(f"[red]Error loading model:[/red] {e}")
        raise typer.Exit(code=1)
    
    try:
        config = _load_yaml_config(grid_config)
        # Inject family name
        config["family_name"] = family
        
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
    
    # Setup Metadata
    try:
        # Get or Create Project
        proj = metadata.get_project(project)
        if not proj:
            console.print(f"[yellow]Creating project '{project}'...[/yellow]")
            proj = metadata.create_project(project)
            
        # Get or Create Family
        fam = metadata.get_family(family, project)
        if not fam:
            console.print(f"[yellow]Creating family '{family}'...[/yellow]")
            fam = metadata.create_family(family, project, description="Created by grid search")
            
        tracker.create_experiment(experiment)
        
        # Link to family
        if experiment not in fam.experiment_names:
            metadata.add_experiment_to_family(family, project, experiment)
            
    except Exception as e:
        console.print(f"[red]Metadata error:[/red] {e}")
        raise typer.Exit(code=1)
    
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


@app.command("quick")
def train_quick(
    model: str = typer.Option("gluformer", "--model", "-m", help="Model name (e.g. 'gluformer')"),
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
    train_data: str = typer.Option(..., "--train", "-t", help="Path to training data (.pkl)"),
    val_data: Optional[str] = typer.Option(None, "--val", "-v", help="Path to validation data (.pkl)"),
    device: str = typer.Option("auto", "--device", "-d", help="Device: auto|cpu|cuda|mps"),
    resume: bool = typer.Option(False, "--resume", "-r", help="Resume from latest checkpoint"),
) -> None:
    """
    Quick training without database or project setup.
    
    Example:
        kladml train quick -c config.yaml -t train.pkl -v val.pkl
        
    Resume interrupted training:
        kladml train quick -c config.yaml -t train.pkl --resume
    """
    import yaml
    
    console.print("[bold blue]🚀 KladML Quick Training[/bold blue]\n")
    
    # Load config
    config_path = Path(config)
    if not config_path.exists():
        console.print(f"[red]Error: Config not found: {config}[/red]")
        raise typer.Exit(1)
    
    with open(config_path) as f:
        train_config = yaml.safe_load(f) or {}
    
    # Override device if specified
    if device != "auto":
        train_config["device"] = device
    
    console.print(f"  Model: [cyan]{model}[/cyan]")
    console.print(f"  Config: [cyan]{config}[/cyan]")
    console.print(f"  Train data: [cyan]{train_data}[/cyan]")
    if val_data:
        console.print(f"  Val data: [cyan]{val_data}[/cyan]")
    console.print()
    
    # Print key config values
    console.print("[bold]Config:[/bold]")
    for key in ['project_name', 'experiment_name', 'epochs', 'loss_mode', 'batch_size', 'learning_rate']:
        if key in train_config:
            console.print(f"  {key}: {train_config[key]}")
    console.print()
    
    # Resolve model class
    try:
        model_class = _resolve_model_class(model)
        console.print(f"Loaded: [green]{model_class.__name__}[/green]")
    except Exception as e:
        console.print(f"[red]Error loading model:[/red] {e}")
        raise typer.Exit(1)
    
    # Create model instance and train
    if resume:
        console.print("\n[bold yellow]⏯ Resuming training from checkpoint...[/bold yellow]\n")
    else:
        console.print("\n[bold]Starting training...[/bold]\n")
    console.print("-" * 60)
    
    try:
        model_instance = model_class(config=train_config)
        metrics = model_instance.train(X_train=train_data, X_val=val_data, resume=resume)
        
        console.print("-" * 60)
        console.print("\n[bold green]✅ Training complete![/bold green]")
        
        if metrics:
            console.print("\n[bold]Final Metrics:[/bold]")
            for key, value in metrics.items():
                if isinstance(value, float):
                    console.print(f"  {key}: {value:.4f}")
                else:
                    console.print(f"  {key}: {value}")
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user.[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]Training failed:[/red] {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
