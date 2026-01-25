
import typer
from typing import Optional
from rich.console import Console

# Utils
from kladml.cli.commands.train.utils import relaunch_with_accelerate
from kladml.utils.config_io import load_yaml_config
from kladml.utils.loading import resolve_model_class

app = typer.Typer(help="Train models")
console = Console()

@app.command("single")
def train_single(
    model: str = typer.Option(..., "--model", "-m", help="Model name (e.g. 'gluformer') or path to .py file"),
    data: str = typer.Option(..., "--data", "-d", help="Path to training data"),
    val_data: Optional[str] = typer.Option(None, "--val", "-v", help="Path to validation data"),
    project: str = typer.Option(..., "--project", "-p", help="Project name"),
    family: str = typer.Option("default", "--family", "-f", help="Family name (default: 'default')"),
    experiment: str = typer.Option(..., "--experiment", "-e", help="Experiment name"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to YAML config"),
    distributed: bool = typer.Option(False, "--distributed", help="Run in distributed mode using Accelerate"),
    num_processes: int = typer.Option(1, "--num-processes", help="Number of processes (GPUs) for distributed mode"),
) -> None:
    """Run a single training."""
    import sys
    
    # Handle Distributed Launch
    if distributed:
        if sys.platform == "darwin":
            console.print("[bold yellow]⚠️  Distributed training not supported efficiently on macOS (MPS).[/bold yellow]")
            console.print("[green]Auto-falling back to standard single-process training (Accelerated via MPS).[/green]")
            distributed = False
        else:
            try:
                import torch
                if torch.cuda.is_available():
                    available_gpus = torch.cuda.device_count()
                    if num_processes > available_gpus:
                        console.print(f"[bold red]Error: Requested {num_processes} processes but only {available_gpus} GPUs found.[/bold red]")
                        raise typer.Exit(code=1)
                else:
                    import multiprocessing
                    available_cpus = multiprocessing.cpu_count()
                    if num_processes > available_cpus:
                        console.print(f"[bold yellow]Warning: Requested {num_processes} processes but only {available_cpus} CPUs found.[/bold yellow]")
            except ImportError:
                pass 
                
            relaunch_with_accelerate(num_processes=num_processes)
            return 
        
    # Lazy imports
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
        model_class = resolve_model_class(model)
        console.print(f"Loaded model: [green]{model_class.__name__}[/green]")
    except Exception as e:
        console.print(f"[red]Error loading model:[/red] {e}")
        raise typer.Exit(code=1)
    
    train_config = load_yaml_config(config) if config else {}
    train_config["family_name"] = family
    
    try:
        proj = metadata.get_project(project)
        if not proj:
            console.print(f"[yellow]Creating project '{project}'...[/yellow]")
            proj = metadata.create_project(project)
            
        fam = metadata.get_family(family, project)
        if not fam:
            console.print(f"[yellow]Creating family '{family}'...[/yellow]")
            fam = metadata.create_family(family, project, description="Created by training CLI")
            
        tracker.create_experiment(experiment)
        
        if experiment not in fam.experiment_names:
            metadata.add_experiment_to_family(family, project, experiment)
            
    except Exception as e:
        console.print(f"[red]Metadata error:[/red] {e}")
        raise typer.Exit(code=1)
    
    executor = LocalTrainingExecutor(
        model_class=model_class,
        experiment_name=experiment,
        config=train_config,
        tracker=tracker,
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
    from kladml.training.executor import LocalTrainingExecutor
    from kladml.backends.local_tracker import LocalTracker
    from kladml.interfaces.tracker import TrackerInterface
    from kladml.backends import get_metadata_backend
    
    tracker: TrackerInterface = LocalTracker()
    metadata = get_metadata_backend()
    
    console.print(f"[bold]Grid Search Training: {model}[/bold]")
    
    try:
        model_class = resolve_model_class(model)
        console.print(f"Loaded model: [green]{model_class.__name__}[/green]")
    except Exception as e:
        console.print(f"[red]Error loading model:[/red] {e}")
        raise typer.Exit(code=1)
    
    try:
        config = load_yaml_config(grid_config)
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
    
    try:
        proj = metadata.get_project(project)
        if not proj:
            console.print(f"[yellow]Creating project '{project}'...[/yellow]")
            proj = metadata.create_project(project)
            
        fam = metadata.get_family(family, project)
        if not fam:
            console.print(f"[yellow]Creating family '{family}'...[/yellow]")
            fam = metadata.create_family(family, project, description="Created by grid search")
            
        tracker.create_experiment(experiment)
        
        if experiment not in fam.experiment_names:
            metadata.add_experiment_to_family(family, project, experiment)
            
    except Exception as e:
        console.print(f"[red]Metadata error:[/red] {e}")
        raise typer.Exit(code=1)
    
    executor = LocalTrainingExecutor(
        model_class=model_class,
        experiment_name=experiment,
        config=config,
        tracker=tracker,
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
    """
    from pathlib import Path
    
    console.print("[bold blue]🚀 KladML Quick Training[/bold blue]\n")
    
    config_path = Path(config)
    if not config_path.exists():
        console.print(f"[red]Error: Config not found: {config}[/red]")
        raise typer.Exit(1)
    
    train_config = load_yaml_config(str(config_path))
    
    if device != "auto":
        train_config["device"] = device
    
    console.print(f"  Model: [cyan]{model}[/cyan]")
    console.print(f"  Config: [cyan]{config}[/cyan]")
    console.print(f"  Train data: [cyan]{train_data}[/cyan]")
    
    # ... Print config keys ...
    
    try:
        model_class = resolve_model_class(model)
        console.print(f"Loaded: [green]{model_class.__name__}[/green]")
    except Exception as e:
        console.print(f"[red]Error loading model:[/red] {e}")
        raise typer.Exit(1)
    
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
                console.print(f"  {key}: {value}")
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user.[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]Training failed:[/red] {e}")
        raise typer.Exit(1)
