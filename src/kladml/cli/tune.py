
import typer
import yaml
from pathlib import Path
from rich.console import Console
from typing import Optional, Any

# KladML Imports
from kladml.tuning.tuner import KladMLTuner, TunerConfig
from kladml.tuning.search_space import SearchSpace, FloatDistribution, IntDistribution, CategoricalDistribution
from kladml.training.executor import LocalTrainingExecutor
from kladml.backends.local_tracker import LocalTracker
from kladml.backends import get_metadata_backend

console = Console()
app = typer.Typer(help="Hyperparameter Tuning")

def _parse_search_space(config: dict) -> SearchSpace:
    """Extract search space from config dict."""
    space_def = config.get("tuning", {}).get("search_space", {})
    if not space_def:
        raise ValueError("No 'tuning.search_space' found in config.")
        
    params = {}
    for name, spec in space_def.items():
        type_ = spec.get("type")
        if type_ == "float":
            params[name] = FloatDistribution(
                low=spec["low"], 
                high=spec["high"], 
                log=spec.get("log", False),
                step=spec.get("step")
            )
        elif type_ == "int":
            params[name] = IntDistribution(
                low=spec["low"], 
                high=spec["high"], 
                log=spec.get("log", False),
                step=spec.get("step", 1)
            )
        elif type_ == "categorical":
            params[name] = CategoricalDistribution(
                choices=spec["choices"]
            )
        else:
            raise ValueError(f"Unknown parameter type: {type_} for {name}")
            
    return SearchSpace(parameters=params)

def _set_nested(d: dict, key: str, value: Any):
    """Set value in nested dict using dot notation (e.g. 'model.layers')."""
    parts = key.split(".")
    current = d
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]
        if not isinstance(current, dict):
             # Conflict
             pass
    current[parts[-1]] = value

from kladml.utils.loading import resolve_model_class as _resolve_model_class

@app.command("run")
def run_tuning(
    config_path: str = typer.Option(..., "--config", "-c", help="Path to YAML config with 'tuning' section"),
    model: str = typer.Option(..., "--model", "-m", help="Model name (e.g. 'gluformer')"),
    data: str = typer.Option(..., "--data", "-d", help="Path to training dataset"),
    project: str = typer.Option(..., "--project", "-p", help="Project name"),
    experiment_base: str = typer.Option("tuning", "--experiment", "-e", help="Base experiment name"),
    trials: int = typer.Option(20, "--trials", "-n", help="Number of trials"),
    study_name: Optional[str] = typer.Option(None, "--study", "-s", help="Optuna study name"),
    storage: Optional[str] = typer.Option(None, "--storage", help="Optuna storage URL (default: ~/.kladml/kladml.db)"),
):
    """
    Run Hyperparameter Tuning using Optuna.
    """
    console.print(f"[bold blue]🧠 KladML Tuning[/bold blue]")
    
    # 1. Load Config
    with open(config_path) as f:
        base_config = yaml.safe_load(f)
        
    console.print(f"Loaded config: {config_path}")
    
    # 2. Parse Search Space
    try:
        search_space = _parse_search_space(base_config)
    except Exception as e:
        console.print(f"[red]Error parsing search space:[/red] {e}")
        raise typer.Exit(1)
        
    console.print(f"Search Space: {len(search_space.parameters)} parameters")
    
    # 3. Setup Tuner
    real_study_name = study_name or f"{project}_{experiment_base}"
    tuner_config = TunerConfig(
        study_name=real_study_name,
        n_trials=trials,
        storage=storage,
        direction="minimize" # Usually loss
    )
    tuner = KladMLTuner(tuner_config)
    
    # 4. Resolve Dependencies (Model, Tracker)
    try:
        model_class = _resolve_model_class(model)
        console.print(f"Model: {model_class.__name__}")
    except Exception as e:
         console.print(f"[red]Model error:[/red] {e}")
         raise typer.Exit(1)
         
    tracker = LocalTracker()
    metadata = get_metadata_backend()
    
    # Ensure Project/Family exists
    if not metadata.get_project(project):
        metadata.create_project(project)
    family = "tuning" 
    if not metadata.get_family(family, project):
        metadata.create_family(family, project, "Auto-created for tuning")

    # 5. Define Objective
    def objective(params, trial):
        # 5a. Update config
        import copy
        trial_config = copy.deepcopy(base_config)
        
        # Apply params
        for k, v in params.items():
            _set_nested(trial_config, k, v)
            
        # Inject Optuna Trial for Pruning
        trial_config["optuna_trial"] = trial
        trial_config["family_name"] = family # Mandatory for base model setup
        
        # Unique experiment name per trial
        trial_exp_name = f"{experiment_base}_trial_{trial.number}"
        
        # 5b. Execute Training
        # We assume LocalTrainingExecutor handles the lifecycle
        executor = LocalTrainingExecutor(
            model_class=model_class,
            experiment_name=trial_exp_name,
            config=trial_config,
            tracker=tracker
        )
        
        # We need validation loss. execute_single returns (run_id, metrics)
        # Note: we disable stdout capturing if needed, or rely on logs
        try:
            # Maybe silence output? Tuning is verbose.
            # executor.execute_single prints. Ideally pass verbose=False.
            # For now let it print.
            run_id, metrics = executor.execute_single(data_path=data)
            
            if not metrics:
                return float('inf') # Failed run
                
            val_loss = metrics.get("val_loss", float('inf'))
            return val_loss
            
        except Exception as e:
            # If Pruned (optuna.TrialPruned), it bubbles up and Tuner handles it.
            # If other error, we might want to return inf or fail.
            # If it IS TrialPruned, we MUST re-raise it.
            import optuna
            if isinstance(e, optuna.TrialPruned):
                raise e
            console.print(f"[yellow]Trial {trial.number} failed: {e}[/yellow]")
            return float('inf')

    # 6. Run Optimization
    console.print(f"\n[bold green]Starting Optimization ({trials} trials)...[/bold green]")
    try:
        best_params = tuner.optimize(objective, search_space)
        console.print("\n[bold]🎉 Tuning Complete![/bold]")
        console.print(f"Best Params: {best_params}")
        console.print(f"Best Value: {tuner.study.best_value}")
        console.print(f"Run dashboard: optuna-dashboard {storage}")
    except KeyboardInterrupt:
        console.print("[yellow]Tuning interrupted.[/yellow]")
