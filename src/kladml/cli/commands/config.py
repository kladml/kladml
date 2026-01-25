
import typer
import yaml
from pathlib import Path
from rich.console import Console
from kladml.utils.inspection import generate_smart_config

app = typer.Typer(help="Configuration management utilities")
console = Console()

@app.command("create")
def create_config(
    model: str = typer.Option(..., "--model", "-m", help="Model name (e.g. gluformer)"),
    data: str = typer.Option(None, "--data", "-d", help="Path to training data for smart heuristics"),
    output: str = typer.Option("config.yaml", "--output", "-o", help="Output path"),
):
    """
    Generate a 'smart' configuration file for a model.
    
    If --data is provided, it inspects the dataset to auto-tune parameters 
    (input dimensions, batch size, etc.).
    """
    try:
        console.print(f"[bold cyan]Generating config for model: {model}[/bold cyan]")
        if data:
            console.print(f"Analyzing dataset: {data}...")
            
        config = generate_smart_config(model, data)
        
        # Add a wrapper for project structure if not present
        if "project" not in config:
             # Standard KladML config structure wrapper
             full_config = {
                 "project": {
                     "name": config.get("project_name", "my-project"),
                     "experiment": config.get("experiment_name", "experiment-1")
                 },
                 "model": {
                     "name": model,
                     # Flatten model params here or keep nested depending on how Trainer reads it.
                     # Trainer reads flat dict passed to model.
                     # But config file usually has sections.
                     # Let's flatten the known architectural params under 'params' if we follow getting_started
                     # But BaseModel receives a flat dict 'config'.
                     # Let's adhere to the flat injection structure or specific sub-keys.
                     # For simplicity and compat with 'kladml train --config', we usually map YAML sections to args.
                     # Let's output a flat structure that matches BaseModel expectation, or a sectioned one if CLI parses it.
                     # The current 'kladml train' CLI (refactored) loads YAML and passes parts of it.
                     # Let's verify 'load_config' logic in train command.
                     # Assuming flat for now for the model part.
                 }
             }
             # Merge the rest
             for k, v in config.items():
                 # Filter out project keys we moved
                 if k not in ["project_name", "experiment_name"]:
                     full_config["model"][k] = v
                     
             # Add training defaults if missing
             full_config["training"] = {
                 "batch_size": config.get("batch_size", 64),
                 "epochs": config.get("epochs", 100),
                 "device": "auto"
             }
             
             final_output = full_config
        else:
            final_output = config

        # Save
        with open(output, "w") as f:
            yaml.dump(final_output, f, sort_keys=False, default_flow_style=False)
            
        console.print(f"[green]✅ Configuration saved to {output}[/green]")
        console.print(f"Edit key parameters before training.")
        
    except Exception as e:
        console.print(f"[red]Failed to generate config: {e}[/red]")
        raise typer.Exit(1)
