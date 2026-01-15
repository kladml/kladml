"""
KladML CLI - Project Commands
"""

import typer
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

app = typer.Typer()
console = Console()


def do_init(name: str, template: str = "default"):
    """Initialize a new KladML project."""
    import yaml
    
    project_path = Path.cwd() / name
    
    if project_path.exists():
        console.print(f"[bold red]❌ Directory '{name}' already exists.[/bold red]")
        raise typer.Exit(code=1)
    
    console.print(Panel.fit(f"[bold blue]🚀 Initializing project:[/bold blue] {name}"))
    
    # Create directory structure
    project_path.mkdir(parents=True)
    (project_path / "data").mkdir()
    (project_path / "models").mkdir()
    (project_path / "experiments").mkdir()
    
    # Create kladml.yaml config
    config = {
        "project": {
            "name": name,
            "version": "0.1.0",
        },
        "training": {
            "device": "auto",  # auto | cpu | cuda | mps
        },
        "storage": {
            "backend": "local",  # local | s3
        },
    }
    
    with open(project_path / "kladml.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Create README
    with open(project_path / "README.md", "w") as f:
        f.write(f"# {name}\n\nKladML Project\n")
    
    # Create example train script based on template
    if template == "timeseries":
        train_content = '''"""
Example: Time Series Forecasting with KladML
"""
from kladml import TimeSeriesModel, MLTask

class MyForecaster(TimeSeriesModel):
    @property
    def ml_task(self):
        return MLTask.TIMESERIES_FORECASTING
    
    def train(self, X_train, y_train=None, **kwargs):
        # Your training logic here
        return {"loss": 0.1}
    
    def predict(self, X, **kwargs):
        # Your prediction logic here
        pass
    
    def evaluate(self, X_test, y_test=None, **kwargs):
        return {"mae": 0.5}
    
    def save(self, path: str):
        pass
    
    def load(self, path: str):
        pass

if __name__ == "__main__":
    model = MyForecaster()
    # Train, evaluate, etc.
'''
    else:
        train_content = '''"""
KladML Training Script
"""

print("Hello from KladML!")
# TODO: Implement your training logic here
'''
    
    with open(project_path / "train.py", "w") as f:
        f.write(train_content)
    
    console.print(f"[bold green]✅ Project created at[/bold green] {project_path}")
    console.print("\n[dim]Next steps:[/dim]")
    console.print(f"  cd {name}")
    console.print("  kladml run train.py")


@app.command("config")
def show_config():
    """Show current project configuration."""
    import yaml
    
    config_path = Path.cwd() / "kladml.yaml"
    if not config_path.exists():
        console.print("[bold red]❌ No kladml.yaml found.[/bold red] Run [cyan]kladml init[/cyan] first.")
        raise typer.Exit(code=1)
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    console.print_json(data=config)
