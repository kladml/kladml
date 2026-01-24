"""
KladML CLI - Main Entry Point

Provides a rich CLI for:
- Project management
- Experiment management
- Training (single and grid search)
- Run management
"""

import typer
from kladml.config.logging import setup_logging
from rich.console import Console

app = typer.Typer(
    name="kladml",
    help="🚀 KladML - Local ML Training & Experiment Tracking",
    add_completion=False,
    no_args_is_help=True,
)

@app.callback()
def main(ctx: typer.Context):
    """
    Manage KladML projects and experiments.
    """
    # Configure logging (Structured Loguru)
    setup_logging()

console = Console()

# Import subcommands
from kladml.cli import run, family
from kladml.cli import projects, experiments, train, data, registry, export, evaluate
from kladml.cli.compare import compare_runs

# Register subcommands
app.add_typer(projects.app, name="project", help="Manage projects")
app.add_typer(family.app, name="family", help="Manage families")
app.add_typer(experiments.app, name="experiment", help="Manage experiments")
app.add_typer(train.app, name="train", help="Train models")
app.add_typer(run.app, name="run", help="Run scripts and manage runs")
app.add_typer(data.app, name="data", help="Inspect and analyze datasets")

# New Commands (v0.6.0)
app.add_typer(registry.app, name="registry", help="Manage artifact registry")
app.add_typer(export.app, name="export", help="Export models (ONNX, TorchScript)")

app.add_typer(evaluate.app, name="eval", help="Evaluate trained models")

app.command("compare", help="Compare runs side-by-side")(compare_runs)


@app.command()
def version():
    """Show KladML version."""
    from kladml import __version__
    console.print(f"[bold blue]KladML[/bold blue] version [green]{__version__}[/green]")


@app.command()
def init(
    force: bool = typer.Option(False, "--force", "-f", help="Re-create structure even if exists"),
):
    """
    Initialize a KladML workspace in the current directory.
    
    Creates standard data directory structure.
    """
    from kladml.cli.init import init_workspace
    init_workspace(force)


@app.command()
def ui():
    """
    Launch the interactive Terminal User Interface (TUI).
    """
    from kladml.tui.app import run_tui
    run_tui()

if __name__ == "__main__":
    app()

