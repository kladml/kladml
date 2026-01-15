"""
KladML CLI - Main Entry Point

Provides a rich CLI for:
- Project management
- Experiment management
- Training (single and grid search)
- Run management
"""

import typer
from rich.console import Console

app = typer.Typer(
    name="kladml",
    help="🚀 KladML - Local ML Training & Experiment Tracking",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()

# Import subcommands
from kladml.cli import run, project
from kladml.cli import projects, experiments, train


# Register subcommands
app.add_typer(projects.app, name="project", help="Manage projects")
app.add_typer(experiments.app, name="experiment", help="Manage experiments")
app.add_typer(train.app, name="train", help="Train models")
app.add_typer(run.app, name="run", help="Run scripts and manage runs")


@app.command()
def version():
    """Show KladML version."""
    from kladml import __version__
    console.print(f"[bold blue]KladML[/bold blue] version [green]{__version__}[/green]")


@app.command()
def init(
    name: str = typer.Argument(..., help="Project name"),
    template: str = typer.Option("default", "--template", "-t", help="Project template"),
):
    """Initialize a new KladML project."""
    from kladml.cli.project import do_init
    do_init(name, template)


if __name__ == "__main__":
    app()

