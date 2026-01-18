"""
KladML CLI - Family Commands
"""

import typer
from rich.console import Console
from rich.table import Table

from kladml.backends import get_metadata_backend

app = typer.Typer()
console = Console()
metadata = get_metadata_backend()


@app.command("create")
def create_family(
    name: str = typer.Argument(..., help="Family name (e.g., 'glucose_forecasting')"),
    project: str = typer.Option(..., "--project", "-p", help="Parent project name"),
    description: str = typer.Option(None, "--description", "-d", help="Family description"),
) -> None:
    """Create a new family within a project."""
    try:
        metadata.create_family(name=name, project_name=project, description=description)
        console.print(f"[green]✓ Created family '[bold]{name}[/bold]' in project '{project}'[/green]")
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("list")
def list_families(
    project: str = typer.Option(None, "--project", "-p", help="Filter by project name"),
) -> None:
    """List all families."""
    try:
        families = metadata.list_families(project_name=project)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    
    if not families:
        console.print("[dim]No families found.[/dim]")
        return
    
    table = Table(title="Families")
    table.add_column("Name", style="cyan")
    table.add_column("Project", style="green")
    table.add_column("Experiments", justify="right")
    table.add_column("Description")
    
    for fam in families:
        exp_count = len(fam.experiment_names) if fam.experiment_names else 0
        table.add_row(
            fam.name,
            fam.project_name,
            str(exp_count),
            fam.description or "-",
        )
    
    console.print(table)


@app.command("delete")
def delete_family(
    name: str = typer.Argument(..., help="Family name to delete"),
    project: str = typer.Option(..., "--project", "-p", help="Parent project name"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Delete a family."""
    family = metadata.get_family(name=name, project_name=project)
    
    if not family:
        console.print(f"[red]Family '{name}' not found in project '{project}'.[/red]")
        raise typer.Exit(1)
    
    exp_count = len(family.experiment_names) if family.experiment_names else 0
    
    if not force:
        console.print(f"[yellow]This will delete family '{name}' with {exp_count} experiments.[/yellow]")
        confirm = typer.confirm("Are you sure?")
        if not confirm:
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Exit(0)
    
    try:
        metadata.delete_family(name=name, project_name=project)
        console.print(f"[green]✓ Deleted family '{name}'[/green]")
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
