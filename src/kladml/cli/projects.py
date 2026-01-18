"""
Project management CLI commands for KladML.

Provides commands for:
- Creating projects
- Listing projects
- Deleting projects
"""

import typer
from typing import Optional
from rich.console import Console
from rich.table import Table

from kladml.backends import get_metadata_backend

app = typer.Typer(help="Manage KladML projects")
console = Console()
metadata = get_metadata_backend()


@app.command("create")
def create_project(
    name: str = typer.Argument(..., help="Project name"),
    description: Optional[str] = typer.Option(None, "--description", "-d", help="Project description"),
) -> None:
    """
    Create a new project.
    
    Example:
        kladml project create my-forecaster --description "Glucose forecasting models"
    """
    try:
        project = metadata.create_project(name=name, description=description)
        console.print(f"[green]✓[/green] Created project '{name}' (id: {project.id})")
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


@app.command("list")
def list_projects() -> None:
    """
    List all projects.
    
    Example:
        kladml project list
    """
    projects = metadata.list_projects()
    
    if not projects:
        console.print("[yellow]No projects found.[/yellow]")
        console.print("Create one with: [bold]kladml project create <name>[/bold]")
        return
    
    table = Table(title="Projects")
    table.add_column("ID", style="dim")
    table.add_column("Name", style="bold")
    table.add_column("Description")
    table.add_column("Families", justify="right")
    table.add_column("Created", style="dim")
    
    for project in projects:
        table.add_row(
            project.id,
            project.name,
            project.description or "-",
            str(project.family_count),
            project.created_at.strftime("%Y-%m-%d %H:%M") if project.created_at else "-",
        )
    
    console.print(table)


@app.command("delete")
def delete_project(
    name: str = typer.Argument(..., help="Project name to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """
    Delete a project.
    
    Note: This only deletes the local project definition. 
    MLflow experiments and runs are preserved.
    """
    project = metadata.get_project(name)
    
    if not project:
        console.print(f"[red]Error:[/red] Project '{name}' not found")
        raise typer.Exit(code=1)
    
    if not force:
        console.print(f"[yellow]Warning:[/yellow] This will delete:")
        console.print(f"  - Project: {name}")
        console.print(f"  - {project.family_count} linked families")
        console.print("\n[dim]Note: Actual MLflow experiments and runs will NOT be deleted.[/dim]")
        
        confirm = typer.confirm("Are you sure?")
        if not confirm:
            console.print("Cancelled")
            raise typer.Exit(code=0)
    
    try:
        metadata.delete_project(name)
        console.print(f"[green]✓[/green] Deleted project '{name}'")
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


@app.command("show")
def show_project(
    name: str = typer.Argument(..., help="Project name"),
) -> None:
    """
    Show details of a project.
    
    Example:
        kladml project show my-forecaster
    """
    project = metadata.get_project(name)
    
    if not project:
        console.print(f"[red]Error:[/red] Project '{name}' not found")
        raise typer.Exit(code=1)
    
    console.print(f"\n[bold]Project: {project.name}[/bold]")
    console.print(f"ID: {project.id}")
    console.print(f"Description: {project.description or '-'}")
    console.print(f"Created: {project.created_at}")
    console.print(f"Families: {project.family_count}")
    
    families = metadata.list_families(project_name=name)
    
    if families:
        console.print(f"\n[bold]Families ({len(families)}):[/bold]")
        for fam in families:
            console.print(f"  • {fam.name} ({len(fam.experiment_names)} experiments)")
    else:
        console.print("\n[yellow]No families yet.[/yellow]")
        console.print(f"Create one with: [bold]kladml family create <name> -p {name}[/bold]")


if __name__ == "__main__":
    app()
