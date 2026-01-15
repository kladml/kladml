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

from kladml.db import Project, init_db
from kladml.db.session import session_scope

app = typer.Typer(help="Manage KladML projects")
console = Console()


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
    init_db()
    
    with session_scope() as session:
        # Check if project exists
        existing = session.query(Project).filter_by(name=name).first()
        if existing:
            console.print(f"[red]Error:[/red] Project '{name}' already exists")
            raise typer.Exit(code=1)
        
        project = Project(name=name, description=description)
        session.add(project)
        session.flush()
        project_id = project.id
    
    console.print(f"[green]✓[/green] Created project '{name}' (id: {project_id})")


@app.command("list")
def list_projects() -> None:
    """
    List all projects.
    
    Example:
        kladml project list
    """
    init_db()
    
    with session_scope() as session:
        projects = session.query(Project).order_by(Project.created_at.desc()).all()
        
        if not projects:
            console.print("[yellow]No projects found.[/yellow]")
            console.print("Create one with: [bold]kladml project create <name>[/bold]")
            return
        
        table = Table(title="Projects")
        table.add_column("ID", style="dim")
        table.add_column("Name", style="bold")
        table.add_column("Description")
        table.add_column("Experiments", justify="right")
        table.add_column("Created", style="dim")
        
        for project in projects:
            exp_count = len(project.experiment_names) if project.experiment_names else 0
            
            table.add_row(
                project.id,
                project.name,
                project.description or "-",
                str(exp_count),
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
    init_db()
    
    with session_scope() as session:
        project = session.query(Project).filter_by(name=name).first()
        
        if not project:
            console.print(f"[red]Error:[/red] Project '{name}' not found")
            raise typer.Exit(code=1)
        
        exp_count = len(project.experiment_names) if project.experiment_names else 0
        
        if not force:
            console.print(f"[yellow]Warning:[/yellow] This will delete:")
            console.print(f"  - Project: {name}")
            console.print(f"  - {exp_count} linked experiment(s)")
            console.print("\n[dim]Note: Actual MLflow experiments and runs will NOT be deleted.[/dim]")
            
            confirm = typer.confirm("Are you sure?")
            if not confirm:
                console.print("Cancelled")
                raise typer.Exit(code=0)
        
        session.delete(project)
    
    console.print(f"[green]✓[/green] Deleted project '{name}'")


@app.command("show")
def show_project(
    name: str = typer.Argument(..., help="Project name"),
) -> None:
    """
    Show details of a project.
    
    Example:
        kladml project show my-forecaster
    """
    init_db()
    
    with session_scope() as session:
        project = session.query(Project).filter_by(name=name).first()
        
        if not project:
            console.print(f"[red]Error:[/red] Project '{name}' not found")
            raise typer.Exit(code=1)
        
        console.print(f"\n[bold]Project: {project.name}[/bold]")
        console.print(f"ID: {project.id}")
        console.print(f"Description: {project.description or '-'}")
        console.print(f"Created: {project.created_at}")
        
        experiment_names = project.experiment_names or []
        
        if experiment_names:
            console.print(f"\n[bold]Experiments ({len(experiment_names)}):[/bold]")
            for exp_name in experiment_names:
                console.print(f"  • {exp_name}")
        else:
            console.print("\n[yellow]No experiments yet.[/yellow]")
            console.print(f"Create one with: [bold]kladml experiment create -p {name} -n <experiment-name>[/bold]")


if __name__ == "__main__":
    app()
