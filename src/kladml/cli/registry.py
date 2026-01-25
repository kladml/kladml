"""
Registry management CLI commands.
Interact with the RegistryArtifact table in the unified database.
"""

import typer
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.table import Table
from sqlmodel import select

from kladml.db.session import get_session
from kladml.db.models import RegistryArtifact

app = typer.Typer(help="Manage the Artifact Registry (Models, Preprocessors, etc.)")
console = Console()



@app.command("list")
def list_artifacts(
    type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by artifact type (model, dataset, etc.)"),
    tag: Optional[str] = typer.Option(None, "--tag", help="Filter by tag"),
    limit: int = typer.Option(50, "--limit", "-n", help="Max number of results"),
):
    """
    List artifacts in the registry.
    """
    with get_session() as session:
        query = select(RegistryArtifact)
        
        if type:
            query = query.where(RegistryArtifact.artifact_type == type)
            
        # Basic tag filtering (exact match in list would require json searching, 
        # for SQLite basic implementation we fetch and filter in python if needed, 
        # or use simple contains if supported. For now, we list all if tag is complex).
        
        query = query.order_by(RegistryArtifact.updated_at.desc()).limit(limit)
        artifacts = session.exec(query).all()
        
        if tag:
            # Client-side filtering for SQLite JSON simplicity
            artifacts = [a for a in artifacts if tag in a.tags]

        if not artifacts:
            console.print("No artifacts found.")
            return

        table = Table(title=f"Registry Artifacts ({len(artifacts)})")
        table.add_column("ID", justify="right", style="cyan")
        table.add_column("Name", style="magenta")
        table.add_column("Version", style="green")
        table.add_column("Type", style="yellow")
        table.add_column("Status")
        table.add_column("Tags")
        table.add_column("Updated", justify="right")

        for art in artifacts:
            tags_str = ", ".join(art.tags) if art.tags else ""
            table.add_row(
                str(art.id),
                art.name,
                art.version,
                art.artifact_type,
                art.status,
                tags_str,
                art.updated_at.strftime("%Y-%m-%d %H:%M")
            )

        console.print(table)


@app.command("show")
def show_artifact(
    name: str = typer.Argument(..., help="Name of the artifact"),
    version: Optional[str] = typer.Option(None, "--version", "-v", help="Specific version (default: latest)"),
):
    """
    Show details of a specific artifact.
    """
    with get_session() as session:
        query = select(RegistryArtifact).where(RegistryArtifact.name == name)
        
        if version:
            query = query.where(RegistryArtifact.version == version)
        else:
            # Get latest by updated_at
            query = query.order_by(RegistryArtifact.updated_at.desc())
            
        artifact = session.exec(query).first()
        
        if not artifact:
            console.print(f"[red]Artifact '{name}' not found.[/red]")
            raise typer.Exit(1)

        console.print(f"[bold cyan]Artifact Details: {artifact.name}[/bold cyan]")
        console.print(f"ID: {artifact.id}")
        console.print(f"Version: {artifact.version}")
        console.print(f"Type: {artifact.artifact_type}")
        console.print(f"Path: {artifact.path}")
        console.print(f"Status: {artifact.status}")
        console.print(f"Run ID: {artifact.run_id}")
        console.print(f"Tags: {artifact.tags}")
        console.print(f"Created: {artifact.created_at}")
        
        if artifact.metadata_json:
            console.print("\n[bold]Metadata:[/bold]")
            console.print(artifact.metadata_json)


@app.command("add")
def add_artifact(
    name: str = typer.Option(..., "--name", "-n", help="Artifact name"),
    path: str = typer.Option(..., "--path", "-p", help="Path to artifact file/dir"),
    type: str = typer.Option("model", "--type", "-t", help="Type (model, preprocessor, etc.)"),
    version: str = typer.Option("v1", "--version", "-v", help="Version string"),
    tags: list[str] = typer.Option([], "--tag", help="Tags"),
    status: str = typer.Option("production", "--status", help="Status"),
):
    """
    Add (track) an existing artifact in the registry.
    """
    path_obj = Path(path).resolve()
    if not path_obj.exists():
        console.print(f"[yellow]Warning: Path {path} does not exist locally.[/yellow]")
    
    with get_session() as session:
        # Check duplicate
        existing = session.exec(
            select(RegistryArtifact)
            .where(RegistryArtifact.name == name)
            .where(RegistryArtifact.version == version)
        ).first()
        
        if existing:
            console.print(f"[red]Artifact {name} version {version} already exists (ID: {existing.id}).[/red]")
            raise typer.Exit(1)
            
        artifact = RegistryArtifact(
            name=name,
            version=version,
            artifact_type=type,
            path=str(path_obj),
            status=status,
            tags=tags,
            metadata_json={"registered_by": "cli"}
        )
        session.add(artifact)
        session.commit()
        session.refresh(artifact)
        
        console.print(f"[green]Registered artifact {name} (ID: {artifact.id})[/green]")
