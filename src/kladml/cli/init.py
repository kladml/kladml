"""
Initialization CLI command.

Bootstraps the local KladML workspace structure.
"""

import typer
from rich.console import Console
from rich.tree import Tree
from pathlib import Path
from kladml.utils.paths import ensure_data_structure, get_root_data_path

app = typer.Typer(help="Initialize KladML workspace")
console = Console()

@app.command("init")
def init_workspace(
    force: bool = typer.Option(False, "--force", "-f", help="Re-create structure even if exists"),
) -> None:
    """
    Initialize a KladML workspace in the current directory.
    
    Creates the standard directory structure:
    data/
    ├── datasets/
    ├── preprocessors/
    ├── models/
    └── projects/
    """
    try:
        root = get_root_data_path()
        if root.exists() and not force:
            console.print(f"[yellow]Workspace already initialized at: {root}[/yellow]")
            console.print("Use --force to re-create/ensure structure.")
            return
            
        root = ensure_data_structure()
        
        console.print(f"[green]✓[/green] KladML workspace initialized at: [bold]{Path.cwd()}[/bold]")
        
        # Show tree
        tree = Tree(f"📁 {Path.cwd().name}")
        data_node = tree.add("📁 data")
        data_node.add("📁 datasets/")
        data_node.add("📁 preprocessors/")
        data_node.add("📁 models/")
        data_node.add("📁 projects/")
        
        console.print(tree)
        
    except Exception as e:
        console.print(f"[red]Error initializing workspace:[/red] {e}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
