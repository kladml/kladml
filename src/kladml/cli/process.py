
import typer
from rich.console import Console
from pathlib import Path
import polars as pl
from kladml.data.pipeline import DataPipeline
from kladml.data.components.io import ParquetWriter

app = typer.Typer(help="Run declarative data pipelines")
console = Console()

@app.command("run")
def run_pipeline(
    pipeline_path: str = typer.Option(..., "--pipeline", "-p", help="Path to pipeline YAML"),
    input_path: str = typer.Option(..., "--input", "-i", help="Input data path (file or directory)"),
    output_path: str = typer.Option(None, "--output", "-o", help="Output path (override last step or append writer)"),
):
    """
    Execute a data processing pipeline defined in YAML.

    Arguments:
        pipeline_path: Path to YAML config (e.g. config/processing.yaml)
        input_path: Source data (e.g. data/raw/logs or data.parquet)
        output_path: Optional override for saving result
    """
    console.print(f"[bold blue]🚀 Starting Pipeline:[/bold blue] {pipeline_path}")
    
    try:
        # 1. Load Strategy
        pipeline = DataPipeline.from_yaml(pipeline_path)
        console.print(f"[green]Loaded pipeline with {len(pipeline.steps)} steps.[/green]")
        for i, step in enumerate(pipeline.steps):
            console.print(f"  {i+1}. {step.__class__.__name__}")
            
        # 2. Execute
        with console.status("[bold green]Processing...[/bold green]"):
            result = pipeline.transform(input_path)
            
        # 3. Handle Output
        if isinstance(result, (pl.DataFrame, pl.LazyFrame)):
            if output_path:
                console.print(f"Saving result to {output_path}...")
                if isinstance(result, pl.LazyFrame):
                    result = result.collect()
                
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                result.write_parquet(output_path)
                console.print("[green]Saved successfully.[/green]")
            else:
                console.print(f"[yellow]Result is DataFrame ({result.height} rows), but no --output specified.[/yellow]")
        elif isinstance(result, str):
            # Assumed path returned
            console.print(f"[green]Pipeline finished. Output: {result}[/green]")
        elif isinstance(result, dict):
             # Splitter returns dict of paths
             console.print(f"[green]Pipeline finished. Outputs: {result}[/green]")
        else:
             console.print(f"[yellow]Pipeline finished with result type: {type(result)}[/yellow]")
             
    except Exception as e:
        console.print(f"[red]Pipeline Error:[/red] {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)
