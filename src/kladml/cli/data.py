"""
KladML CLI - Dataset Inspector

CLI command to analyze and classify .pkl dataset files.
"""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

app = typer.Typer(help="Inspect and analyze datasets")
console = Console()


def _format_size(size_bytes: int) -> str:
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _analyze_pkl(path: Path) -> dict:
    """Analyze a .pkl file and return metadata."""
    import joblib
    import numpy as np
    
    data = joblib.load(path)
    
    result = {
        "path": str(path),
        "size": path.stat().st_size,
        "type": type(data).__name__,
    }
    
    # Analyze based on type
    if isinstance(data, list):
        result["num_items"] = len(data)
        
        if len(data) > 0:
            first = data[0]
            result["item_type"] = type(first).__name__
            
            if isinstance(first, np.ndarray):
                # List of arrays (time series)
                lengths = [len(arr) for arr in data]
                result["series_lengths"] = {
                    "min": min(lengths),
                    "max": max(lengths),
                    "mean": np.mean(lengths),
                    "total_samples": sum(lengths),
                }
                
                # Sample statistics
                all_values = np.concatenate([arr.flatten() for arr in data[:10]])
                result["sample_stats"] = {
                    "min": float(np.min(all_values)),
                    "max": float(np.max(all_values)),
                    "mean": float(np.mean(all_values)),
                    "std": float(np.std(all_values)),
                }
                result["classification"] = "timeseries_list"
                
            elif hasattr(first, 'shape'):
                # Pandas DataFrame or similar
                result["item_shape"] = str(first.shape) if hasattr(first, 'shape') else "N/A"
                if hasattr(first, 'columns'):
                    result["columns"] = list(first.columns)[:10]
                result["classification"] = "dataframe_list"
    
    elif isinstance(data, np.ndarray):
        result["shape"] = data.shape
        result["dtype"] = str(data.dtype)
        result["sample_stats"] = {
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
        }
        result["classification"] = "numpy_array"
    
    elif isinstance(data, dict):
        result["keys"] = list(data.keys())[:20]
        result["num_keys"] = len(data)
        
        # Check if it's a scaler
        if 'mean_' in data or 'scale_' in data or 'mean' in data:
            result["classification"] = "scaler_coefficients"
            if 'mean_' in data:
                result["scaler_mean"] = float(data['mean_'][0]) if hasattr(data['mean_'], '__getitem__') else float(data['mean_'])
            if 'scale_' in data:
                result["scaler_scale"] = float(data['scale_'][0]) if hasattr(data['scale_'], '__getitem__') else float(data['scale_'])
        else:
            result["classification"] = "dictionary"
    
    elif hasattr(data, 'mean_') and hasattr(data, 'scale_'):
        # sklearn scaler object
        result["classification"] = "sklearn_scaler"
        result["scaler_mean"] = float(data.mean_[0])
        result["scaler_scale"] = float(data.scale_[0])
    
    else:
        result["classification"] = "unknown"
        if hasattr(data, 'shape'):
            result["shape"] = data.shape
    
    return result


@app.command("inspect")
def inspect_dataset(
    path: str = typer.Argument(..., help="Path to .pkl file or directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed info"),
) -> None:
    """
    Inspect a .pkl dataset file and show its structure.
    
    Example:
        kladml data inspect ./data/datasets/glucose/cgm_um_train.pkl
        kladml data inspect ./data/datasets/glucose/  # Inspect all in directory
    """
    import numpy as np
    
    target = Path(path)
    
    if not target.exists():
        console.print(f"[bold red]❌ Path not found:[/bold red] {path}")
        raise typer.Exit(code=1)
    
    # Get list of files to inspect
    if target.is_dir():
        files = list(target.glob("*.pkl"))
        if not files:
            console.print(f"[yellow]No .pkl files found in {path}[/yellow]")
            raise typer.Exit(code=0)
    else:
        files = [target]
    
    console.print(f"\n[bold blue]📊 Dataset Inspector[/bold blue]\n")
    
    for file_path in sorted(files):
        try:
            info = _analyze_pkl(file_path)
            
            # Classification badge
            class_colors = {
                "timeseries_list": "green",
                "dataframe_list": "cyan",
                "numpy_array": "blue",
                "scaler_coefficients": "magenta",
                "sklearn_scaler": "magenta",
                "dictionary": "yellow",
                "unknown": "red",
            }
            classification = info.get("classification", "unknown")
            color = class_colors.get(classification, "white")
            
            console.print(Panel(
                f"[bold]{file_path.name}[/bold]\n"
                f"Type: [{color}]{classification}[/{color}]\n"
                f"Size: {_format_size(info['size'])}\n"
                f"Python Type: {info['type']}",
                title=f"📁 {file_path.stem}",
                border_style=color,
            ))
            
            # Details table
            table = Table(show_header=True, header_style="bold")
            table.add_column("Property", style="dim")
            table.add_column("Value")
            
            if "num_items" in info:
                table.add_row("Items", str(info["num_items"]))
            if "item_type" in info:
                table.add_row("Item Type", info["item_type"])
            if "series_lengths" in info:
                sl = info["series_lengths"]
                table.add_row("Series Length (min)", str(int(sl["min"])))
                table.add_row("Series Length (max)", str(int(sl["max"])))
                table.add_row("Series Length (avg)", f"{sl['mean']:.1f}")
                table.add_row("Total Samples", f"{int(sl['total_samples']):,}")
            if "sample_stats" in info:
                ss = info["sample_stats"]
                table.add_row("Value Range", f"{ss['min']:.2f} - {ss['max']:.2f}")
                table.add_row("Mean ± Std", f"{ss['mean']:.2f} ± {ss['std']:.2f}")
            if "scaler_mean" in info:
                table.add_row("Scaler Mean", f"{info['scaler_mean']:.4f}")
            if "scaler_scale" in info:
                table.add_row("Scaler Scale", f"{info['scaler_scale']:.4f}")
            if "shape" in info:
                table.add_row("Shape", str(info["shape"]))
            if "keys" in info:
                table.add_row("Keys", ", ".join(str(k) for k in info["keys"][:5]))
            if "columns" in info:
                table.add_row("Columns", ", ".join(str(c) for c in info["columns"][:5]))
            
            console.print(table)
            console.print()
            
        except Exception as e:
            console.print(f"[red]Error analyzing {file_path.name}: {e}[/red]")


@app.command("summary")
def summary_directory(
    path: str = typer.Argument(".", help="Directory containing datasets"),
) -> None:
    """
    Show a summary table of all datasets in a directory.
    
    Example:
        kladml data summary ./data/datasets/glucose/
    """
    target = Path(path)
    
    if not target.is_dir():
        console.print(f"[bold red]❌ Not a directory:[/bold red] {path}")
        raise typer.Exit(code=1)
    
    files = list(target.glob("*.pkl"))
    if not files:
        console.print(f"[yellow]No .pkl files found in {path}[/yellow]")
        raise typer.Exit(code=0)
    
    table = Table(title=f"📊 Datasets in {path}", show_header=True, header_style="bold blue")
    table.add_column("File", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Size")
    table.add_column("Items/Samples")
    table.add_column("Value Range")
    
    for file_path in sorted(files):
        try:
            info = _analyze_pkl(file_path)
            
            items = ""
            if "num_items" in info:
                items = str(info["num_items"])
                if "series_lengths" in info:
                    items += f" ({int(info['series_lengths']['total_samples']):,} samples)"
            elif "shape" in info:
                items = str(info.get("shape", ""))
            
            value_range = ""
            if "sample_stats" in info:
                ss = info["sample_stats"]
                value_range = f"{ss['min']:.0f} - {ss['max']:.0f}"
            elif "scaler_mean" in info:
                value_range = f"μ={info['scaler_mean']:.2f}"
            
            table.add_row(
                file_path.name,
                info.get("classification", "unknown"),
                _format_size(info["size"]),
                items,
                value_range,
            )
        except Exception as e:
            table.add_row(file_path.name, "[red]error[/red]", "", "", str(e)[:20])
    
    console.print(table)


@app.command("convert")
def convert_dataset(
    input_path: str = typer.Option(..., "--input", "-i", help="Path to input .pkl file"),
    output_path: str = typer.Option(..., "--output", "-o", help="Path to output .h5 file"),
    format: str = typer.Option("hdf5", "--format", "-f", help="Output format (currently only 'hdf5')"),
    compression: str = typer.Option("gzip", "--compression", "-z", help="Compression (gzip, lzf, none)"),
) -> None:
    """
    Convert a dataset from PKL to HDF5 format.
    
    Example:
        kladml data convert -i data.pkl -o data.h5
    """
    from kladml.data.converter import convert_pkl_to_hdf5
    
    if format.lower() != "hdf5":
        console.print(f"[red]Error:[/red] Only 'hdf5' format is currently supported.")
        raise typer.Exit(code=1)
        
    input_file = Path(input_path)
    if not input_file.exists():
         console.print(f"[red]Error:[/red] Input file not found: {input_path}")
         raise typer.Exit(code=1)
         
    console.print(f"[bold]Converting dataset from PKL to HDF5...[/bold]")
    console.print(f"Input: {input_path}")
    console.print(f"Output: {output_path}")
    
    try:
        stats = convert_pkl_to_hdf5(input_path, output_path, compression=compression)
        console.print("[green]✓ Conversion successful![/green]")
    except Exception as e:
        console.print(f"[red]Conversion failed:[/red] {e}")
        raise typer.Exit(code=1)

@app.command("process")
def process_data(
    input_source: str = typer.Option(..., "--input", "-i", help="Input data (directory or file)"),
    pipeline_config: str = typer.Option(..., "--pipeline", "-p", help="Path to pipeline YAML config"),
) -> None:
    """
    Run a data processing pipeline defined in YAML.
    
    The pipeline is composed of registered components (Parsers, Cleaners, Splitters).
    
    Example:
        kladml data process -i data/raw -p config/canbus_pipeline.yaml
    """
    from kladml.data.pipeline import DataPipeline
    from kladml.data.defaults import register_all_components
    
    console.print(f"[bold]Running Data Pipeline...[/bold]")
    console.print(f"Input: {input_source}")
    console.print(f"Config: {pipeline_config}")
    
    # 1. Register Components
    register_all_components()
    
    # 2. Load Pipeline
    try:
        pipeline = DataPipeline.from_yaml(pipeline_config)
    except Exception as e:
        console.print(f"[bold red]Error loading pipeline:[/bold red] {e}")
        raise typer.Exit(code=1)
        
    # 3. Execute
    try:
        # We pass input_source to the first component
        result = pipeline.transform(input_source)
        
        # If result is a dict (splitter) or string (file path), print it
        if isinstance(result, (str, dict)):
            console.print(f"[green]✓ Pipeline complete![/green]")
            console.print(f"Result: {result}")
        else:
             console.print(f"[green]✓ Pipeline complete![/green]")
             
    except Exception as e:
        console.print(f"[red]Pipeline execution failed:[/red] {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
