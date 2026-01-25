
import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from kladml.cli.commands.data.utils import format_size
from kladml.cli.commands.data.inspect import analyze_pkl, analyze_parquet
from kladml.cli.process import run_pipeline

app = typer.Typer(help="Inspect and analyze datasets")
console = Console()

# Register pipeline runner
app.command("process", help="Run a data processing pipeline defined in YAML.")(run_pipeline)

@app.command("inspect")
def inspect_dataset(
    path: str = typer.Argument(..., help="Path to .pkl file or directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed info"),
) -> None:
    """
    Inspect a .pkl dataset file and show its structure.
    """
    target = Path(path)
    
    if not target.exists():
        console.print(f"[bold red]❌ Path not found:[/bold red] {path}")
        raise typer.Exit(code=1)
    
    # Get list of files to inspect
    if target.is_dir():
        files = list(target.glob("*.pkl")) + list(target.glob("*.parquet"))
        if not files:
            console.print(f"[yellow]No .pkl or .parquet files found in {path}[/yellow]")
            raise typer.Exit(code=0)
    else:
        files = [target]
    
    console.print(f"\n[bold blue]📊 Dataset Inspector[/bold blue]\n")
    
    for file_path in sorted(files):
        try:
            if file_path.suffix == ".parquet":
                info = analyze_parquet(file_path)
            else:
                info = analyze_pkl(file_path)
            
            # Classification badge
            class_colors = {
                "timeseries_list": "green",
                "dataframe_list": "cyan",
                "numpy_array": "blue",
                "scaler_coefficients": "magenta",
                "sklearn_scaler": "magenta",
                "dictionary": "yellow",
                "unknown": "red",
                "tabular": "cyan",
                "corrupt": "red"
            }
            classification = info.get("classification", "unknown")
            color = class_colors.get(classification, "white")
            
            console.print(Panel(
                f"[bold]{file_path.name}[/bold]\n"
                f"Type: [{color}]{classification}[/{color}]\n"
                f"Size: {format_size(info['size'])}\n"
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
            if "error" in info:
                table.add_row("Error", f"[red]{info['error']}[/red]")
            
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
    """
    target = Path(path)
    
    if not target.is_dir():
        console.print(f"[bold red]❌ Not a directory:[/bold red] {path}")
        raise typer.Exit(code=1)
    
    files = list(target.glob("*.pkl")) + list(target.glob("*.parquet"))
    if not files:
        console.print(f"[yellow]No .pkl or .parquet files found in {path}[/yellow]")
        raise typer.Exit(code=0)
    
    table = Table(title=f"📊 Datasets in {path}", show_header=True, header_style="bold blue")
    table.add_column("File", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Size")
    table.add_column("Items/Samples")
    table.add_column("Value Range")
    
    for file_path in sorted(files):
        try:
            if file_path.suffix == ".parquet":
                info = analyze_parquet(file_path)
            else:
                info = analyze_pkl(file_path)
            
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
                format_size(info["size"]),
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
    """
    from kladml.data.converter import convert_pkl_to_hdf5
    
    if format.lower() == "parquet":
        from kladml.data.converter import convert_pkl_to_parquet
        # Default ZSTD for parquet if user left gzip default (which is HDF5 default)
        comp = "zstd" if compression == "gzip" else compression
        
        try:
            convert_pkl_to_parquet(input_path, output_path, compression=compression)
            console.print("[green]✓ Parquet conversion successful![/green]")
            return
        except Exception as e:
            console.print(f"[red]Conversion failed:[/red] {e}")
            raise typer.Exit(code=1)

    if format.lower() != "hdf5":
        console.print(f"[red]Error:[/red] Supported formats: 'hdf5', 'parquet'")
        raise typer.Exit(code=1)
        
    input_file = Path(input_path)
    if not input_file.exists():
         console.print(f"[red]Error:[/red] Input file not found: {input_path}")
         raise typer.Exit(code=1)
         
    console.print(f"[bold]Converting dataset from PKL to HDF5...[/bold]")
    console.print(f"Input: {input_path}")
    console.print(f"Output: {output_path}")
    
    try:
        convert_pkl_to_hdf5(input_path, output_path, compression=compression)
        console.print("[green]✓ Conversion successful![/green]")
    except Exception as e:
        console.print(f"[red]Conversion failed:[/red] {e}")
        raise typer.Exit(code=1)
