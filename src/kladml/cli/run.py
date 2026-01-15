"""
KladML CLI - Run Commands
"""

import typer
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer()
console = Console()


@app.command("local")
def run_local(
    script: str = typer.Argument(..., help="Python script to run"),
    device: str = typer.Option("auto", "--device", "-d", help="Device: auto|cpu|cuda|mps"),
    runtime: str = typer.Option("auto", "--runtime", "-r", help="Container runtime: auto|docker|podman"),
    image: str = typer.Option(None, "--image", "-i", help="Custom Docker image to use"),
):
    """
    Run training locally using a container runtime (Docker, Podman, etc).
    """
    import subprocess
    import os
    import shutil
    
    script_path = Path(script)
    if not script_path.exists():
        console.print(f"[bold red]❌ Script not found:[/bold red] {script}")
        raise typer.Exit(code=1)
    
    # 1. Detect Runtime
    if runtime == "auto":
        if shutil.which("docker"):
            runtime_cmd = "docker"
        elif shutil.which("podman"):
            runtime_cmd = "podman"
        else:
            console.print("[bold red]❌ No container runtime found (Docker/Podman).[/bold red]")
            console.print("[yellow]💡 Tip: Use 'kladml run native <script>' to run in your local Python environment.[/yellow]")
            raise typer.Exit(code=1)
    else:
        if not shutil.which(runtime):
             console.print(f"[bold red]❌ Runtime '{runtime}' not found in PATH.[/bold red]")
             raise typer.Exit(code=1)
        runtime_cmd = runtime

    # 2. Detect Device & Image
    if device == "auto":
        # Try to detect CUDA via nvidia-smi
        try:
            subprocess.run(
                ["nvidia-smi"], capture_output=True, check=True
            )
            device = "cuda"
        except (FileNotFoundError, subprocess.CalledProcessError):
            device = "cpu"
    
    # Select image
    if image is None:
        image_map = {
            "cpu": "ghcr.io/kladml/worker:cpu",
            "cuda": "ghcr.io/kladml/worker:cuda12",
            "cuda11": "ghcr.io/kladml/worker:cuda11",
            "cuda12": "ghcr.io/kladml/worker:cuda12",
            "mps": "ghcr.io/kladml/worker:cpu",  # MPS runs on host/cpu image mainly
        }
        docker_image = image_map.get(device, image_map["cpu"])
    else:
        docker_image = image
    
    console.print(Panel.fit(
        f"[bold blue]🐳 Running with {runtime_cmd.capitalize()}[/bold blue]\n"
        f"Runtime: [cyan]{runtime_cmd}[/cyan]\n"
        f"Image: [cyan]{docker_image}[/cyan]\n"
        f"Script: [cyan]{script}[/cyan]\n"
        f"Device: [cyan]{device}[/cyan]"
    ))
    
    # 3. Build Command
    cwd = os.getcwd()
    cmd = [
        runtime_cmd, "run", "--rm",
        "-v", f"{cwd}:/workspace",
        "-w", "/workspace",
    ]
    
    # Add GPU support
    if device.startswith("cuda"):
        if runtime_cmd == "docker":
            cmd.extend(["--gpus", "all"])
        elif runtime_cmd == "podman":
            # Podman uses --device nvidia.com/gpu=all or hooks
            # Simplest often is --device nvidia.com/gpu=all if cdi is set up, 
            # or --gpus all checks for nvidia-container-toolkit compatibility
            cmd.extend(["--device", "nvidia.com/gpu=all"]) 
            # Note: Podman GPU support varies by version/setup. 
            # Some setups use --security-opt=label=disable --hooks-dir...
    
    # Add environment variables
    # 1. Project Config (kladml.yaml)
    from kladml.backends.local_config import YamlConfig
    config = YamlConfig()
    
    # Pass common KladML variables
    env_vars = {
        "KLADML_PROJECT_NAME": config.get("project.name", "unknown"),
        "KLADML_TRAINING_DEVICE": device,
    }
    
    # 3. Add to command
    for k, v in env_vars.items():
        if v:  # Only pass if value exists
             cmd.extend(["-e", f"{k}={v}"])
    
    cmd.extend([docker_image, "python", script])
    
    console.print(f"[dim]Executing: {' '.join(cmd)}[/dim]\n")
    
    # 4. Execute
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        
        for line in process.stdout:
            console.print(line, end="")
        
        process.wait()
        
        if process.returncode == 0:
            console.print("\n[bold green]✅ Run completed successfully.[/bold green]")
        else:
            console.print(f"\n[bold red]❌ Run failed with code {process.returncode}[/bold red]")
            
            # Suggest fixes for common GPU runtime errors
            if process.returncode == 126 and device.startswith("cuda") and runtime_cmd == "podman":
                console.print(Panel(
                    "[bold yellow]💡 GPU Runtime Hint:[/bold yellow]\n"
                    "It looks like Podman failed to access the GPU (CDI error).\n"
                    "Please execute the following command on your host to generate the NVIDIA CDI config:\n\n"
                    "  [bold cyan]sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml[/bold cyan]\n\n"
                    "Then retry this command.",
                    title="Setup Required",
                    border_style="yellow"
                ))
                
            raise typer.Exit(code=process.returncode)
            
    except Exception as e:
        console.print(f"[bold red]❌ Error:[/bold red] {e}")
        raise typer.Exit(code=1)



@app.command("native")
def run_native(
    script: str = typer.Argument(..., help="Python script to run"),
    experiment: str = typer.Option("default", "--experiment", "-e", help="Experiment name"),
):
    """
    Run training natively (no Docker) using local backends.
    
    Uses filesystem storage, SQLite tracking, and console output.
    Perfect for development and testing.
    """
    import subprocess
    import sys
    
    script_path = Path(script)
    if not script_path.exists():
        console.print(f"[bold red]❌ Script not found:[/bold red] {script}")
        raise typer.Exit(code=1)
    
    console.print(Panel.fit(
        f"[bold blue]🚀 Running natively (no Docker)[/bold blue]\n"
        f"Script: [cyan]{script}[/cyan]\n"
        f"Experiment: [cyan]{experiment}[/cyan]\n"
        f"Storage: [dim]./kladml_data[/dim]\n"
        f"Tracking: [dim]./mlruns/mlflow.db[/dim]"
    ))
    
    # Set environment variables for the script
    import os
    env = os.environ.copy()
    env["KLADML_EXPERIMENT"] = experiment
    
    # Run the script directly
    try:
        process = subprocess.Popen(
            [sys.executable, script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        
        for line in process.stdout:
            console.print(line, end="")
        
        process.wait()
        
        if process.returncode == 0:
            console.print("\n[bold green]✅ Run completed successfully.[/bold green]")
            console.print("[dim]Check ./mlruns for experiment tracking data.[/dim]")
        else:
            console.print(f"\n[bold red]❌ Run failed with code {process.returncode}[/bold red]")
            raise typer.Exit(code=process.returncode)
            
    except Exception as e:
        console.print(f"[bold red]❌ Error:[/bold red] {e}")
        raise typer.Exit(code=1)

