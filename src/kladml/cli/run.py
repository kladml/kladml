"""
KladML CLI - Run Commands
"""

import typer
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

# NOTE: Heavy imports (interfaces, backends) are done inside functions
# for faster CLI startup time

app = typer.Typer()
console = Console()


@app.command("local")
def run_local(
    script: str = typer.Argument(..., help="Python script to run"),
    device: str = typer.Option("auto", "--device", "-d", help="Device: auto|cpu|cuda|mps"),
    runtime: str = typer.Option("auto", "--runtime", "-r", help="Container runtime: auto|docker|podman"),
    image: str = typer.Option(None, "--image", "-i", help="Custom Docker image to use"),
    experiment: str = typer.Option(None, "--experiment", "-e", help="Associated experiment name"),
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
        
    # Validation via Tracker
    if experiment:
        if not tracker.get_experiment_by_name(experiment):
             # Just a warning, container might create it
             console.print(f"[yellow]Note: Experiment '{experiment}' does not exist yet locally.[/yellow]")

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
            "mps": "ghcr.io/kladml/worker:cpu", 
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
        "--network", "host", # Allow access to local MLflow/services
    ]
    
    # Add GPU support
    if device.startswith("cuda"):
        if runtime_cmd == "docker":
            cmd.extend(["--gpus", "all"])
        elif runtime_cmd == "podman":
            cmd.extend(["--device", "nvidia.com/gpu=all"]) 
            cmd.extend(["--security-opt=label=disable"]) # Often needed for Podman GPU
    
    # Add environment variables
    from kladml.backends.local_config import YamlConfig
    config = YamlConfig()
    
    env_vars = {
        "KLADML_PROJECT_NAME": config.get("project.name", "unknown"),
        "KLADML_TRAINING_DEVICE": device,
    }
    
    if experiment:
        env_vars["KLADML_EXPERIMENT"] = experiment

    for k, v in env_vars.items():
        if v:
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
            if process.returncode == 126 and device.startswith("cuda") and runtime_cmd == "podman":
                console.print(Panel(
                    "Podman GPU Error? Try: sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml",
                    title="Setup Hint",
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
    Run training natively (no Docker).
    """
    import subprocess
    import sys
    import os
    
    script_path = Path(script)
    if not script_path.exists():
        console.print(f"[bold red]❌ Script not found:[/bold red] {script}")
        raise typer.Exit(code=1)
    
    # Ensure experiment exists via Tracker
    exp_id = tracker.create_experiment(experiment)
    
    console.print(Panel.fit(
        f"[bold blue]🚀 Running natively[/bold blue]\n"
        f"Script: [cyan]{script}[/cyan]\n"
        f"Experiment: [cyan]{experiment}[/cyan] (ID: {exp_id})\n"
        f"Tracking: [dim]SQLite[/dim]"
    ))
    
    env = os.environ.copy()
    env["KLADML_EXPERIMENT"] = experiment
    
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
        else:
            console.print(f"\n[bold red]❌ Run failed with code {process.returncode}[/bold red]")
            raise typer.Exit(code=process.returncode)
            
    except Exception as e:
        console.print(f"[bold red]❌ Error:[/bold red] {e}")
        raise typer.Exit(code=1)
