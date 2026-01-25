
import sys
import os
from rich.console import Console

console = Console()

def relaunch_with_accelerate(num_processes: int = 1):
    """
    Relaunch the current command using 'accelerate launch'.
    Removes the --distributed/--num-processes flags to avoid infinite loop.
    """
    # 1. Filter out distributed flags
    new_args = []
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
            
        if arg in ("--distributed",):
            continue
        if arg in ("--num-processes",):
            skip_next = True # Skip value
            continue
            
        new_args.append(arg)

    # 2. Construct launch command
    # accelerate launch --num_processes X -m kladml.cli.main ...
    
    cmd = [
        "accelerate", "launch",
        "--num_processes", str(num_processes),
        "-m", "kladml.cli.main"
    ] + new_args
    
    console.print(f"[bold yellow]🚀 Relaunching via Accelerate (Distributed: {num_processes} processes)...[/bold yellow]")
    console.print(f"[dim]Command: {' '.join(cmd)}[/dim]\n")
    
    # 3. Replace process
    os.execvp("accelerate", cmd)
