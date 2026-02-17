
import typer
from pathlib import Path
from rich.console import Console
from typing import Optional

from kladml.evaluation.registry import EvaluatorRegistry
from kladml.tasks import MLTask
from kladml.utils.loading import detect_ml_task

app = typer.Typer(help="Run evaluations.")
console = Console()

@app.command("run")
def run_evaluation(
    run_id: str = typer.Option(..., "--run-id", "-r", help="Run ID to evaluate"),
    model_path: str = typer.Option(..., "--model", "-m", help="Path to model file"),
    data_path: str = typer.Option(..., "--data", "-d", help="Path to test data"),
    task: Optional[str] = typer.Option(None, "--task", "-t", help="Override ML Task (classification, regression)"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
) -> None:
    """
    Run evaluation on a model.
    """
    try:
        # 1. Determine Output Dir
        if output_dir:
            out_path = Path(output_dir)
        else:
            # Default to runs/<run_id>/evaluation
            # But we don't know where runs are without querying the DB/Tracker or assuming folder structure.
            # providing an explicit output_dir for now is safer if stand-alone.
            out_path = Path(f"./evaluations/{run_id}")
        
        out_path.mkdir(parents=True, exist_ok=True)
            
        # 2. Determine Task (Auto-detect or Override)
        # In a real scenario, we'd load the model config to find the task.
        # For now, we require --task if we can't load the model class to check.
        # But let's assume the user provides it or we default to Classification if uncertain?
        # Better: Try to deduce from EvaluatorRegistry if not provided? No, we need task to PICK evaluator.
        
        if not task:
            console.print("[yellow]Task not specified. Trying to detect...[/yellow]")

            # Try to detect task from model/run metadata
            detected_task = detect_ml_task(
                model_path=model_path,
                run_dir=str(Path(model_path).parent) if model_path else None,
            )

            if detected_task:
                console.print(f"[green]Detected task:[/green] [cyan]{detected_task.value}[/cyan]")
                task_enum = detected_task
            else:
                # Fallback to classification with warning
                console.print("[yellow]Could not auto-detect task. Defaulting to classification.[/yellow]")
                console.print("[dim]Tip: Specify --task for accurate evaluation.[/dim]")
                task_enum = MLTask.CLASSIFICATION
        else:
            try:
                task_enum = MLTask(task.lower())
            except ValueError:
                console.print(f"[red]Invalid task: {task}[/red]")
                raise typer.Exit(1)
        
        # 3. Get Evaluator Class
        try:
            evaluator_cls = EvaluatorRegistry.get_evaluator(task_enum)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)
            
        console.print(f"Using Evaluator: [green]{evaluator_cls.__name__}[/green]")
        
        # 4. Instantiate and Run
        evaluator = evaluator_cls(
            run_dir=out_path,
            model_path=Path(model_path),
            data_path=Path(data_path),
            config={"compute_auroc": True} # Default config
        )
        
        metrics = evaluator.run()
        
        console.print(f"\n[bold green]Evaluation Complete![/bold green]")
        console.print(f"Report saved to: {out_path}/evaluation_report.md")
        
    except Exception as e:
        console.print(f"[red]Evaluation failed:[/red] {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)
