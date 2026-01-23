"""
Run ID Generator for KladML SDK.

Generates sequential + timestamp run IDs:
- run_001_20260115_2317
- run_002_20260115_2345
"""

import os
import re
from pathlib import Path
from datetime import datetime
from typing import Optional


def generate_run_id(
    project_name: str,
    experiment_name: str,
    family_name: Optional[str] = None,
    base_dir: str = "data/projects",
) -> str:
    """
    Generate a new run ID with format: run_XXX_YYYYMMDD_HHMM
    
    Args:
        project_name: Project name
        experiment_name: Experiment name
        family_name: Optional family/domain name (e.g., "canbus_anomaly")
        base_dir: Base directory for projects
        
    Returns:
        Run ID string like "run_001_20260115_2317"
    """
    # Build path: base/project/[family/]experiment
    if family_name:
        log_dir = Path(base_dir) / project_name / family_name / experiment_name
    else:
        log_dir = Path(base_dir) / project_name / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    
    # Find existing runs (directories starting with run_)
    existing_run_dirs = [p for p in log_dir.glob("run_*") if p.is_dir()]
    
    # Extract run numbers
    run_numbers = []
    pattern = re.compile(r"run_(\d+)_")
    for path in existing_run_dirs:
        match = pattern.search(path.name)
        if match:
            run_numbers.append(int(match.group(1)))
    
    # Next run number
    next_num = max(run_numbers, default=0) + 1
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    return f"run_{next_num:03d}_{timestamp}"


def get_run_checkpoint_dir(
    project_name: str,
    experiment_name: str,
    run_id: str,
    base_dir: str = "./models",
) -> Path:
    """
    Get the checkpoint directory for a specific run.
    
    Directory structure:
        models/<project>_<experiment>/<run_id>/
    
    Args:
        project_name: Project name
        experiment_name: Experiment name
        run_id: Run identifier
        base_dir: Base directory for models
        
    Returns:
        Path to run checkpoint directory
    """
    run_dir = Path(base_dir) / f"{project_name}_{experiment_name}" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
