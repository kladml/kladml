
import os
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
import mlflow
from kladml.backends import LocalTracker, get_metadata_backend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reconstruct")

def reconstruct_runs(base_dir: str = "data/projects"):
    """
    Scan the data directory and reconstruct missing runs in the MLflow DB.
    """
    tracker = LocalTracker()
    metadata = get_metadata_backend()
    base_path = Path(base_dir)

    if not base_path.exists():
        logger.error(f"Base directory {base_dir} does not exist.")
        return

    logger.info(f"Scanning {base_dir} for runs...")

    # Expected structure: projects/{project}/{family}/{experiment}/{run_id}
    # We walk and look for 'training.jsonl' as a marker of a valid run.
    
    for log_file in base_path.rglob("training.jsonl"):
        run_dir = log_file.parent
        run_id = run_dir.name
        
        # Path parts extraction
        # .../projects/sentinella/glucose_forecasting/foundation_v4/run_id
        try:
            # relative path from base_dir
            rel_path = run_dir.relative_to(base_path)
            parts = rel_path.parts
            
            if len(parts) != 4:
                logger.warning(f"Skipping {run_dir}: unexpected depth {len(parts)} (expected 4: proj/fam/exp/run)")
                continue
                
            project_name, family_name, experiment_name, _ = parts
        except Exception as e:
            logger.warning(f"Skipping {run_dir}: path parsing error {e}")
            continue

        logger.info(f"Found Run: {project_name}/{family_name}/{experiment_name}/{run_id}")

        # 1. Check if run exists in DB
        existing_run = tracker.get_run(run_id)
        # Note: get_run might fail if experiment logic differs, but let's try.
        # Actually proper way: check if mlflow.get_run(run_id) finds it.
        try:
            mlflow_run = mlflow.get_run(run_id)
            logger.info(f"  -> Run {run_id} already exists in DB. Skipping.")
            continue
        except Exception:
             # Run not found, proceed to import
             pass

        logger.info(f"  -> Importing run {run_id}...")

        # 2. Ensure Hierarchy Exists
        # Project
        if not metadata.get_project(project_name):
            try:
                metadata.create_project(project_name)
                logger.info(f"  -> Created Project '{project_name}'")
            except: pass # Race condition or exists
            
        # Family
        if not metadata.get_family(family_name, project_name):
            try:
                metadata.create_family(family_name, project_name)
                logger.info(f"  -> Created Family '{family_name}'")
            except: pass

        # Experiment (MLflow)
        # Ensure experiment is created/exists in MLflow
        tracker.create_experiment(experiment_name)
        
        # Link to Family (KladML Metadata)
        try:
            metadata.add_experiment_to_family(family_name, project_name, experiment_name)
        except Exception as e:
            # might already exist
            pass
        
        # 3. Load Data
        # Config (Params)
        params = {}
        config_path = run_dir / "config.yaml"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                    # Flatten config for logging? Or just log key top-levels
                    # Simple flattening 1-level
                    for k, v in config.items():
                        if isinstance(v, (dict, list)):
                            params[k] = str(v)
                        else:
                            params[k] = v
            except Exception as e:
                logger.warning(f"  -> Failed to read config: {e}")

        # Metrics (and config fallback)
        metrics_history = []
        try:
            with open(log_file) as f:
                for line in f:
                    data = json.loads(line)
                    metrics_history.append(data)
                    
                    # Fallback: Extraction config from first log line if needed
                    if not params and data.get("message") == "Training started":
                        run_config = data.get("data", {}).get("config", {})
                        if run_config:
                            logger.info(f"  -> Found config in training.jsonl fallback")
                            for k, v in run_config.items():
                                if isinstance(v, (dict, list)):
                                    params[k] = str(v)
                                else:
                                    params[k] = v
                                    
        except Exception as e:
            logger.warning(f"  -> Failed to read metrics/config from jsonl: {e}")

        # 4. Create Run in DB
        # We use explicit run_id
        try:
            # We must set correct experiment ID context
            exp = tracker.get_experiment_by_name(experiment_name)
            if not exp:
                logger.error(f"  -> Experiment {experiment_name} creation failed?")
                continue
                
            client = mlflow.tracking.MlflowClient(tracker._tracking_uri)
            
            # Create run
            # We allow MLflow to generate a UUID for the run_id (since we can't force it easily/safely).
            # We store the directory name 'run_001_...' as the run_name.
            
            with mlflow.start_run(experiment_id=exp["id"], run_name=run_id) as run:
                # Log original directory name as a tag for reference
                mlflow.set_tag("original_run_dir", run_id)
                
                # Log Params
                mlflow.log_params(params)
                
                # Log Metrics (replay history)
                # We prioritize "Epoch completed" logs for main metrics.
                step_count = 0
                for data_line in metrics_history:
                    # data_line structure: {timestamp, level, message, data: { ... }}
                    payload = data_line.get("data", {})
                    if not payload:
                        continue
                        
                    # If this is an Epoch Completion log, use it
                    # Check for train_loss/val_loss specifically or generic numbers
                    
                    # Determine step (Epoch or just generic counter)
                    current_step = payload.get("epoch", step_count)
                    
                    # Log numeric values
                    for key, val in payload.items():
                        if isinstance(val, (int, float)) and key != "epoch":
                            mlflow.log_metric(key, val, step=current_step)
                            
                    step_count += 1
                
                # Set tags for project/family
                mlflow.set_tag("project", project_name)
                mlflow.set_tag("family", family_name)
                
            logger.info(f"  -> Import successful.")
            
        except Exception as e:
            logger.error(f"  -> Failed to import run: {e}")

    logger.info("Reconstruction complete.")

if __name__ == "__main__":
    reconstruct_runs()
