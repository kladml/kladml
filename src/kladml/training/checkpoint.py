"""
Checkpoint Manager for KladML SDK.

Handles model checkpoint saving and loading with structured directory layout:
- models/<project>_<experiment>/checkpoint_epoch_<N>.pth
- models/<project>_<experiment>/best_model.pth
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages model checkpoints with structured directory layout.
    
    Directory structure:
        models/
        └── <project>_<experiment>/
            ├── best_model.pth
            ├── checkpoint_epoch_5.pth
            ├── checkpoint_epoch_10.pth
            └── metadata.json
    
    Example:
        >>> manager = CheckpointManager("sentinella", "gluformer_v1")
        >>> manager.save_checkpoint(model, optimizer, epoch=5, metrics={"val_loss": 0.1})
        >>> model, optimizer, epoch, metrics = manager.load_checkpoint("best")
    """
    
    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        run_id: Optional[str] = None,
        base_dir: str = "./data/projects",
        checkpoint_frequency: int = 5,
        family_name: Optional[str] = None,
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            project_name: Name of the project
            experiment_name: Name of the experiment
            run_id: Run identifier (if provided, creates per-run directory)
            base_dir: Base directory for projects (default: ./data/projects)
            checkpoint_frequency: Save checkpoint every N epochs
            family_name: Optional family name for hierarchy
        """
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.run_id = run_id
        self.base_dir = Path(base_dir)
        self.checkpoint_frequency = checkpoint_frequency
        self.family_name = family_name
        
        # Create unified run directory structure
        # data/projects/<project>/[<family>/]<experiment>/<run_id>/checkpoints/
        path_parts = [project_name]
        if family_name:
            path_parts.append(family_name)
        path_parts.append(experiment_name)
        
        if run_id:
            path_parts.append(run_id)
            
        self.checkpoint_dir = self.base_dir.joinpath(*path_parts) / "checkpoints"
            
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Track best metric
        self._best_metric: Optional[float] = None
        self._best_epoch: Optional[int] = None
        self._lower_is_better: bool = True
        
        # Metadata file
        self._metadata_path = self.checkpoint_dir / "metadata.json"
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load existing metadata if present."""
        if self._metadata_path.exists():
            try:
                with open(self._metadata_path) as f:
                    meta = json.load(f)
                    self._best_metric = meta.get("best_metric")
                    self._best_epoch = meta.get("best_epoch")
                    logger.info(f"Loaded metadata: best_epoch={self._best_epoch}, best_metric={self._best_metric}")
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
    
    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        meta = {
            "project": self.project_name,
            "experiment": self.experiment_name,
            "best_metric": self._best_metric,
            "best_epoch": self._best_epoch,
            "updated_at": datetime.now().isoformat(),
        }
        with open(self._metadata_path, "w") as f:
            json.dump(meta, f, indent=2)
    
    def save_checkpoint(
        self,
        model: Any,
        optimizer: Any,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        comparison_metric: str = "val_loss",
        scaler: Any = None,
        scheduler: Any = None,
        config: Optional[Dict] = None,
        save_random_states: bool = True,
    ) -> Optional[str]:
        """
        Save a checkpoint with full training state for pause/resume.
        
        Args:
            model: PyTorch model (or any object with state_dict())
            optimizer: Optimizer (or any object with state_dict())
            epoch: Current epoch number
            metrics: Dictionary of metrics
            is_best: Force save as best model
            comparison_metric: Metric to use for best model comparison
            scaler: Optional sklearn scaler for data normalization
            scheduler: Optional learning rate scheduler
            config: Optional training configuration dict
            save_random_states: Whether to save RNG states for reproducibility
            
        Returns:
            Path to saved checkpoint, or None if skipped
        """
        try:
            import torch
        except ImportError:
            logger.error("PyTorch required for checkpoint saving")
            return None
        
        # Check if we should save (based on frequency)
        should_save_periodic = (epoch % self.checkpoint_frequency == 0)
        
        # Check if this is the best model
        current_metric = metrics.get(comparison_metric)
        if current_metric is not None:
            if self._best_metric is None:
                is_best = True
            elif self._lower_is_better and current_metric < self._best_metric:
                is_best = True
            elif not self._lower_is_better and current_metric > self._best_metric:
                is_best = True
        
        if not should_save_periodic and not is_best:
            return None
        
        # Prepare checkpoint data
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "scaler": scaler,
            "config": config,
        }
        
        # Save random states for reproducibility
        if save_random_states:
            import random
            import numpy as np
            checkpoint["random_states"] = {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            }
            if torch.cuda.is_available():
                checkpoint["random_states"]["cuda"] = torch.cuda.get_rng_state_all()
        
        saved_path = None
        
        # Save periodic checkpoint
        if should_save_periodic:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, path)
            logger.info(f"Saved checkpoint: {path}")
            saved_path = str(path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self._best_metric = current_metric
            self._best_epoch = epoch
            self._save_metadata()
            logger.info(f"Saved best model (epoch {epoch}, {comparison_metric}={current_metric})")
            saved_path = str(best_path)
        
        return saved_path
    
    def load_checkpoint(
        self,
        checkpoint_type: str = "best",
        epoch: Optional[int] = None,
        model: Any = None,
        optimizer: Any = None,
        scheduler: Any = None,
        device: str = "cpu",
        restore_random_states: bool = False,
    ) -> Tuple[int, Dict[str, float], Optional[Dict]]:
        """
        Load a checkpoint with full training state for resume.
        
        Args:
            checkpoint_type: "best", "latest", or "epoch"
            epoch: Specific epoch to load (when checkpoint_type="epoch")
            model: Model to load weights into
            optimizer: Optimizer to load state into
            scheduler: Learning rate scheduler to load state into
            device: Device to load tensors to
            restore_random_states: Whether to restore RNG states
            
        Returns:
            Tuple of (epoch, metrics, config)
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required for checkpoint loading")
        
        # Determine checkpoint path
        if checkpoint_type == "best":
            path = self.checkpoint_dir / "best_model.pt"
        elif checkpoint_type == "latest":
            # Find latest checkpoint by epoch number
            checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
            if not checkpoints:
                raise FileNotFoundError("No checkpoints found")
            path = max(checkpoints, key=lambda p: int(p.stem.split("_")[-1]))
        elif checkpoint_type == "epoch" and epoch is not None:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        else:
            raise ValueError(f"Invalid checkpoint_type: {checkpoint_type}")
        
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        # Load checkpoint
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        if model is not None:
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded model weights from {path}")
        
        if optimizer is not None and checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info(f"Loaded optimizer state from {path}")
        
        if scheduler is not None and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logger.info(f"Loaded scheduler state from {path}")
        
        # Restore random states for reproducibility
        if restore_random_states and checkpoint.get("random_states"):
            import random
            import numpy as np
            states = checkpoint["random_states"]
            if "python" in states:
                random.setstate(states["python"])
            if "numpy" in states:
                np.random.set_state(states["numpy"])
            if "torch" in states:
                torch.set_rng_state(states["torch"])
            if "cuda" in states and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(states["cuda"])
            logger.info("Restored random states for reproducibility")
        
        return checkpoint["epoch"], checkpoint.get("metrics", {}), checkpoint.get("config")
    
    def list_checkpoints(self) -> list:
        """List all available checkpoints."""
        checkpoints = []
        
        # Best model
        best_path = self.checkpoint_dir / "best_model.pt"
        if best_path.exists():
            checkpoints.append({
                "type": "best",
                "path": str(best_path),
                "epoch": self._best_epoch,
                "metric": self._best_metric,
            })
        
        # Periodic checkpoints
        for path in sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt")):
            epoch = int(path.stem.split("_")[-1])
            checkpoints.append({
                "type": "periodic",
                "path": str(path),
                "epoch": epoch,
            })
        
        return checkpoints
    
    @property
    def best_epoch(self) -> Optional[int]:
        """Get the epoch of the best model."""
        return self._best_epoch
    
    @property
    def best_metric(self) -> Optional[float]:
        """Get the best metric value."""
        return self._best_metric
