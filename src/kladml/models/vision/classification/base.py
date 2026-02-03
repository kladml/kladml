from typing import Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from torchmetrics import Accuracy, F1Score, Precision, Recall
from kladml.models.vision.base import ImageModel
from kladml.tasks import MLTask
from kladml.training.trainer import UniversalTrainer
from kladml.training.callbacks import Callback
from kladml.training.checkpoint import CheckpointManager

# --- Parity Helpers ---
def set_seed(seed):
    """Orchid parity seeding."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # Required for deterministic algos
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


class OrchidCheckpointCallback(Callback):
    """
    Parity Callback: Saves model based on F1 Score (maximizing).
    Matches Orchid logic: if actualF1Score >= lastF1Score -> save.
    """
    def __init__(self, manager: CheckpointManager):
        self.manager = manager
        self.trainer = None
        self.manager._lower_is_better = False 
        
    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        if not self.trainer:
            return
        
        # Logs now contain the true computed metrics from ImageClassifier.on_epoch_end
        self.manager.save_checkpoint(
            model=self.trainer.model,
            optimizer=self.trainer.optimizer,
            epoch=epoch,
            metrics=logs or {},
            comparison_metric="val_f1",
            scheduler=getattr(self.trainer, 'lr_scheduler_config', {}).get('scheduler') if getattr(self.trainer, 'lr_scheduler_config', None) else None
        )

class ImageClassifier(ImageModel):
    """
    Generic Image Classifier.
    Implements standard training logic for classification tasks.
    """
    
    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__(config)
        self.num_classes = self.config.get("num_classes", 1000)
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Metrics - Macro average as in Orchid
        # Note: We do NOT rely on UniversalTrainer averaging these. 
        # We accumulate state and compute at epoch end.
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes, average="macro")
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        self.val_prec = Precision(task="multiclass", num_classes=self.num_classes, average="macro")
        self.val_recall = Recall(task="multiclass", num_classes=self.num_classes, average="macro")
        
    @property
    def ml_task(self) -> MLTask:
        return MLTask.IMAGE_CLASSIFICATION
        
    def train(self, X_train: Any, y_train: Any = None, X_val: Any = None, y_val: Any = None, **kwargs) -> dict[str, float]:
        """
        Train using UniversalTrainer.
        """
        # 1. Set Seed (Parity)
        seed = self.config.get("seed", 42)
        set_seed(seed)
        
        # 2. Setup Checkpoint Manager (Parity)
        run_id = kwargs.get("run_id", "default_run")
        project_name = self.config.get("project_name", "orchid_parity")
        experiment_name = self.config.get("experiment_name", "resnet18")
        
        checkpoint_manager = CheckpointManager(
            project_name=project_name,
            experiment_name=experiment_name,
            run_id=run_id
        )
        
        parity_callback = OrchidCheckpointCallback(checkpoint_manager)
        callbacks = [parity_callback]
        
        trainer = UniversalTrainer(config=self.config, callbacks=callbacks)
        return trainer.fit(
            model=self,
            train_dataloaders=X_train,
            val_dataloaders=X_val
        )

    def forward(self, x):
        raise NotImplementedError

    def prediction_step(self, batch: Any, batch_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = batch
        logits = self(x)
        return logits, y

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        logits, y = self.prediction_step(batch, batch_idx)
        loss = self.loss_fn(logits, y)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> dict[str, torch.Tensor]:
        logits, y = self.prediction_step(batch, batch_idx)
        loss = self.loss_fn(logits, y)
        
        # Update metric state (predictions are accumulated internally)
        preds = torch.argmax(logits, dim=1)
        self.val_acc.update(preds, y)
        self.val_f1.update(preds, y)
        self.val_prec.update(preds, y)
        self.val_recall.update(preds, y)
        
        # Return only loss for Trainer to average
        return {"val_loss": loss}

    def on_epoch_end(self, epoch, logs=None):
        """Compute epoch-level metrics and inject into logs."""
        if logs is None:
            logs = {}
            
        # Compute metrics on full validation set
        # Move to cpu for logging
        acc = self.val_acc.compute().item()
        f1 = self.val_f1.compute().item()
        prec = self.val_prec.compute().item()
        rec = self.val_recall.compute().item()
        
        # Update logs (in-place modification so callbacks see it)
        logs.update({
            "val_acc": acc,
            "val_f1": f1,
            "val_prec": prec,
            "val_recall": rec
        })
        
        # Reset for next epoch
        self.val_acc.reset()
        self.val_f1.reset()
        self.val_prec.reset()
        self.val_recall.reset()
        
        # Call super (BaseModel hooks)
        super().on_epoch_end(epoch, logs)

    def configure_optimizers(self):
        """Configure optimizers based on config."""
        lr = self.config.get("learning_rate", 1e-3)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.1, 
            patience=self.config.get("patience", 3)
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

    def predict(self, X: Any, **kwargs) -> Any:
        self.eval()
        with torch.no_grad():
            if isinstance(X, torch.Tensor):
                return self(X)
            return self(torch.tensor(X))

    def evaluate(self, X_test: Any, y_test: Any = None, **kwargs) -> dict[str, float]:
        return {} 
            
    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)
        
    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
