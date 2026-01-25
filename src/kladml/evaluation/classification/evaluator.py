
from typing import Any
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix, AUROC
from kladml.evaluation.base import BaseEvaluator

class ClassificationEvaluator(BaseEvaluator):
    """
    Evaluator for Classification tasks.
    
    Metrics: Accuracy, Precision, Recall, F1, AUROC.
    Plots: Confusion Matrix, ROC Curve (if probs available).
    """
    
    def __init__(self, run_dir: Path, model_path: Path, data_path: Path, config: dict[str, Any] | None = None):
        super().__init__(run_dir, model_path, data_path, config)
        self.num_classes = self.config.get("num_classes", 2)
        self.task_type = "binary" if self.num_classes == 2 else "multiclass"
        
        # Initialize metrics
        self.acc = Accuracy(task=self.task_type, num_classes=self.num_classes)
        self.prec = Precision(task=self.task_type, num_classes=self.num_classes, average='macro')
        self.rec = Recall(task=self.task_type, num_classes=self.num_classes, average='macro')
        self.f1 = F1Score(task=self.task_type, num_classes=self.num_classes, average='macro')
        self.confmat = ConfusionMatrix(task=self.task_type, num_classes=self.num_classes)
        
        # AUROC usually requires probabilities
        self.auroc = AUROC(task=self.task_type, num_classes=self.num_classes) if self.config.get("compute_auroc", True) else None

    def load_model(self) -> Any:
        # User defined or generic loading
        # For now, we assume the model object is passed directly OR loaded via torch.load if it's a file
        if self.model_path.exists():
            # Minimal logic: try loading JIT or Pickle
             try:
                return torch.jit.load(str(self.model_path))
             except:
                return torch.load(self.model_path)
        return None

    def load_data(self) -> Any:
         # Placeholder: In a real scenario, this loads a DataLoader
         # For this implementation, we assume 'data_path' might point to a .pt file with (X, y)
         if self.data_path.suffix == '.pt':
             return torch.load(self.data_path)
         return None

    def inference(self, model: Any, data: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference. 
        Expects data to be a tuple (X, y) or a DataLoader.
        Returns (preds, targets).
        """
        # Simplification for the example: assume data is (X, y) tensor tuple
        if isinstance(data, (tuple, list)) and len(data) == 2:
            X, y = data
            model.eval()
            with torch.no_grad():
                preds = model(X)
            return preds, y
        raise  NotImplementedError("Complex data loading not yet implemented in baseline evaluator")

    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        """Compute metrics using torchmetrics."""
        # Ensure primitive types
        if not isinstance(predictions, torch.Tensor):
            predictions = torch.tensor(predictions)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets)

        # If predictions are logits/probs, get labels for some metrics
        if predictions.is_floating_point():
             preds_labels = predictions.argmax(dim=1) if self.num_classes > 2 else (predictions > 0.5).long()
        else:
            preds_labels = predictions

        results = {
            "accuracy": float(self.acc(preds_labels, targets)),
            "precision": float(self.prec(preds_labels, targets)),
            "recall": float(self.rec(preds_labels, targets)),
            "f1": float(self.f1(preds_labels, targets)),
        }
        
        if self.auroc and predictions.is_floating_point():
             results["auroc"] = float(self.auroc(predictions, targets))
             
        return results

    def save_plots(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Generate confusion matrix plot."""
         # Ensure primitive types
        if not isinstance(predictions, torch.Tensor):
            predictions = torch.tensor(predictions)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets)

        if predictions.is_floating_point():
             preds_labels = predictions.argmax(dim=1) if self.num_classes > 2 else (predictions > 0.5).long()
        else:
            preds_labels = predictions
            
        # 1. Confusion Matrix
        cm = self.confmat(preds_labels, targets).cpu().numpy()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        cm_path = self.plots_dir / "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        
        self._logger.info(f"Saved plot: {cm_path}")

    def generate_report(self) -> str:
        """Generate MD report."""
        lines = [
            "# Classification Report",
            "",
            "## Metrics",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        for k, v in self.metrics.items():
            lines.append(f"| {k.capitalize()} | {v:.4f} |")
        
        lines.extend([
            "",
            "## Plots",
            "### Confusion Matrix",
            "![Confusion Matrix](./plots/confusion_matrix.png)",
        ])
        
        return "\n".join(lines)
