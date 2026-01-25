
from typing import Any
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score
from kladml.evaluation.base import BaseEvaluator

class RegressionEvaluator(BaseEvaluator):
    """
    Evaluator for Regression tasks.
    
    Metrics: MAE, MSE, RMSE, R2.
    Plots: Prediction vs Actual, Residuals.
    """
    
    def __init__(self, run_dir: Path, model_path: Path, data_path: Path, config: dict[str, Any] | None = None):
        super().__init__(run_dir, model_path, data_path, config)
        
        # Initialize metrics
        self.mae = MeanAbsoluteError()
        self.mse = MeanSquaredError()
        self.r2 = R2Score()

    def load_model(self) -> Any:
         if self.model_path.exists():
             try:
                return torch.jit.load(str(self.model_path))
             except:
                return torch.load(self.model_path)
         return None

    def load_data(self) -> Any:
         if self.data_path.suffix == '.pt':
             return torch.load(self.data_path)
         return None

    def inference(self, model: Any, data: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(data, (tuple, list)) and len(data) == 2:
            X, y = data
            model.eval()
            with torch.no_grad():
                preds = model(X)
            return preds, y
        raise  NotImplementedError("Complex data loading not yet implemented")

    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        """Compute metrics."""
        if not isinstance(predictions, torch.Tensor):
            predictions = torch.tensor(predictions)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets)
            
        # Ensure shapes match (flatten if needed or squeeze)
        if predictions.shape != targets.shape:
             predictions = predictions.view_as(targets)

        mse_val = self.mse(predictions, targets)
        
        results = {
            "mae": float(self.mae(predictions, targets)),
            "mse": float(mse_val),
            "rmse": float(torch.sqrt(mse_val)),
            "r2": float(self.r2(predictions, targets)),
        }
        return results

    def save_plots(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Generate regression plots."""
        if not isinstance(predictions, torch.Tensor):
            predictions = torch.tensor(predictions)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets)
            
        preds_np = predictions.cpu().numpy().flatten()
        targets_np = targets.cpu().numpy().flatten()
        
        # 1. Prediction vs Actual
        plt.figure(figsize=(8, 6))
        plt.scatter(targets_np, preds_np, alpha=0.5)
        
        # Ideal line
        min_val = min(targets_np.min(), preds_np.min())
        max_val = max(targets_np.max(), preds_np.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        
        plt.title('Prediction vs Actual')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        
        pva_path = self.plots_dir / "pred_vs_actual.png"
        plt.savefig(pva_path)
        plt.close()
        
        # 2. Residuals
        residuals = targets_np - preds_np
        plt.figure(figsize=(8, 6))
        plt.scatter(preds_np, residuals, alpha=0.5)
        plt.axhline(0, color='k', linestyle='--', lw=2)
        plt.title('Residual Plot')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        
        res_path = self.plots_dir / "residuals.png"
        plt.savefig(res_path)
        plt.close()
        
        self._logger.info(f"Saved plots: {pva_path}, {res_path}")

    def generate_report(self) -> str:
        """Generate MD report."""
        lines = [
            "# Regression Report",
            "",
            "## Metrics",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        for k, v in self.metrics.items():
            lines.append(f"| {k.upper()} | {v:.4f} |")
        
        lines.extend([
            "",
            "## Plots",
            "### Prediction vs Actual",
            "![Pred vs Actual](./plots/pred_vs_actual.png)",
            "",
            "### Residuals",
            "![Residuals](./plots/residuals.png)",
        ])
        
        return "\n".join(lines)
