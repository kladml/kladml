"""
Time Series Forecasting Evaluator.

Evaluator for time series forecasting tasks.
Metrics: MAE, MSE, RMSE, MAPE, R2.
Plots: Prediction vs Actual (line plot), Residuals over time, Error distribution.
"""

from typing import Any
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score
from kladml.evaluation.base import BaseEvaluator


class TimeSeriesEvaluator(BaseEvaluator):
    """
    Evaluator for Time Series Forecasting tasks.

    Metrics: MAE, MSE, RMSE, MAPE, R2.
    Plots: Prediction vs Actual, Residuals over time, Error distribution.
    """

    def __init__(
        self,
        run_dir: Path,
        model_path: Path,
        data_path: Path,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(run_dir, model_path, data_path, config)

        # Horizon (prediction length)
        self.horizon = self.config.get("horizon", 1)

        # Initialize metrics
        self.mae = MeanAbsoluteError()
        self.mse = MeanSquaredError()
        self.r2 = R2Score()

    def load_model(self) -> Any:
        """Load the model from checkpoint."""
        if self.model_path.exists():
            try:
                return torch.jit.load(str(self.model_path))
            except Exception:
                return torch.load(self.model_path, weights_only=False)
        return None

    def load_data(self) -> Any:
        """Load evaluation dataset."""
        if self.data_path.suffix == ".pt":
            return torch.load(self.data_path, weights_only=False)
        return None

    def inference(self, model: Any, data: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference on time series data.

        Args:
            model: The loaded model.
            data: The loaded data, expected as (X, y) tuple.

        Returns:
            Tuple of (predictions, targets).
        """
        if isinstance(data, (tuple, list)) and len(data) == 2:
            X, y = data
            model.eval()
            with torch.no_grad():
                preds = model(X)
            return preds, y
        raise NotImplementedError(
            "Complex data loading not yet implemented in TimeSeriesEvaluator"
        )

    def compute_metrics(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, float]:
        """Compute time series forecasting metrics."""
        if not isinstance(predictions, torch.Tensor):
            predictions = torch.tensor(predictions)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets)

        # Ensure shapes match
        if predictions.shape != targets.shape:
            predictions = predictions.view_as(targets)

        # Flatten for metric computation
        preds_flat = predictions.flatten()
        targets_flat = targets.flatten()

        mse_val = self.mse(preds_flat, targets_flat)

        # Compute MAPE (avoid division by zero)
        epsilon = 1e-8
        mape = torch.mean(torch.abs((targets_flat - preds_flat) / (targets_flat.abs() + epsilon))) * 100

        results = {
            "mae": float(self.mae(preds_flat, targets_flat)),
            "mse": float(mse_val),
            "rmse": float(torch.sqrt(mse_val)),
            "mape": float(mape),
            "r2": float(self.r2(preds_flat, targets_flat)),
        }
        return results

    def save_plots(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> None:
        """Generate time series evaluation plots."""
        if not isinstance(predictions, torch.Tensor):
            predictions = torch.tensor(predictions)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets)

        preds_np = predictions.cpu().numpy().flatten()
        targets_np = targets.cpu().numpy().flatten()

        # 1. Prediction vs Actual (Line Plot)
        plt.figure(figsize=(12, 5))
        plt.plot(targets_np, label="Actual", alpha=0.7)
        plt.plot(preds_np, label="Predicted", alpha=0.7)
        plt.title("Time Series: Prediction vs Actual")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()

        line_path = self.plots_dir / "pred_vs_actual_line.png"
        plt.savefig(line_path)
        plt.close()

        # 2. Scatter Plot: Prediction vs Actual
        plt.figure(figsize=(8, 6))
        plt.scatter(targets_np, preds_np, alpha=0.5)

        min_val = min(targets_np.min(), preds_np.min())
        max_val = max(targets_np.max(), preds_np.max())
        plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=2)

        plt.title("Prediction vs Actual (Scatter)")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")

        scatter_path = self.plots_dir / "pred_vs_actual_scatter.png"
        plt.savefig(scatter_path)
        plt.close()

        # 3. Residuals over time
        residuals = targets_np - preds_np
        plt.figure(figsize=(12, 4))
        plt.plot(residuals, alpha=0.7)
        plt.axhline(0, color="k", linestyle="--", lw=2)
        plt.title("Residuals Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Residual (Actual - Predicted)")

        res_path = self.plots_dir / "residuals_over_time.png"
        plt.savefig(res_path)
        plt.close()

        # 4. Error Distribution (Histogram)
        plt.figure(figsize=(8, 5))
        plt.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
        plt.axvline(0, color="red", linestyle="--", lw=2)
        plt.title("Error Distribution")
        plt.xlabel("Residual")
        plt.ylabel("Frequency")

        hist_path = self.plots_dir / "error_distribution.png"
        plt.savefig(hist_path)
        plt.close()

        self._logger.info(
            f"Saved plots: {line_path}, {scatter_path}, {res_path}, {hist_path}"
        )

    def generate_report(self) -> str:
        """Generate Markdown report for time series evaluation."""
        lines = [
            "# Time Series Forecasting Report",
            "",
            "## Metrics",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        for k, v in self.metrics.items():
            lines.append(f"| {k.upper()} | {v:.4f} |")

        lines.extend(
            [
                "",
                "## Plots",
                "### Prediction vs Actual (Line)",
                "![Prediction vs Actual Line](./plots/pred_vs_actual_line.png)",
                "",
                "### Prediction vs Actual (Scatter)",
                "![Prediction vs Actual Scatter](./plots/pred_vs_actual_scatter.png)",
                "",
                "### Residuals Over Time",
                "![Residuals Over Time](./plots/residuals_over_time.png)",
                "",
                "### Error Distribution",
                "![Error Distribution](./plots/error_distribution.png)",
            ]
        )

        return "\n".join(lines)
