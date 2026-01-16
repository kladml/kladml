"""
Time Series Evaluator for KladML.

Base evaluator for all time series forecasting models.
Provides standard metrics (MAE, RMSE, MAPE) and plots.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List
from datetime import datetime
import numpy as np

from .base import BaseEvaluator
from .plots import create_figure, save_figure


class TimeSeriesEvaluator(BaseEvaluator):
    """
    Evaluator for time series forecasting models.
    
    Provides:
    - Standard metrics: MAE, RMSE, MAPE
    - Plots: Predictions vs Actual, Error Distribution
    """
    
    def compute_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Compute standard time series forecasting metrics.
        
        Args:
            predictions: Predicted values [N, Horizon] or [N, Horizon, 1].
            targets: Ground truth values with same shape.
            
        Returns:
            Dictionary with MAE, RMSE, MAPE.
        """
        # Flatten if needed
        preds = np.array(predictions).flatten()
        targs = np.array(targets).flatten()
        
        # Compute errors
        errors = preds - targs
        abs_errors = np.abs(errors)
        
        # MAE
        mae = np.mean(abs_errors)
        
        # RMSE
        rmse = np.sqrt(np.mean(errors ** 2))
        
        # MAPE (avoid division by zero)
        mask = np.abs(targs) > 1e-8
        if np.sum(mask) > 0:
            mape = np.mean(np.abs(errors[mask] / targs[mask])) * 100
        else:
            mape = np.nan
        
        # Additional metrics
        std_error = np.std(errors)
        max_error = np.max(abs_errors)
        
        return {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "MAPE": float(mape),
            "Std_Error": float(std_error),
            "Max_Error": float(max_error),
        }
    
    def save_plots(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """
        Generate and save standard time series evaluation plots.
        
        Args:
            predictions: Predicted values.
            targets: Ground truth values.
        """
        self._plot_predictions_sample(predictions, targets)
        self._plot_error_distribution(predictions, targets)
        self._plot_scatter(predictions, targets)
    
    def _plot_predictions_sample(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray,
        num_samples: int = 5
    ) -> None:
        """Plot sample predictions vs actual values."""
        fig, axes = create_figure(nrows=num_samples, ncols=1, figsize=(10, 3 * num_samples))
        
        # Handle single axis case
        if num_samples == 1:
            axes = [axes]
        
        preds = np.array(predictions)
        targs = np.array(targets)
        
        # If shape is [N, Horizon, 1], squeeze
        if preds.ndim == 3:
            preds = preds.squeeze(-1)
            targs = targs.squeeze(-1)
        
        n_samples = min(num_samples, len(preds))
        indices = np.linspace(0, len(preds) - 1, n_samples, dtype=int)
        
        for i, idx in enumerate(indices):
            ax = axes[i]
            horizon = np.arange(len(preds[idx]))
            
            ax.plot(horizon, targs[idx], "o-", label="Actual", color="#2ecc71", markersize=4)
            ax.plot(horizon, preds[idx], "s--", label="Predicted", color="#3498db", markersize=4)
            
            ax.set_xlabel("Horizon Step")
            ax.set_ylabel("Value")
            ax.set_title(f"Sample {idx}")
            ax.legend(loc="upper right")
        
        fig.suptitle("Sample Predictions vs Actual", fontsize=14)
        fig.tight_layout()
        save_figure(fig, self.plots_dir, "predictions_sample")
    
    def _plot_error_distribution(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray
    ) -> None:
        """Plot error distribution histogram."""
        fig, ax = create_figure()
        
        errors = np.array(predictions).flatten() - np.array(targets).flatten()
        
        ax.hist(errors, bins=50, color="#3498db", edgecolor="white", alpha=0.8)
        ax.axvline(0, color="#e74c3c", linestyle="--", linewidth=2, label="Zero Error")
        ax.axvline(np.mean(errors), color="#2ecc71", linestyle="-", linewidth=2, 
                   label=f"Mean: {np.mean(errors):.2f}")
        
        ax.set_xlabel("Prediction Error")
        ax.set_ylabel("Frequency")
        ax.set_title("Error Distribution")
        ax.legend()
        
        save_figure(fig, self.plots_dir, "error_distribution")
    
    def _plot_scatter(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray
    ) -> None:
        """Plot scatter of predicted vs actual."""
        fig, ax = create_figure()
        
        preds = np.array(predictions).flatten()
        targs = np.array(targets).flatten()
        
        ax.scatter(targs, preds, alpha=0.3, s=10, color="#3498db")
        
        # Perfect prediction line
        min_val = min(targs.min(), preds.min())
        max_val = max(targs.max(), preds.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")
        
        ax.set_xlabel("Actual Value")
        ax.set_ylabel("Predicted Value")
        ax.set_title("Predicted vs Actual")
        ax.legend()
        ax.set_aspect("equal", adjustable="box")
        
        save_figure(fig, self.plots_dir, "scatter_pred_vs_actual")
    
    def generate_report(self) -> str:
        """
        Generate Markdown evaluation report.
        
        Returns:
            Markdown formatted report string.
        """
        duration = (self._end_time - self._start_time).total_seconds() if self._end_time else 0
        
        report = f"""# Evaluation Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Model**: `{self.model_path.name}`  
**Data**: `{self.data_path.name}`  
**Duration**: {duration:.1f}s

---

## Metrics Summary

| Metric | Value |
|--------|-------|
"""
        for name, value in self.metrics.items():
            report += f"| {name} | {value:.4f} |\n"
        
        report += f"""
---

## Plots

### Predictions vs Actual
![Predictions](plots/predictions_sample.png)

### Error Distribution
![Errors](plots/error_distribution.png)

### Scatter Plot
![Scatter](plots/scatter_pred_vs_actual.png)

---

## Configuration

```json
{self.config}
```
"""
        return report
