
import numpy as np
import logging
from typing import Dict, Tuple, Any
from .timeseries import TimeSeriesEvaluator
from .plots import create_figure, save_figure

class ProbabilisticEvaluator(TimeSeriesEvaluator):
    """
    Evaluator for probabilistic time series forecasting models.
    Expects predictions to be a dictionary with "mean" and "logvar" (or "variance").
    
    Adds:
    - Probabilistic metrics: Coverage, CRPS, Calibration Error, Sharpness
    - Uncertainty visualization: Confidence cones, Calibration curves
    """
    
    def compute_metrics(self, predictions: Dict[str, np.ndarray], targets: np.ndarray) -> Dict[str, float]:
        """
        Compute metrics including probabilistic ones.
        """
        # Ensure predictions have mean/var
        if not isinstance(predictions, dict) or "mean" not in predictions:
            raise ValueError("ProbabilisticEvaluator expects predictions dict with 'mean'.")
            
        mean = predictions["mean"]
        
        # Base point metrics
        base_metrics = super().compute_metrics(mean, targets)
        
        # Check for variance info
        if "logvar" in predictions:
            sigma = np.sqrt(np.exp(predictions["logvar"]))
        elif "variance" in predictions:
            sigma = np.sqrt(predictions["variance"])
        elif "std" in predictions:
            sigma = predictions["std"]
        else:
            self._logger.warning("No uncertainty info (logvar/variance/std) found. Skipping probabilistic metrics.")
            return base_metrics

        # Probabilistic Metrics
        base_metrics.update(self._compute_probabilistic_metrics(mean, sigma, targets))
        
        return base_metrics

    def _compute_probabilistic_metrics(self, mean, sigma, targets) -> Dict[str, float]:
        metrics = {}
        
        # Coverage
        for level in [0.50, 0.90, 0.95]:
            z = self._z_score(level)
            lower = mean - z * sigma
            upper = mean + z * sigma
            coverage = np.mean((targets >= lower) & (targets <= upper))
            metrics[f"Coverage_{int(level * 100)}"] = float(coverage)
            
        # CRPS
        metrics["CRPS"] = float(self._crps_gaussian(mean, sigma, targets))
        
        # Calibration Error
        metrics["Calibration_Error"] = float(self._calibration_error(mean, sigma, targets))
        
        # Sharpness (Width of 95% interval)
        metrics["Sharpness_95"] = float(np.mean(2 * 1.96 * sigma))
        
        return metrics

    def save_plots(self, predictions: Dict[str, np.ndarray], targets: np.ndarray) -> None:
        """
        Save all plots using mean for standard plots + adding probabilistic ones.
        """
        if isinstance(predictions, dict):
            mean = predictions["mean"]
            super().save_plots(mean, targets)
            
            # Uncertainty plots if available
            sigma = None
            if "logvar" in predictions:
                sigma = np.sqrt(np.exp(predictions["logvar"]))
            elif "variance" in predictions: 
                sigma = np.sqrt(predictions["variance"])
                
            if sigma is not None:
                self._plot_uncertainty_cone(mean, sigma, targets)
                self._plot_calibration_curve(mean, sigma, targets)
                self._plot_sharpness(sigma)
        else:
            # Fallback for point estimates
            super().save_plots(predictions, targets)

    def generate_report(self) -> str:
        """
        Generate comprehensive Markdown report for Probabilistic Evaluation.
        Standardizes format for all models inheriting from this.
        """
        from datetime import datetime
        duration = (self._end_time - self._start_time).total_seconds() if self._end_time else 0
        
        # Separate metrics
        point_metrics = {k: v for k, v in self.metrics.items() 
                        if not k.startswith("Coverage") and k not in ["CRPS", "Calibration_Error", "Sharpness_95"]}
        
        prob_metrics = {k: v for k, v in self.metrics.items() 
                       if k.startswith("Coverage") or k in ["CRPS", "Calibration_Error", "Sharpness_95"]}
        
        report = f"""# {self.__class__.__name__} Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Model**: `{self.model_path.name}`  
**Data**: `{self.data_path.name}`  
**Duration**: {duration:.1f}s  
**Config**: `{self.config}`

---

## Point Prediction Metrics

| Metric | Value |
|--------|-------|
"""
        for name, value in point_metrics.items():
            report += f"| {name} | {value:.4f} |\n"
            
        report += f"""
---

## Probabilistic Metrics

| Metric | Value |
|--------|-------|
"""
        for name, value in prob_metrics.items():
            report += f"| {name} | {value:.4f} |\n"
            
        # Interpretation
        cal_err = prob_metrics.get("Calibration_Error", 0)
        interpretation = ""
        if cal_err:
             if cal_err < 0.05:
                 interpretation = "✅ **Well Calibrated**: Uncertainty estimates are reliable."
             elif cal_err < 0.10:
                 interpretation = "⚠️ **Slightly Miscalibrated**: Uncertainty may be over/underestimated."
             else:
                 interpretation = "❌ **Poorly Calibrated**: Uncertainty estimates need improvement."

        report += f"""
### Interpretation

{interpretation}

---

## Plots

### Predictions with Uncertainty Cones
![Uncertainty](plots/uncertainty_cones.png)

### Calibration Curve
![Calibration](plots/calibration_curve.png)

### Sharpness (CI Width) Over Horizon
![Sharpness](plots/sharpness.png)

### Error Distribution (Point Estimate)
![Errors](plots/error_distribution.png)

### Scatter: Predicted vs Actual
![Scatter](plots/scatter_pred_vs_actual.png)

---

*Report generated by KladML ProbabilisticEvaluator*
"""
        return report

    # --- Helper Calculation Methods (Static) ---
    
    @staticmethod
    def _z_score(confidence: float) -> float:
        from scipy import stats
        return stats.norm.ppf((1 + confidence) / 2)
    
    @staticmethod
    def _crps_gaussian(mean, sigma, target) -> float:
        from scipy import stats
        z = (target - mean) / (sigma + 1e-8)
        phi = stats.norm.pdf(z)
        Phi = stats.norm.cdf(z)
        crps = sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / np.sqrt(np.pi))
        return np.mean(crps)
        
    @staticmethod
    def _calibration_error(mean, sigma, target) -> float:
        from scipy import stats
        errors = []
        for expected in np.linspace(0.1, 0.9, 9):
            z = stats.norm.ppf((1 + expected) / 2)
            lower = mean - z * sigma
            upper = mean + z * sigma
            observed = np.mean((target >= lower) & (target <= upper))
            errors.append(abs(observed - expected))
        return np.mean(errors)

    # --- Plotting Methods ---

    def _plot_uncertainty_cone(self, mean, sigma, targets, num_samples=4):
        fig, axes = create_figure(nrows=2, ncols=2, figsize=(12, 8))
        axes = axes.flatten()
        n_samples = min(num_samples, len(mean))
        indices = np.linspace(0, len(mean) - 1, n_samples, dtype=int)
        
        for i, idx in enumerate(indices):
            ax = axes[i]
            horizon = np.arange(len(mean[idx]))
            
            # 95% and 68% CI
            ax.fill_between(horizon, mean[idx] - 1.96*sigma[idx], mean[idx] + 1.96*sigma[idx], alpha=0.3, color="#3498db", label="95% CI")
            ax.fill_between(horizon, mean[idx] - sigma[idx], mean[idx] + sigma[idx], alpha=0.3, color="#2980b9", label="68% CI")
            
            ax.plot(horizon, mean[idx], "-", color="#2c3e50", linewidth=2, label="Prediction")
            ax.plot(horizon, targets[idx], "o", color="#e74c3c", markersize=5, label="Actual")
            
            ax.set_title(f"Sample {idx}")
            if i == 0: ax.legend()
            
        fig.suptitle("Uncertainty Cones")
        fig.tight_layout()
        save_figure(fig, self.plots_dir, "uncertainty_cones")

    def _plot_calibration_curve(self, mean, sigma, targets):
        from scipy import stats
        fig, ax = create_figure()
        
        mean_flat = mean.flatten()
        sigma_flat = sigma.flatten()
        targs_flat = targets.flatten()
        
        expected = np.linspace(0.1, 0.99, 20)
        observed = []
        
        for exp in expected:
            z = stats.norm.ppf((1 + exp) / 2)
            lower = mean_flat - z * sigma_flat
            upper = mean_flat + z * sigma_flat
            obs = np.mean((targs_flat >= lower) & (targs_flat <= upper))
            observed.append(obs)
            
        ax.plot([0, 1], [0, 1], "k--", label="Perfect")
        ax.plot(expected, observed, "o-", color="#3498db", label="Model")
        ax.set_xlabel("Expected Coverage")
        ax.set_ylabel("Observed Coverage")
        ax.legend()
        ax.set_title("Calibration Curve")
        
        save_figure(fig, self.plots_dir, "calibration_curve")

    def _plot_sharpness(self, sigma):
        fig, ax = create_figure()
        ci_width = 2 * 1.96 * sigma
        mean_width = np.mean(ci_width, axis=0)
        horizon = np.arange(len(mean_width))
        
        ax.plot(horizon, mean_width, "-o", color="#3498db")
        ax.set_title("Average 95% CI Width over Horizon")
        ax.set_xlabel("Horizon Step")
        ax.set_ylabel("Width")
        
        save_figure(fig, self.plots_dir, "sharpness")
