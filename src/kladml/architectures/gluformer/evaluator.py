"""
Gluformer Evaluator for KladML.

Specialized evaluator for probabilistic glucose forecasting.
Adds uncertainty-aware metrics and visualizations.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from datetime import datetime
import numpy as np
import torch

from kladml.evaluation.timeseries import TimeSeriesEvaluator
from kladml.evaluation.plots import create_figure, save_figure


class GluformerEvaluator(TimeSeriesEvaluator):
    """
    Evaluator for Gluformer probabilistic glucose forecasting.
    
    Extends TimeSeriesEvaluator with:
    - Probabilistic metrics: Coverage, CRPS, Calibration Error
    - Uncertainty visualization: Confidence cones, Calibration curves
    """
    
    def __init__(
        self, 
        run_dir: Path, 
        model_path: Path, 
        data_path: Path,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cpu"
    ):
        super().__init__(run_dir, model_path, data_path, config)
        self.device = device
        self._scaler_mean: float = 0.0
        self._scaler_scale: float = 1.0
    
    def load_model(self) -> torch.jit.ScriptModule:
        """
        Load the TorchScript Gluformer model.
        
        Returns:
            Loaded JIT model.
        """
        self._logger.info(f"Loading JIT model from {self.model_path}")
        
        extra_files = {"scaler_mean": "", "scaler_scale": ""}
        model = torch.jit.load(str(self.model_path), _extra_files=extra_files)
        model.eval()
        model.to(self.device)
        
        # Extract scaler stats
        try:
            self._scaler_mean = float(extra_files["scaler_mean"])
            self._scaler_scale = float(extra_files["scaler_scale"])
            self._logger.info(f"Scaler: mean={self._scaler_mean:.2f}, scale={self._scaler_scale:.2f}")
        except Exception as e:
            self._logger.warning(f"Could not extract scaler stats: {e}")
        
        return model
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load evaluation dataset.
        
        Supports both HDF5 (.h5) and PKL (.pkl) formats.
        
        Returns:
            Tuple of (inputs, targets) arrays.
        """
        self._logger.info(f"Loading data from {self.data_path}")
        
        inputs_list = []
        targets_list = []
        
        seq_len = 60
        pred_len = 12
        
        suffix = self.data_path.suffix.lower()
        
        if suffix in [".h5", ".hdf5"]:
            # HDF5 format
            import h5py
            
            with h5py.File(self.data_path, "r") as f:
                if "series" in f:
                    for key in f["series"]:
                        glucose = f["series"][key]["glucose"][:]
                        
                        for i in range(len(glucose) - seq_len - pred_len + 1):
                            x = glucose[i:i + seq_len]
                            y = glucose[i + seq_len:i + seq_len + pred_len]
                            inputs_list.append(x)
                            targets_list.append(y)
        
        elif suffix in [".pkl", ".pickle"]:
            # PKL format (list of dicts with 'x_enc' and 'y' keys, or raw glucose arrays)
            import joblib
            
            data = joblib.load(self.data_path)
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        # Structured format: {'x_enc': [...], 'y': [...]}
                        if 'x_enc' in item and 'y' in item:
                            x = np.array(item['x_enc']).flatten()[:seq_len]
                            y = np.array(item['y']).flatten()[:pred_len]
                            if len(x) == seq_len and len(y) == pred_len:
                                inputs_list.append(x)
                                targets_list.append(y)
                        # Raw glucose series
                        elif 'glucose' in item:
                            glucose = np.array(item['glucose'])
                            for i in range(len(glucose) - seq_len - pred_len + 1):
                                inputs_list.append(glucose[i:i + seq_len])
                                targets_list.append(glucose[i + seq_len:i + seq_len + pred_len])
                    elif isinstance(item, np.ndarray):
                        # Raw glucose array
                        glucose = item.flatten()
                        for i in range(len(glucose) - seq_len - pred_len + 1):
                            inputs_list.append(glucose[i:i + seq_len])
                            targets_list.append(glucose[i + seq_len:i + seq_len + pred_len])
            else:
                raise ValueError(f"Unsupported PKL data structure: {type(data)}")
        
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Use .h5 or .pkl")
        
        inputs = np.array(inputs_list)
        targets = np.array(targets_list)
        
        self._logger.info(f"Loaded {len(inputs)} windows")
        
        return inputs, targets
    
    def inference(
        self, 
        model: torch.jit.ScriptModule, 
        data: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Run inference on the data.
        
        Args:
            model: Loaded JIT model.
            data: Tuple of (inputs, targets).
            
        Returns:
            Tuple of (predictions_dict, targets).
            predictions_dict contains 'mean' and 'logvar'.
        """
        inputs, targets = data
        
        # Scale inputs
        inputs_scaled = (inputs - self._scaler_mean) / self._scaler_scale
        
        # Convert to tensor
        inputs_tensor = torch.tensor(
            inputs_scaled, dtype=torch.float32
        ).unsqueeze(-1).to(self.device)  # [N, 60, 1]
        
        all_means = []
        all_logvars = []
        
        batch_size = 256
        n_batches = (len(inputs_tensor) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in range(n_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, len(inputs_tensor))
                batch = inputs_tensor[start:end]
                
                pred_mean, pred_logvar = model(batch)
                
                # Denormalize predictions
                pred_mean_denorm = pred_mean.cpu().numpy() * self._scaler_scale + self._scaler_mean
                
                all_means.append(pred_mean_denorm.squeeze(-1))
                all_logvars.append(pred_logvar.cpu().numpy().squeeze(-1))
                
                if (i + 1) % 10 == 0:
                    self._logger.debug(f"Inference batch {i + 1}/{n_batches}")
        
        predictions = {
            "mean": np.concatenate(all_means, axis=0),
            "logvar": np.concatenate(all_logvars, axis=0),
        }
        
        self._logger.info(f"Inference complete: {len(predictions['mean'])} predictions")
        
        return predictions, targets
    
    def compute_metrics(
        self, 
        predictions: Dict[str, np.ndarray], 
        targets: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute metrics including probabilistic ones.
        
        Args:
            predictions: Dict with 'mean' and 'logvar'.
            targets: Ground truth values.
            
        Returns:
            Dictionary with all metrics.
        """
        # Base time series metrics on the mean predictions
        base_metrics = super().compute_metrics(predictions["mean"], targets)
        
        # Probabilistic metrics
        mean = predictions["mean"]
        logvar = predictions["logvar"]
        sigma = np.sqrt(np.exp(logvar))
        
        # Coverage at different levels
        for level in [0.50, 0.90, 0.95]:
            z = self._z_score(level)
            lower = mean - z * sigma
            upper = mean + z * sigma
            coverage = np.mean((targets >= lower) & (targets <= upper))
            base_metrics[f"Coverage_{int(level * 100)}"] = float(coverage)
        
        # CRPS (Continuous Ranked Probability Score)
        base_metrics["CRPS"] = float(self._crps_gaussian(mean, sigma, targets))
        
        # Calibration Error (average deviation from expected coverage)
        base_metrics["Calibration_Error"] = float(self._calibration_error(mean, sigma, targets))
        
        # Sharpness (average width of 95% CI)
        base_metrics["Sharpness_95"] = float(np.mean(2 * 1.96 * sigma))
        
        return base_metrics
    
    @staticmethod
    def _z_score(confidence: float) -> float:
        """Get z-score for confidence level."""
        from scipy import stats
        return stats.norm.ppf((1 + confidence) / 2)
    
    @staticmethod
    def _crps_gaussian(mean: np.ndarray, sigma: np.ndarray, target: np.ndarray) -> float:
        """
        Compute CRPS for Gaussian predictions.
        
        CRPS = σ * [z * (2Φ(z) - 1) + 2φ(z) - 1/√π]
        where z = (y - μ) / σ
        """
        from scipy import stats
        
        z = (target - mean) / (sigma + 1e-8)
        phi = stats.norm.pdf(z)
        Phi = stats.norm.cdf(z)
        
        crps = sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / np.sqrt(np.pi))
        return np.mean(crps)
    
    @staticmethod
    def _calibration_error(
        mean: np.ndarray, 
        sigma: np.ndarray, 
        target: np.ndarray
    ) -> float:
        """
        Compute calibration error.
        
        For each confidence level, check if observed coverage matches expected.
        """
        from scipy import stats
        
        errors = []
        for expected in np.linspace(0.1, 0.9, 9):
            z = stats.norm.ppf((1 + expected) / 2)
            lower = mean - z * sigma
            upper = mean + z * sigma
            observed = np.mean((target >= lower) & (target <= upper))
            errors.append(abs(observed - expected))
        
        return np.mean(errors)
    
    def save_plots(
        self, 
        predictions: Dict[str, np.ndarray], 
        targets: np.ndarray
    ) -> None:
        """
        Save all plots including probabilistic ones.
        """
        # Base plots
        super().save_plots(predictions["mean"], targets)
        
        # Probabilistic plots
        self._plot_uncertainty_cone(predictions, targets)
        self._plot_calibration_curve(predictions, targets)
        self._plot_sharpness(predictions)
    
    def _plot_uncertainty_cone(
        self, 
        predictions: Dict[str, np.ndarray], 
        targets: np.ndarray,
        num_samples: int = 4
    ) -> None:
        """Plot predictions with uncertainty cones."""
        fig, axes = create_figure(nrows=2, ncols=2, figsize=(12, 8))
        axes = axes.flatten()
        
        mean = predictions["mean"]
        sigma = np.sqrt(np.exp(predictions["logvar"]))
        
        n_samples = min(num_samples, len(mean))
        indices = np.linspace(0, len(mean) - 1, n_samples, dtype=int)
        
        for i, idx in enumerate(indices):
            ax = axes[i]
            horizon = np.arange(len(mean[idx]))
            
            # Prediction with 95% CI
            ax.fill_between(
                horizon,
                mean[idx] - 1.96 * sigma[idx],
                mean[idx] + 1.96 * sigma[idx],
                alpha=0.3, color="#3498db", label="95% CI"
            )
            ax.fill_between(
                horizon,
                mean[idx] - sigma[idx],
                mean[idx] + sigma[idx],
                alpha=0.3, color="#3498db", label="68% CI"
            )
            ax.plot(horizon, mean[idx], "-", color="#3498db", linewidth=2, label="Prediction")
            ax.plot(horizon, targets[idx], "o", color="#e74c3c", markersize=5, label="Actual")
            
            ax.set_xlabel("Horizon Step (x5 min)")
            ax.set_ylabel("Glucose (mg/dL)")
            ax.set_title(f"Sample {idx}")
            if i == 0:
                ax.legend(loc="upper right")
        
        fig.suptitle("Predictions with Uncertainty Cones", fontsize=14)
        fig.tight_layout()
        save_figure(fig, self.plots_dir, "uncertainty_cones")
    
    def _plot_calibration_curve(
        self, 
        predictions: Dict[str, np.ndarray], 
        targets: np.ndarray
    ) -> None:
        """Plot calibration curve."""
        from scipy import stats
        
        fig, ax = create_figure()
        
        mean = predictions["mean"].flatten()
        sigma = np.sqrt(np.exp(predictions["logvar"])).flatten()
        targs = targets.flatten()
        
        expected_coverages = np.linspace(0.1, 0.99, 20)
        observed_coverages = []
        
        for expected in expected_coverages:
            z = stats.norm.ppf((1 + expected) / 2)
            lower = mean - z * sigma
            upper = mean + z * sigma
            observed = np.mean((targs >= lower) & (targs <= upper))
            observed_coverages.append(observed)
        
        ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Perfect Calibration")
        ax.plot(expected_coverages, observed_coverages, "o-", color="#3498db", 
                linewidth=2, markersize=6, label="Model")
        
        ax.set_xlabel("Expected Coverage")
        ax.set_ylabel("Observed Coverage")
        ax.set_title("Calibration Curve")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        
        save_figure(fig, self.plots_dir, "calibration_curve")
    
    def _plot_sharpness(self, predictions: Dict[str, np.ndarray]) -> None:
        """Plot sharpness (CI width) over horizon."""
        fig, ax = create_figure()
        
        sigma = np.sqrt(np.exp(predictions["logvar"]))
        ci_width = 2 * 1.96 * sigma  # 95% CI width
        
        # Average CI width per horizon step
        mean_width = np.mean(ci_width, axis=0)
        std_width = np.std(ci_width, axis=0)
        
        horizon = np.arange(len(mean_width))
        
        ax.fill_between(
            horizon,
            mean_width - std_width,
            mean_width + std_width,
            alpha=0.3, color="#3498db"
        )
        ax.plot(horizon, mean_width, "-o", color="#3498db", linewidth=2, markersize=5)
        
        ax.set_xlabel("Horizon Step (x5 min)")
        ax.set_ylabel("95% CI Width (mg/dL)")
        ax.set_title("Sharpness: Uncertainty Growth Over Horizon")
        
        save_figure(fig, self.plots_dir, "sharpness")
    
    def generate_report(self) -> str:
        """
        Generate comprehensive Markdown report for Gluformer.
        """
        duration = (self._end_time - self._start_time).total_seconds() if self._end_time else 0
        
        # Separate metrics by category
        point_metrics = {k: v for k, v in self.metrics.items() 
                        if k in ["MAE", "RMSE", "MAPE", "Std_Error", "Max_Error"]}
        prob_metrics = {k: v for k, v in self.metrics.items() 
                       if k.startswith("Coverage") or k in ["CRPS", "Calibration_Error", "Sharpness_95"]}
        
        report = f"""# Gluformer Evaluation Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Model**: `{self.model_path.name}`  
**Data**: `{self.data_path.name}`  
**Duration**: {duration:.1f}s  
**Device**: {self.device}

---

## Point Prediction Metrics

| Metric | Value | Description |
|--------|-------|-------------|
"""
        metric_desc = {
            "MAE": "Mean Absolute Error (mg/dL)",
            "RMSE": "Root Mean Squared Error (mg/dL)",
            "MAPE": "Mean Absolute Percentage Error (%)",
            "Std_Error": "Standard Deviation of Errors",
            "Max_Error": "Maximum Absolute Error",
        }
        for name, value in point_metrics.items():
            desc = metric_desc.get(name, "")
            report += f"| {name} | {value:.4f} | {desc} |\n"
        
        report += f"""
---

## Probabilistic Metrics

| Metric | Value | Description |
|--------|-------|-------------|
"""
        prob_desc = {
            "Coverage_50": "50% CI contains target",
            "Coverage_90": "90% CI contains target",
            "Coverage_95": "95% CI contains target",
            "CRPS": "Continuous Ranked Probability Score",
            "Calibration_Error": "Average deviation from expected coverage",
            "Sharpness_95": "Average 95% CI width (mg/dL)",
        }
        for name, value in prob_metrics.items():
            desc = prob_desc.get(name, "")
            report += f"| {name} | {value:.4f} | {desc} |\n"
        
        # Interpretation
        cal_err = prob_metrics.get("Calibration_Error", 0)
        cov_95 = prob_metrics.get("Coverage_95", 0)
        
        interpretation = ""
        if cal_err < 0.05:
            interpretation = "✅ **Well Calibrated**: Uncertainty estimates are reliable."
        elif cal_err < 0.10:
            interpretation = "⚠️ **Slightly Miscalibrated**: Uncertainty may be over/underestimated."
        else:
            interpretation = "❌ **Poorly Calibrated**: Uncertainty estimates need improvement."
        
        if cov_95 < 0.90:
            interpretation += " Model is **overconfident** (intervals too narrow)."
        elif cov_95 > 0.98:
            interpretation += " Model is **underconfident** (intervals too wide)."
        
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

### Error Distribution
![Errors](plots/error_distribution.png)

### Scatter: Predicted vs Actual
![Scatter](plots/scatter_pred_vs_actual.png)

---

## Scaler Information

| Parameter | Value |
|-----------|-------|
| Mean | {self._scaler_mean:.4f} |
| Scale | {self._scaler_scale:.4f} |

---

*Report generated by KladML Evaluation Module*
"""
        return report
