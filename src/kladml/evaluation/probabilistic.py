
from typing import Any
import numpy as np
from kladml.evaluation.base import BaseEvaluator
import logging

class ProbabilisticEvaluator(BaseEvaluator):
    """
    Base class for probabilistic evaluators.
    
    Expects predictions to be a dictionary with keys:
    - 'mean': Point forecast
    - 'logvar': Log variance (uncertainty)
    """
    
    def compute_metrics(self, predictions: dict[str, np.ndarray], targets: np.ndarray) -> dict[str, float]:
        """
        Compute probabilistic metrics (NLL, MSE).
        """
        mean = predictions["mean"]
        logvar = predictions.get("logvar") # Optional
        
        # Flatten for metrics
        mean_flat = mean.flatten()
        target_flat = targets.flatten()
        
        # 1. MSE
        mse = np.mean((mean_flat - target_flat) ** 2)
        metrics = {"mse": float(mse), "rmse": float(np.sqrt(mse))}
        
        # 2. NLL (if uncertainty provided)
        if logvar is not None:
            logvar_flat = logvar.flatten()
            # Gaussian NLL
            # = 0.5 * (log(2pi) + logvar + (y-mean)^2 / exp(logvar))
            # We omit log(2pi) for simplicity or include it
            var = np.exp(logvar_flat)
            nll = 0.5 * (np.log(2 * np.pi) + logvar_flat + (mean_flat - target_flat)**2 / var)
            metrics["nll"] = float(np.mean(nll))
            
        return metrics

    def save_plots(self, predictions: dict[str, np.ndarray], targets: np.ndarray) -> None:
        """
        Save probabilistic forecast plots.
        """
        try:
            import matplotlib.pyplot as plt
            
            # Plot first 3 examples
            n_plots = min(3, len(targets))
            
            for i in range(n_plots):
                gt = targets[i]
                mean = predictions["mean"][i]
                
                plt.figure(figsize=(10, 5))
                plt.plot(gt, label="Ground Truth", color="black")
                plt.plot(mean, label="Forecast", color="blue")
                
                if "logvar" in predictions:
                    logvar = predictions["logvar"][i]
                    std = np.sqrt(np.exp(logvar))
                    plt.fill_between(
                        range(len(mean)),
                        mean - 2*std,
                        mean + 2*std,
                        color="blue",
                        alpha=0.2,
                        label="95% CI"
                    )
                
                plt.title(f"Forecast {i}")
                plt.legend()
                
                plot_path = self.plots_dir / f"example_{i}.png"
                plt.savefig(plot_path)
                plt.close()
                self._logger.info(f"Saved plot: {plot_path}")
                
        except ImportError:
            self._logger.warning("Matplotlib not installed, skipping plots.")
        except Exception as e:
            self._logger.error(f"Failed to plot: {e}")

    def generate_report(self) -> str:
        """Generate markdown report."""
        lines = []
        lines.append(f"# Probabilistic Evaluation Report")
        lines.append(f"Model: {self.model_path}")
        lines.append(f"## Metrics")
        for k, v in self.metrics.items():
            lines.append(f"- **{k}**: {v:.4f}")
        
        return "\n".join(lines)
