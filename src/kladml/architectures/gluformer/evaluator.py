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

from kladml.evaluation.probabilistic import ProbabilisticEvaluator
from kladml.evaluation.plots import create_figure, save_figure


class GluformerEvaluator(ProbabilisticEvaluator):
    """
    Evaluator for Gluformer probabilistic glucose forecasting.
    
    Inherits probabilistic capabilities from ProbabilisticEvaluator.
    Specializes in loading Gluformer specific JIT models and data formats.
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

    # ... [load_model, load_data, inference methods kept specific to Gluformer handling] ...
    # ... [compute_metrics removed -> uses parent] ...
    # ... [save_plots removed -> uses parent] ...
    # ... [plot helper methods removed -> uses parent] ...

    
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
    
    # compute_metrics, save_plots, and _plot_* methods are now inherited from ProbabilisticEvaluator.

    
    # generate_report is now inherited from ProbabilisticEvaluator for standardization.

