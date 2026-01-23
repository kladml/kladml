"""
Deployment utilities for Gluformer.

Handles specialized wrappers for Edge/TorchScript export.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class GluformerDeploymentWrapper(nn.Module):
    """
    Wrapper for Gluformer model optimized for deployment (TorchScript).
    
    Simplifies the input signature to a single tensor [Batch, 60, 1].
    Hardcodes the decoder input construction (Start Token + Zeros).
    
    Args:
        model: Trained Gluformer model instance
        label_len: Length of start token (default: 48)
        pred_len: Prediction horizon (default: 12)
    """
    
    def __init__(self, model: nn.Module, label_len: int = 48, pred_len: int = 12, temperature: float = 1.0):
        super().__init__()
        self.model = model
        self.label_len = label_len
        self.pred_len = pred_len
        self.temperature = temperature
        
        # Ensure model is in eval mode
        self.model.eval()

    def forward(self, x_enc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for deployment inference.
        
        Args:
            x_enc: Input sequence [Batch, 60, 1] (Glucose values)
            
        Returns:
            pred_mean: [Batch, 12, 1]
            pred_logvar: [Batch, 12, 1]
        """
        # 1. Dummy ID: [Batch, 1] -> [0]
        # In Gluformer.forward, x_id is passed to DataEmbedding.
        # DataEmbedding usually expects [Batch, NumStaticFeatures] (default 1).
        # We create a zero tensor.
        batch_size = x_enc.size(0)
        device = x_enc.device
        
        x_id = torch.zeros(batch_size, 1, device=device)
        
        # 2. Decoder Input Construction
        # x_dec should be [Batch, LabelLen + PredLen, 1]
        # Start Token: Last label_len steps of encoder input
        # Future: Zeros
        
        # Extract start token: x_enc[:, -label_len:, :]
        # Note: x_enc is [Batch, 60, 1]. label_len is 48.
        # We take indices [60-48 : 60] -> [12:60]
        
        token_start_idx = x_enc.size(1) - self.label_len
        start_token = x_enc[:, token_start_idx:, :]
        
        # Create zeros for prediction horizon
        zeros = torch.zeros(batch_size, self.pred_len, 1, device=device)
        
        # Concatenate
        x_dec = torch.cat([start_token, zeros], dim=1)
        
        # 3. Covariates (x_mark_enc, x_mark_dec)
        # Assuming model trained without temporal covariates (or accepts None)
        # Based on Gluformer.forward: enc_out = self.enc_embedding(x_id, x_enc, x_mark_enc)
        # Let's check DataEmbedding signature. If it handles None, we are good.
        # BUT for TorchScript strictness, passing None might be tricky if typed optional.
        # Better to pass None if supported.
        
        x_mark_enc = None
        x_mark_dec = None
        
        # 4. Forward Pass
        # self.model(x_id, x_enc, x_mark_enc, x_dec, x_mark_dec)
        pred_mean, pred_logvar = self.model(x_id, x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # 5. Temperature Scaling for calibration
        # Temperature adjusts variance: var_adjusted = var_original / temperature
        # In log space: logvar_adjusted = logvar - log(temperature)
        # temperature < 1 => subtract negative => logvar increases => wider CI
        # temperature > 1 => subtract positive => logvar decreases => narrower CI
        import math
        log_temp = math.log(self.temperature) if self.temperature > 0 else 0.0
        pred_logvar_scaled = pred_logvar - log_temp
        
        return pred_mean, pred_logvar_scaled


def export_to_torchscript(
    model: nn.Module,
    output_path: str,
    scaler: Optional[Any] = None,
    seq_len: int = 60,
    pred_len: int = 12,
    label_len: int = 48,
    temperature: float = 1.0,
) -> None:
    """
    Export Gluformer model to TorchScript (.pt) for deployment.
    
    Args:
        model: Trained Gluformer PyTorch model
        output_path: Path to save the exported .pt file
        scaler: Optional sklearn scaler (StandardScaler) to embed metadata
        seq_len: Input sequence length
        pred_len: Prediction length
        label_len: Decoder label length
        temperature: Calibration temperature for logvar scaling (< 1 = wider CI)
    """
    try:
        # 1. Prepare Metadata (Extra Files)
        extra_files = {}
        if scaler:
            try:
                mean_val = scaler.mean_[0]
                scale_val = scaler.scale_[0]
                extra_files["scaler_mean"] = str(mean_val).encode('utf-8')
                extra_files["scaler_scale"] = str(scale_val).encode('utf-8')
                logger.info(f"Embedding scaler stats: mean={mean_val}, scale={scale_val}")
            except Exception as e:
                logger.warning(f"Could not extract scaler stats: {e}")
                extra_files["scaler_mean"] = b"0.0"
                extra_files["scaler_scale"] = b"1.0"
        else:
            logger.warning("No scaler provided. Embedding defaults (0.0, 1.0).")
            extra_files["scaler_mean"] = b"0.0"
            extra_files["scaler_scale"] = b"1.0"
        
        # Add temperature to metadata
        extra_files["temperature"] = str(temperature).encode('utf-8')
        logger.info(f"Embedding temperature: {temperature}")
            
        # 2. Wrap Model
        # Ensure CPU
        model.to("cpu")
        model.eval()
        
        deploy_model = GluformerDeploymentWrapper(
            model=model,
            label_len=label_len,
            pred_len=pred_len,
            temperature=temperature
        )
        deploy_model.eval()
        
        # 3. Trace
        # Input shape: [1, seq_len, 1]
        dummy_input = torch.randn(1, seq_len, 1)
        
        logger.info(f"Tracing model with input shape {dummy_input.shape}...")
        traced_script_module = torch.jit.trace(deploy_model, dummy_input)
        
        # 4. Save
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        traced_script_module.save(str(output_path), _extra_files=extra_files)
        logger.info(f"Exported model to: {output_path}")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise
