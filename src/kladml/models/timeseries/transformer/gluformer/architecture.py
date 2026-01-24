"""
Gluformer Model.

Native PyTorch implementation of Gluformer for Glucose Forecasting.
Supports univariate (Glucose Only) and multivariate (Glucose + Insulin/Carbs) modes.

Based on Transformer architecture with:
- Encoder: Self-attention + Feedforward
- Decoder: Self-attention + Cross-attention + Feedforward
- Probabilistic output: Mean + Variance (LogVar)
"""

import torch
import torch.nn as nn
import numpy as np

from kladml.models.timeseries.transformer.gluformer.components.embed import DataEmbedding
from kladml.models.timeseries.transformer.gluformer.components.attention import MultiheadAttention, CausalConv1d
from kladml.models.timeseries.transformer.gluformer.components.encoder import EncoderLayer, Encoder, ConvLayer
from kladml.models.timeseries.transformer.gluformer.components.decoder import DecoderLayer, Decoder


class Gluformer(nn.Module):
    """
    Native implementation of Gluformer for Glucose Forecasting.
    
    Supports univariate (Glucose Only) and multivariate (Glucose + Insulin/Carbs) modes.
    
    Args:
        d_model: Model dimension (default: 512)
        n_heads: Number of attention heads (default: 8)
        d_fcn: Feedforward network dimension (default: 256)
        r_drop: Dropout rate (default: 0.1)
        activ: Activation function ('gelu' or 'relu')
        num_enc_layers: Number of encoder layers (default: 3)
        num_dec_layers: Number of decoder layers (default: 2)
        distil: Use distillation convolution (default: False)
        len_seq: Input sequence length (default: 60)
        len_pred: Prediction length (default: 12)
        num_dynamic_features: Number of dynamic covariates (default: 0)
        num_static_features: Number of static features (default: 1)
        label_len: Decoder label length (default: 48)
    
    Returns:
        Tuple of (pred_mean, pred_logvar):
            - pred_mean: [Batch, PredLen, 1] predicted mean
            - pred_logvar: [Batch, PredLen, 1] predicted log-variance
    """
    
    def __init__(
        self, 
        d_model: int = 512, 
        n_heads: int = 8, 
        d_fcn: int = 256, 
        r_drop: float = 0.1, 
        activ: str = 'gelu', 
        num_enc_layers: int = 3, 
        num_dec_layers: int = 2, 
        distil: bool = False, 
        len_seq: int = 60, 
        len_pred: int = 12, 
        num_dynamic_features: int = 0,
        num_static_features: int = 1,
        label_len: int = 48,
    ):
        super(Gluformer, self).__init__()
        
        self.len_pred = len_pred
        self.label_len = label_len
        self.len_seq = len_seq

        # Embedding
        # Encoder Input: [Batch, Seq, NumFeatures] -> c_in=num_dynamic_features
        # If univariate, num_dynamic_features should be 1.
        # But wait, num_dynamic_features usually excludes target?
        # In my definition in model.py, I set num_dynamic_features=2 (Glucose+Insulin).
        # So c_in should be 2.
        
        c_in_enc = num_dynamic_features if num_dynamic_features > 0 else 1
        # Decoder Input: [Batch, Pred, NumFeatures] 
        # Same channels as encoder usually
        c_in_dec = num_dynamic_features if num_dynamic_features > 0 else 1

        self.enc_embedding = DataEmbedding(d_model, r_drop, num_dynamic_features=0, num_static_features=num_static_features, c_in=c_in_enc)
        self.dec_embedding = DataEmbedding(d_model, r_drop, num_dynamic_features=0, num_static_features=num_static_features, c_in=c_in_dec)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    att=MultiheadAttention(
                        d_model=d_model, 
                        n_heads=n_heads, 
                        d_keys=d_model // n_heads, 
                        mask_flag=False, 
                        r_att_drop=r_drop
                    ),
                    d_model=d_model,
                    d_fcn=d_fcn,
                    r_drop=r_drop,
                    activ=activ
                ) for _ in range(num_enc_layers)
            ],
            [
                ConvLayer(d_model) for _ in range(num_enc_layers - 1)
            ] if distil else None, 
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    self_att=MultiheadAttention(
                        d_model=d_model, 
                        n_heads=n_heads, 
                        d_keys=d_model // n_heads, 
                        mask_flag=True, 
                        r_att_drop=r_drop
                    ),
                    cross_att=MultiheadAttention(
                        d_model=d_model, 
                        n_heads=n_heads, 
                        d_keys=d_model // n_heads, 
                        mask_flag=False, 
                        r_att_drop=r_drop
                    ),
                    d_model=d_model,
                    d_fcn=d_fcn,
                    r_drop=r_drop,
                    activ=activ
                ) for _ in range(num_dec_layers)
            ], 
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # Output Projection (Mean)
        D_OUT = 1
        self.projection = nn.Linear(d_model, D_OUT, bias=True)

        # Variance Projection (LogVar) - Intelligent Variance Head (MLP)
        self.projection_var = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x_id, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forward pass.
        
        Args:
            x_id: [Batch, StaticDim] Patient ID features
            x_enc: [Batch, SeqLen, 1] Past Glucose values
            x_mark_enc: [Batch, SeqLen, DynDim] Past Covariates (Optional)
            x_dec: [Batch, LabelLen+PredLen, 1] Decoder Input
            x_mark_dec: [Batch, LabelLen+PredLen, DynDim] Future Covariates (Optional)
            
        Returns:
            Tuple of (pred_mean, pred_logvar):
                - pred_mean: [Batch, PredLen, 1]
                - pred_logvar: [Batch, PredLen, 1]
        """
        enc_out = self.enc_embedding(x_id, x_enc, x_mark_enc)
        enc_out = self.encoder(enc_out)

        dec_out = self.dec_embedding(x_id, x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out)
        
        # Project Mean
        pred_mean = self.projection(dec_out)
        
        # Project LogVar (Variance)
        pred_logvar = self.projection_var(dec_out)
        
        return pred_mean[:, -self.len_pred:, :], pred_logvar[:, -self.len_pred:, :]
    
    def get_config(self) -> dict:
        """Get model configuration."""
        return {
            "len_seq": self.len_seq,
            "len_pred": self.len_pred,
            "label_len": self.label_len,
        }
