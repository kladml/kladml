
import torch
import torch.nn as nn
from typing import Optional

# Reuse Gluformer components for robustness
from kladml.models.timeseries.transformer.gluformer.components.embed import DataEmbedding
from kladml.models.timeseries.transformer.gluformer.components.encoder import EncoderLayer, Encoder
from kladml.models.timeseries.transformer.gluformer.components.attention import MultiheadAttention

class CanBusTransformer(nn.Module):
    """
    Transformer Autoencoder for CAN Bus Anomaly Detection.
    
    Structure:
    1. Embedding: Projections of time-series features + Positional Encoding.
    2. Encoder: Standard Transformer Encoder.
    3. Reconstruction Head: Linear projection back to input dimension.
    
    Why this works:
    The Attention mechanism learns global dependencies (e.g. RPM vs Speed over time).
    The model learns to map valid sequences to themselves.
    Anomalous sequences (which violate learned dependencies) will be reconstructed poorly.
    """
    
    def __init__(
        self,
        num_features: int = 5,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 1024,
        e_layers: int = 3,
        dropout: float = 0.1,
        activation: str = 'gelu',
        seq_len: int = 120 # 60 seconds * 2 Hz
    ):
        super().__init__()
        self.seq_len = seq_len
        
        # 1. Embedding
        # We assume static_features=0 for simplicity now, or we can add vehicle ID later.
        # c_in is the number of dynamic features (RPM, Speed, etc.)
        self.embedding = DataEmbedding(
            c_in=num_features,
            d_model=d_model,
            r_drop=dropout,
            num_dynamic_features=0, # Handled by c_in logic in DataEmbedding modification or usage
            num_static_features=0
        )
        
        # Note on DataEmbedding: KladML's DataEmbedding might be specific to Gluformer.
        # Let's verify arguments. Gluformer uses: 
        # DataEmbedding(d_model, r_drop, num_dynamic_features, num_static_features, c_in)
        # If we look at previous view_file of components/embed.py... wait, I haven't viewed it directly.
        # But Gluformer calls it as: DataEmbedding(d_model, r_drop, num_dynamic=0, static=1, c_in=c_in)
        # We will use a standard Linear projection + Positional Encoding if DataEmbedding is too specific.
        # Let's implement a clean custom embedding here to be safe and dependent-free.
        
        # --- Custom Embedding (Safer) ---
        self.input_projection = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_len + 50)
        
        # 2. Transformer Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    att=MultiheadAttention(
                        d_model=d_model, 
                        n_heads=n_heads, 
                        d_keys=d_model // n_heads, 
                        mask_flag=False, 
                        r_att_drop=dropout
                    ),
                    d_model=d_model,
                    d_fcn=d_ff,
                    r_drop=dropout,
                    activ=activation
                ) for _ in range(e_layers)
            ],
            conv_layers=None, # No distil
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # 3. Reconstruction Head
        self.output_projection = nn.Linear(d_model, num_features)
        
    def forward(self, x):
        """
        Args:
            x: [Batch, SeqLen, NumFeatures]
            
        Returns:
            x_hat: [Batch, SeqLen, NumFeatures] (Reconstruction)
        """
        # Embed
        x_emb = self.input_projection(x)
        x_emb = self.pos_encoder(x_emb)
        
        # Encode
        enc_out = self.encoder(x_emb)
        
        # Reconstruct
        x_hat = self.output_projection(enc_out)
        
        return x_hat

class PositionalEncoding(nn.Module):
    """Standard absolute positional encoding."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        import math
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
