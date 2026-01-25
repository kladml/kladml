
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TriangularCausalMask:
    """
    Creates a triangular causal mask for autoregressive attention.
    Used to prevent positions from attending to subsequent positions.
    """
    def __init__(self, b, n, device="cpu"):
        mask_shape = [b, 1, n, n]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class MultiheadAttention(nn.Module):
    """
    Scalable Multi-Head Attention using PyTorch 2.0+ Scaled Dot Product Attention (SDPA).
    
    Supports:
    - Flash Attention (via SDPA backend if available)
    - Memory Efficient Attention
    - Causal Masking (autoregressive)
    """
    def __init__(self, d_model, n_heads, d_keys=None, mask_flag=False, r_att_drop=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.h = n_heads
        self.d = d_keys or (d_model // n_heads)
        self.mask_flag = mask_flag
        
        # Projections
        self.proj_q = nn.Linear(d_model, self.h * self.d)
        self.proj_k = nn.Linear(d_model, self.h * self.d)
        self.proj_v = nn.Linear(d_model, self.h * self.d)
        self.proj_out = nn.Linear(self.h * self.d, d_model)
        
        self.dropout = nn.Dropout(r_att_drop) 
        
    def forward(self, q, k, v, attn_mask=None):
        """
        Args:
            q: [Batch, SeqLen_Q, D_Model]
            k: [Batch, SeqLen_K, D_Model]
            v: [Batch, SeqLen_K, D_Model]
        """
        b, n_q, n_k = q.size(0), q.size(1), k.size(1)
        h, d = self.h, self.d

        # Project and Reshape for SDPA: [Batch, Heads, SeqLen, HeadDim]
        q_proj = self.proj_q(q).view(b, n_q, h, d).transpose(1, 2) 
        k_proj = self.proj_k(k).view(b, n_k, h, d).transpose(1, 2)
        v_proj = self.proj_v(v).view(b, n_k, h, d).transpose(1, 2)
        
        # Flash Attention / SDPA
        # This automatically selects the best kernel (FlashAttention, MemEfficient, or Math)
        if hasattr(F, 'scaled_dot_product_attention'):
            # If attn_mask is provided, we disable is_causal flag for SDPA to avoid conflict?
            is_causal = self.mask_flag
            if attn_mask is not None:
                is_causal = False

            out = F.scaled_dot_product_attention(
                q_proj, k_proj, v_proj, 
                dropout_p=self.dropout.p if self.training else 0.0, 
                is_causal=is_causal,
                attn_mask=attn_mask
            )
        else:
            # Fallback for old PyTorch (should rarely be hit in KladML v0.9+)
            scores = torch.einsum('bhnm,bmhd->bhnm', (q_proj, k_proj.transpose(-2, -1)))
            
            if self.mask_flag:
                mask_shape = (b, 1, n_q, n_k)
                mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool, device=q.device), diagonal=1)
                scores.masked_fill_(mask, -np.inf)

            att = F.softmax(scores / (d ** 0.5), dim=-1)
            att = self.dropout(att)
            out = torch.matmul(att, v_proj) # b, h, n_q, d

        # Reshape back: [Batch, SeqLen, D_Model]
        out = out.transpose(1, 2).contiguous().view(b, n_q, h * d)
        return self.proj_out(out)
