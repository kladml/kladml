import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class CausalConv1d(torch.nn.Conv1d):
  def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                dilation=1,
                groups=1,
                bias=True):
    self.__padding = (kernel_size - 1) * dilation

    super(CausalConv1d, self).__init__(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=self.__padding,
        dilation=dilation,
        groups=groups,
        bias=bias)

  def forward(self, input):
    result = super(CausalConv1d, self).forward(input)
    if self.__padding != 0:
        return result[:, :, :-self.__padding]
    return result

class TriangularCausalMask():
    def __init__(self, b, n, device="cpu"):
        mask_shape = [b, 1, n, n]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class MultiheadAttention(nn.Module):
  def __init__(self, d_model, n_heads, d_keys, mask_flag, r_att_drop=0.1):
    super(MultiheadAttention, self).__init__()
    self.h, self.d, self.mask_flag= n_heads, d_keys, mask_flag
    self.proj_q = nn.Linear(d_model, self.h * self.d)
    self.proj_k = nn.Linear(d_model, self.h * self.d)
    self.proj_v = nn.Linear(d_model, self.h * self.d)
    self.proj_out = nn.Linear(self.h * self.d, d_model)
    self.dropout = nn.Dropout(r_att_drop) 

  def forward(self, q, k, v):
    b, n_q, n_k, h, d = q.size(0), q.size(1), k.size(1), self.h, self.d

    q = self.proj_q(q).reshape(b, n_q, h, d).transpose(1, 2) # b, h, n_q, d
    k = self.proj_k(k).reshape(b, n_k, h, d).transpose(1, 2) # b, h, n_k, d
    v = self.proj_v(v).reshape(b, n_k, h, d).transpose(1, 2) # b, h, n_k, d
    
    # Flash Attention (PyTorch 2.0+)
    if hasattr(F, 'scaled_dot_product_attention'):
        # Automatic Causal Masking if mask_flag is True
        # For Cross-Attn (mask_flag=False), is_causal=False.
        # For Self-Attn (mask_flag=True), is_causal=True ensures triangular mask.
        out = F.scaled_dot_product_attention(q, k, v, 
                                             dropout_p=self.dropout.p if self.training else 0.0, 
                                             is_causal=self.mask_flag)
    else:
        # Fallback to Manual Attention
        scores = torch.einsum('bhnm,bmhd->bhnm', (q,k.transpose(-2, -1))) # b, h, n_q, n_k
        
        if self.mask_flag:
            # Re-create mask logic for manual path
            mask_shape = (b, 1, n_q, n_k)
            mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool, device=q.device), diagonal=1)
            scores.masked_fill_(mask, -np.inf)

        att = F.softmax(scores / (d ** .5), dim=-1)
        att = self.dropout(att)
        out = torch.matmul(att, v) # b, h, n_q, d

    out = out.transpose(1, 2).reshape(b, n_q, h * d) # b, n_q, h*d
    return self.proj_out(out)
