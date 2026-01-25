
import pytest
import torch
from kladml.models.layers.attention import MultiheadAttention

def test_multihead_attention_sdpa():
    """Test SDPA based attention."""
    batch_size = 2
    seq_len = 10
    embed_dim = 32
    num_heads = 4
    
    attn = MultiheadAttention(embed_dim, num_heads)
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Forward pass
    out = attn(x, x, x)
    
    assert out.shape == (batch_size, seq_len, embed_dim)
    # Weights might be None if using SDPA fast path?
    # Implementation: if need_weights=True, returns weights.
    
    # Verify values logic (simple run)
    assert not torch.isnan(out).any()

def test_multihead_attention_mask():
    """Test with mask."""
    batch_size = 2
    seq_len = 10
    embed_dim = 32
    num_heads = 4
    
    attn = MultiheadAttention(embed_dim, num_heads)
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Causal Mask (Upper triangular -inf)
    # Shape: (seq_len, seq_len)
    mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
    
    out = attn(x, x, x, attn_mask=mask)
    assert out.shape == (batch_size, seq_len, embed_dim)
