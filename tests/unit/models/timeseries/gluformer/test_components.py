
import pytest
import torch
import torch.nn as nn
from kladml.models.timeseries.transformer.gluformer.components.encoder import Encoder, EncoderLayer, ConvLayer
from kladml.models.timeseries.transformer.gluformer.components.decoder import Decoder, DecoderLayer
from kladml.models.timeseries.transformer.gluformer.components.embed import DataEmbedding, PositionalEmbedding, TokenEmbedding, SubjectEmbedding, TemporalEmbedding

class MockAttention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k, v):
        # Return same shape as query
        return q

# --- ENCODER TESTS ---
def test_conv_layer():
    d_model = 32
    layer = ConvLayer(d_model)
    
    # Input: [Batch, Len, D]
    x = torch.randn(2, 10, d_model)
    out = layer(x)
    
    # Conv1d(k=3, s=1) + MaxPool(k=3, s=2)
    # Len 10 -> Conv -> 10 (padded) -> Pool -> floor((10 + 2*1 - 3)/2 + 1) = 5
    assert out.shape[0] == 2
    assert out.shape[2] == d_model
    assert out.shape[1] == 5 # Halved sequence length

def test_encoder_layer():
    d_model = 32
    d_ff = 64
    layer = EncoderLayer(MockAttention(), d_model, d_ff, r_drop=0.1)
    
    x = torch.randn(2, 10, d_model)
    out = layer(x)
    assert out.shape == x.shape

def test_encoder_stack_simple():
    d_model = 32
    layers = [EncoderLayer(MockAttention(), d_model, 64, 0.1) for _ in range(2)]
    encoder = Encoder(layers, conv_layers=None, norm_layer=nn.LayerNorm(d_model))
    
    x = torch.randn(2, 10, d_model)
    out = encoder(x)
    assert out.shape == x.shape

def test_encoder_stack_preform():
    # Test path with conv layers (Pyramidal attention)
    d_model = 32
    layers = [EncoderLayer(MockAttention(), d_model, 64, 0.1) for _ in range(2)]
    convs = [ConvLayer(d_model)] # Only 1 conv between 2 layers
    
    encoder = Encoder(layers, conv_layers=convs)
    
    x = torch.randn(2, 10, d_model)
    out = encoder(x)
    # 1st layer: 10 -> 10
    # Conv: 10 -> 5
    # 2nd layer: 5 -> 5
    assert out.shape[1] == 5

# --- DECODER TESTS ---
def test_decoder_layer():
    d_model = 32
    d_ff = 64
    layer = DecoderLayer(MockAttention(), MockAttention(), d_model, d_fcn=d_ff, r_drop=0.1)
    
    x_dec = torch.randn(2, 10, d_model)
    x_enc = torch.randn(2, 12, d_model) # Different len
    
    out = layer(x_dec, x_enc)
    assert out.shape == x_dec.shape

def test_decoder_stack():
    d_model = 32
    layers = [DecoderLayer(MockAttention(), MockAttention(), d_model, 64, 0.1) for _ in range(2)]
    decoder = Decoder(layers, norm_layer=nn.LayerNorm(d_model))
    
    x_dec = torch.randn(2, 10, d_model)
    x_enc = torch.randn(2, 12, d_model)
    
    out = decoder(x_dec, x_enc)
    assert out.shape == x_dec.shape

# --- EMBED TESTS ---
def test_embeddings():
    d_model = 32
    bs = 2
    seq_len = 10
    
    # 1. Positional
    pos_emb = PositionalEmbedding(d_model)
    dummy = torch.zeros(bs, seq_len, 1)
    out = pos_emb(dummy)
    assert out.shape == (1, seq_len, d_model)
    
    # 2. Token
    tok_emb = TokenEmbedding(c_in=1, d_model=d_model)
    dummy_x = torch.randn(bs, seq_len, 1)
    out = tok_emb(dummy_x)
    assert out.shape == (bs, seq_len, d_model)
    
    # 3. Subject
    subj_emb = SubjectEmbedding(d_model, num_features=1)
    dummy_id = torch.randn(bs, 1)
    out = subj_emb(dummy_id)
    assert out.shape == (bs, d_model)

def test_data_embedding_full():
    d_model = 32
    # Dyn feats = 1, Static = 1
    embed = DataEmbedding(d_model, r_drop=0.1, num_dynamic_features=1, num_static_features=1)
    
    x = torch.randn(2, 10, 1)
    x_id = torch.randn(2, 1)
    x_mark = torch.randn(2, 10, 1)
    
    out = embed(x_id, x, x_mark)
    # Output length = SeqLen + 1 (because ID is prepended)
    # x_id unsqueezed -> [2, 1, D]
    # x -> [2, 10, D]
    # concat -> [2, 11, D]
    assert out.shape == (2, 11, d_model)

def test_data_embedding_no_mark():
    d_model = 32
    # Dyn feats = 0
    embed = DataEmbedding(d_model, r_drop=0.1, num_dynamic_features=0, num_static_features=1)
    
    x = torch.randn(2, 10, 1)
    x_id = torch.randn(2, 1)
    
    out = embed(x_id, x, x_mark=None)
    assert out.shape == (2, 11, d_model)
