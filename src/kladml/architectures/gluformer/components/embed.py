import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEmbedding(nn.Module):
  def __init__(self, d_model, max_len=5000):
    super(PositionalEmbedding, self).__init__()
    # Compute the positional encodings once in log space.
    pos_emb = torch.zeros(max_len, d_model)
    pos_emb.require_grad = False

    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = (torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)).exp()

    pos_emb[:, 0::2] = torch.sin(position * div_term)
    pos_emb[:, 1::2] = torch.cos(position * div_term)

    pos_emb = pos_emb.unsqueeze(0)
    self.register_buffer('pos_emb', pos_emb)

  def forward(self, x):
    return self.pos_emb[:, :x.size(1)]

class TokenEmbedding(nn.Module):
  def __init__(self, c_in, d_model):
    super(TokenEmbedding, self).__init__()
    self.conv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                          kernel_size=3, padding=1, padding_mode='replicate')
    # nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='leaky_relu')

  def forward(self, x):
    x = self.conv(x.transpose(-1, 1)).transpose(-1, 1)
    return x

class TemporalEmbedding(nn.Module):
  def __init__(self, d_model, num_features):
    super(TemporalEmbedding, self).__init__()
    self.embed = nn.Linear(num_features, d_model)
  
  def forward(self, x):
    return self.embed(x)

class SubjectEmbedding(nn.Module):
  def __init__(self, d_model, num_features):
    super(SubjectEmbedding, self).__init__()
    self.id_embedding = nn.Linear(num_features, d_model)

  def forward(self, x):
    embed_x = self.id_embedding(x)

    return embed_x

class DataEmbedding(nn.Module):
  def __init__(self, d_model, r_drop, num_dynamic_features, num_static_features, c_in=1):
    super(DataEmbedding, self).__init__()
    # Value Embedding (Target + Covariates if combined)
    self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
    
    # Temporal Embedding (Separate covariates like Time)
    # If num_dynamic_features > 0 and we use x_mark, this is used.
    # Note: If we put covariates in 'x' (channels), we might not use this.
    if num_dynamic_features > 0:
        self.time_embedding = TemporalEmbedding(d_model, num_dynamic_features) 
    else:
        self.time_embedding = None
        
    self.positional_embedding = PositionalEmbedding(d_model)
    self.subject_embedding = SubjectEmbedding(d_model, num_static_features)
    self.dropout = nn.Dropout(r_drop)

  def forward(self, x_id, x, x_mark=None):
    # x_id: [Batch, StaticDim] (e.g. ID)
    # x: [Batch, SeqLen, 1] (Target)
    # x_mark: [Batch, SeqLen, DynDim] (Time/Other)
    
    x_emb = self.value_embedding(x) + self.positional_embedding(x)
    
    if self.time_embedding is not None and x_mark is not None:
        x_emb = x_emb + self.time_embedding(x_mark)
        
    x_id_emb = self.subject_embedding(x_id)
    # Concat static embedding as the 'start token' or similar
    # The original Gluformer concat: x = torch.cat((x_id.unsqueeze(1), x), dim = 1)
    # x_id is [Batch, D]. unsqueeze(1) -> [Batch, 1, D].
    # x is [Batch, L, D]
    # Result [Batch, L+1, D]
    
    x_out = torch.cat((x_id_emb.unsqueeze(1), x_emb), dim = 1)
    return self.dropout(x_out)
