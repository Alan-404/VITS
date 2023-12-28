import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int, d_model: int, dropout_rate: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.heads = heads

        self.head_samples = self.d_model // self.heads

        self.sqrt_dim = math.sqrt(self.head_samples)

        self.linear_q = nn.Linear(in_features=d_model, out_features=d_model)
        self.linear_k = nn.Linear(in_features=d_model, out_features=d_model)
        self.linear_v = nn.Linear(in_features=d_model, out_features=d_model)
        self.linear_pos = nn.Linear(in_features=d_model, out_features=d_model)

        self.content_bias = nn.Parameter(torch.Tensor(self.heads, self.head_samples))
        self.position_bias = nn.Parameter(torch.Tensor(self.heads, self.head_samples))
        
        self.dropout_layer = nn.Dropout(p=dropout_rate)

        self.linear_out = nn.Linear(in_features=d_model, out_features=d_model)

        torch.nn.init.xavier_uniform_(self.content_bias)
        torch.nn.init.xavier_uniform_(self.position_bias)

    def scaled_dot_product_relative_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pos_embedding: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # q x k^T
        content_score = torch.matmul((q + self.content_bias).transpose(1, 2), k.transpose(2, 3))
        
        # Relative Postional Encoding with score
        pos_score = torch.matmul((q + self.position_bias).transpose(1, 2), pos_embedding)
        pos_score = self.relative_positional_encoding(pos_score)

        # Accumulate score and Scale
        attention_score = (content_score + pos_score) / self.sqrt_dim

        # (Optional) Apply Mask
        if mask is not None:
            attention_score = attention_score.masked_fill(mask, float('-inf'))

        # Softmax
        attention_weights = F.softmax(attention_score, dim=-1)
        attention_weights = self.dropout_layer(attention_weights)

        # attention_weights x v
        attention_context = torch.matmul(attention_weights, v)

        return attention_context
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pos_embedding: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = q.size(0)

        # Project and Split Heads
        q = self.linear_q(q).view(batch_size, -1, self.heads, self.head_samples)
        k = self.linear_k(k).view(batch_size, -1, self.heads, self.head_samples).permute(0, 2, 1, 3)
        v = self.linear_v(v).view(batch_size, -1, self.heads, self.head_samples).permute(0, 2, 1, 3)
        pos_embedding = self.linear_pos(pos_embedding).view(batch_size, -1, self.heads, self.head_samples).permute(0, 2, 3, 1)

        # Scaled - dot Attention with Relative Position
        attention_context = self.scaled_dot_product_relative_attention(q, k, v, pos_embedding, mask)

        # Concat Heads
        attention_context = attention_context.permute([0, 2, 1, 3])
        attention_context = attention_context.reshape((batch_size, -1, self.d_model))

        attention_context = self.linear_out(attention_context)

        return attention_context
    
    def relative_positional_encoding(self, pos_score: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score