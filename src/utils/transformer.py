import torch
import torch.nn as nn
from src.utils.attention import MultiHeadAttention
from src.utils.ffn import FeedForward
from typing import Optional


class TranformerEncoder(nn.Module):
    def __init__(self, n: int, d_model: int, heads: int, eps: float, dropout_rate: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model, heads=heads, eps=eps, dropout_rate=dropout_rate) for _ in range(n)])

    def forward(self, x: torch.Tensor, pos_embedding: torch.Tensor, mask: Optional[torch.Tensor] = None):
        for layer in self.layers:
            x = layer(x, pos_embedding, mask)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, eps: float, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.attention_layer = MultiHeadAttention(heads=heads, d_model=d_model, dropout_rate=dropout_rate)
        self.ffn = FeedForward(d_model=d_model, dropout_rate=dropout_rate)

        self.residual_1 = ResidualConnection(d_model=d_model, eps=eps, dropout_rate=dropout_rate)
        self.residual_2 = ResidualConnection(d_model=d_model, eps=eps, dropout_rate=dropout_rate)

    def forward(self, x: torch.Tensor, pos_embedding: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # sublayer 1
        attention_out = self.attention_layer(x, x, x, pos_embedding, mask)
        sublayer_1 = self.residual_1(attention_out, x)

        # sublayer 2
        sublayer_1 = sublayer_1.transpose(-1, -2)
        ffn_out = self.ffn(sublayer_1)
        sublayer_1 = sublayer_1.transpose(-1, -2)
        ffn_out = ffn_out.transpose(-1, -2)
        output = self.residual_2(ffn_out, sublayer_1)

        return output

class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, eps: float, dropout_rate: float) -> None:
        super().__init__()
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model, eps=eps)

    def forward(self, x: torch.Tensor, prev_x: torch.Tensor):
        x = self.dropout_layer(x)
        x = self.layer_norm(x + prev_x)
        return x