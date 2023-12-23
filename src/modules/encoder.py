import torch
import torch.nn as nn
from src.utils.position import PositionalEncoding
from src.utils.transformer import TranformerEncoder
from src.utils.net import WN
from typing import Optional

class PosteriorEncoder(nn.Module):
    def __init__(self, n_layers: int, in_channels: int, out_channels: int, hidden_channels: int, kernel_size: int, dilation_rate: int = 2, gin_channels: Optional[int] = None) -> None:
        super().__init__()
        self.out_channels = out_channels

        self.pre = nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1)
        self.encoder = WN(hidden_channels=hidden_channels, kernel_size=kernel_size, n_layers=n_layers, dilation_rate=dilation_rate, gin_channels=gin_channels)
        self.proj = nn.Conv1d(in_channels=hidden_channels, out_channels=2*out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, g: Optional[torch.Tensor] = None):
        x = self.pre(x)
        x = self.encoder(x, g)
        stats = self.proj(x)

        m, logs = torch.split(stats, self.out_channels, dim=1)

        z = m + torch.rand_like(m) * torch.exp(logs)

        return z, m, logs

class PriorEncoder(nn.Module):
    def __init__(self, phoneme_size: int, n: int, d_model: int, heads: int, eps: float, dropout_rate: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.text_encoder = TextEncoder(phoneme_size=phoneme_size, n=n, d_model=d_model, heads=heads, eps=eps, dropout_rate=dropout_rate)
        self.projection = nn.Linear(in_features=d_model, out_features=2*d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        hidden_text = self.text_encoder(x, mask)
        mean, sigma = torch.split(self.projection(hidden_text), self.d_model, dim=-1)
        return hidden_text, mean, sigma

class TextEncoder(nn.Module):
    def __init__(self, phoneme_size: int, n: int, d_model: int, heads: int, eps: float, dropout_rate: float) -> None:
        super().__init__()
        self.phoneme_embedding = nn.Embedding(num_embeddings=phoneme_size, embedding_dim=d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model)
        self.encoder = TranformerEncoder(n=n, d_model=d_model, heads=heads, eps=eps, dropout_rate=dropout_rate)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = self.phoneme_embedding(x)
        pos_embedding = self.positional_encoding(x.size(1)).repeat(x.size(0), 1, 1)
        x = self.encoder(x, pos_embedding, mask)
        return x