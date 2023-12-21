import torch
import torch.nn as nn
from src.utils.position import PositionalEncoding
from src.utils.transformer import TranformerEncoder
from typing import Optional

class PriorEncoder(nn.Module):
    def __init__(self, phoneme_size: int, n: int, d_model: int, heads: int, eps: float, dropout_rate: float) -> None:
        super().__init__()
        self.text_encoder = TextEncoder(phoneme_size=phoneme_size, n=n, d_model=d_model, heads=heads, eps=eps, dropout_rate=dropout_rate)
        self.projection = nn.Linear(in_features=d_model, out_features=d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        hidden_text = self.text_encoder(x, mask)
        proj_text = self.projection(hidden_text)

        return hidden_text, proj_text

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