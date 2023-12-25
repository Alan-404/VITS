import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.ffn import LayerNorm
from typing import Optional

class StochasticDurationPredictor(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

class DurationPredictor(nn.Module):
    def __init__(self, in_channels: int, filter_channels: int, kernel_size: int, dropout_rate: float = 0.0, gin_channels: Optional[int] = None) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.dropout = nn.Dropout(p=dropout_rate)

        self.conv_1 = nn.Conv1d(in_channels=in_channels, out_channels=filter_channels, kernel_size=kernel_size, padding=padding)
        self.norm_1 = LayerNorm(dim=filter_channels)
        self.conv_2 = nn.Conv1d(in_channels=filter_channels, out_channels=filter_channels, kernel_size=kernel_size, padding=padding)
        self.norm_2 = LayerNorm(dim=filter_channels)
        self.proj = nn.Conv1d(in_channels=filter_channels, out_channels=1, kernel_size=1)

        if gin_channels is not None:
            self.cond = nn.Conv1d(in_channels=gin_channels, out_channels=in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, g: Optional[torch.Tensor] = None):
        x = torch.detach(x)
        
        if g is not None:
            g = torch.detach(g)
            x += self.cond(g)

        x = self.conv_1(x)
        x = F.relu(x)
        x = self.norm_1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.conv_2(x)
        x = self.norm_2(x)
        x = self.dropout(x)
        
        x = self.proj(x)

        return x


        