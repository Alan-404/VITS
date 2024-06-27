import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from typing import Optional

class TemporalProcessing(nn.Module):
    def __init__(self, n_layers: int, hidden_channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.hidden_channels = hidden_channels
        self.sqrt_dim = math.sqrt(0.5)
        self.layers = nn.ModuleList()

        for _ in range(n_layers):
            self.layers.append(nn.Conv1d(in_channels=hidden_channels, out_channels=2 * hidden_channels, kernel_size=kernel_size, padding=padding))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size, n_samples, time, _ = x.size()
        x = x.reshape((batch_size * n_samples, time, self.hidden_channels)).transpose(-1, -2)
        for layer in self.layers:
            xt = layer(x)
            xt = F.glu(xt, dim=1)
            x = (x + xt) * self.sqrt_dim
        x = x.transpose(-1, -2).reshape((batch_size, n_samples, time, self.hidden_channels))
        if mask is not None:
            x = x * mask
        x = torch.sum(x, dim=2) / x.size(2)
        return x

class SpectralProcessing(nn.Module):
    def __init__(self, n_layers: int, n_mel_channels: int, hidden_features: int) -> None:
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Linear(in_features=n_mel_channels, out_features=hidden_features)
        )

        for _ in range(n_layers - 1):
            self.layers.append(
                nn.Linear(in_features=hidden_features, out_features=hidden_features)
            )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        for layer in self.layers:
            x = layer(x)
            if mask is not None:
                x = x * mask
            x = F.elu(x)
        return x