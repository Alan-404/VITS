import torch
import torch.nn as nn
from typing import Optional
from src.utils.net import WN

class ResidualCouplingBlock(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, kernel_size: int, n_layers: int, dilation_rate: int = 2, n_flows: int = 4, dropout_rate: float = 0.0, gin_channels: Optional[int] = None) -> None:
        super().__init__()

        self.n_flows = n_flows

        self.flows = nn.ModuleList()
        self.flips = nn.ModuleList()

        for _ in range(n_flows):
            self.flows.append(ResidualCouplingLayer(
                channels=channels,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                n_layers=n_layers,
                dropout_rate=dropout_rate,
                gin_channels=gin_channels
            ))
            self.flips.append(Flip())
    
    def forward(self, x: torch.Tensor, reverse: bool = False, g: Optional[torch.Tensor] = None):
        if not reverse:
            for idx in range(self.n_flows):
                x, _ = self.flows[idx](x, reverse, g)
                x, _ = self.flips[idx](x, reverse)
        else:
            for idx in range(self.n_flows):
                x = self.flips[self.n_flows - idx - 1](x, reverse)
                x = self.flows[self.n_flows - idx - 1](x, reverse, g)
        return x

class ResidualCouplingLayer(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, kernel_size: int, dilation_rate: int, n_layers: int, dropout_rate: float = 0.0, gin_channels: Optional[int] = None, mean_only: bool = False) -> None:
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(in_channels=self.half_channels, out_channels=hidden_channels, kernel_size=1)
        self.encode = WN(
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            n_layers=n_layers,
            dilation_rate=dilation_rate,
            gin_channels=gin_channels,
            dropout_rate=dropout_rate
        )

        self.post = nn.Conv1d(in_channels=hidden_channels, out_channels=self.half_channels * (2 - mean_only), kernel_size=1)

    def forward(self, x: torch.Tensor, reverse: bool = False, g: Optional[torch.Tensor] = None):
        x0, x1 = torch.split(x, [self.half_channels]*2, dim=1)

        h = self.pre(x0)
        h = self.encode(h, g)

        stats = self.post(h)

        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, dim=1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs)
            x = torch.cat([x0, x1], dim=1)
            logdet = torch.sum(logs, [1,2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs)
            x = torch.cat([x0, x1], dim=1)
            return x

class Flip(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x: torch.Tensor, reverse: bool = False):
        x = torch.flip(x, dims=[1])
        if not reverse:
            logdet = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x
        
class Log(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x: torch.Tensor, reverse: bool = False):
        if not reverse:
            y = torch.log(torch.clamp(x, 1e-5))
            logdet = torch.sum(-y, dim=[1,2])
            return y, logdet
        else:
            x = torch.exp(x)
            return x

class ElementwiseAffine(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.m = nn.Parameter(torch.zeros(channels, 1))
        self.logs = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x: torch.Tensor, reverse: bool = False):
        if reverse == False:
            y = self.m + torch.exp(self.logs) * x
            logdet = torch.sum(self.logs, dim=0)
            return y, logdet
        else:
            x = (x - self.m) * torch.exp(-self.logs)
            return x