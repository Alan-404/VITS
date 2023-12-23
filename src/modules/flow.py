import torch
import torch.nn as nn
from src.utils.convolution import DDSConv

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

class ConvFlow(nn.Module):
    def __init__(self, in_channels: int, filter_channels: int, kernel_size: int, n_layers: int, num_bins: int = 10, tail_bound: float = 5.0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.half_channels = in_channels // 2

        self.pre = nn.Conv1d(self.half_channels, filter_channels, kernel_size=1)
        self.convs = DDSConv(n_layers=n_layers, channels=filter_channels, kernel_size=kernel_size)
        self.proj = nn.Conv1d(filter_channels, self.half_channels * (num_bins * 3 -1), kernel_size=1)

        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()
    