import torch
import torch.nn as nn
from src.utils.convolution import DDSConv
import math
from src.utils.transform import piecewise_rational_quadratic_transform
from typing import Optional

class ConvFlow(nn.Module):
  def __init__(self, in_channels: int, filter_channels: int, kernel_size: int, n_layers: int, num_bins: int=10, tail_bound: int=5.0):
    super().__init__()
    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.n_layers = n_layers
    self.num_bins = num_bins
    self.tail_bound = tail_bound
    self.half_channels = in_channels // 2

    self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)
    self.convs = DDSConv(n_layers, filter_channels, kernel_size)
    self.proj = nn.Conv1d(filter_channels, self.half_channels * (num_bins * 3 - 1), 1)
    self.proj.weight.data.zero_()
    self.proj.bias.data.zero_()

  def forward(self, x: torch.Tensor, reverse: bool=False, g: Optional[torch.Tensor]=None):
    x0, x1 = torch.split(x, [self.half_channels]*2, 1)
    h = self.pre(x0)
    h = self.convs(h, g=g)
    h = self.proj(h)

    b, c, t = x0.shape
    h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2) # [b, cx?, t] -> [b, c, t, ?]

    unnormalized_widths = h[..., :self.num_bins] / math.sqrt(self.filter_channels)
    unnormalized_heights = h[..., self.num_bins:2*self.num_bins] / math.sqrt(self.filter_channels)
    unnormalized_derivatives = h[..., 2 * self.num_bins:]

    x1, logabsdet = piecewise_rational_quadratic_transform(
        x1,
        unnormalized_widths,
        unnormalized_heights,
        unnormalized_derivatives,
        inverse=reverse,
        tails='linear',
        tail_bound=self.tail_bound
    )

    x = torch.cat([x0, x1], 1)
    logdet = torch.sum(logabsdet, [1,2])
    if not reverse:
        return x, logdet
    else:
        return x