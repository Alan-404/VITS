import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.ffn import LayerNorm
from typing import Optional
import math
from src.utils.spline import piecewise_rational_quadratic_transform

class DDSConv(nn.Module):
    def __init__(self, n_layers: int, channels: int, kernel_size: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.channels = channels
        self.n_layers = n_layers
        self.kernel_size = kernel_size

        self.layers = nn.ModuleList([DDSConvLayer(i, channels, kernel_size, dropout_rate) for i in range(n_layers)])
    
    def forward(self, x: torch.Tensor, g: Optional[torch.Tensor] = None):
        for layer in self.layers:
            x = layer(x, g)
        return x

class DDSConvLayer(nn.Module):
    def __init__(self, index: int, channels: int, kernel_size: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate

        dilation = kernel_size ** index
        padding = (kernel_size * dilation - dilation) // 2
        
        self.depth_sep_conv = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=channels)
        self.norm_1 = LayerNorm(dim=channels)
        self.conv_1x1 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.norm_2 = LayerNorm(dim=channels)

    def forward(self, x: torch.Tensor, g: Optional[torch.Tensor] = None):
        if g is not None:
            x += g

        y = self.depth_sep_conv(x)
        y = F.gelu(self.norm_1(y))
        y = self.conv_1x1(y)
        y = F.gelu(self.norm_2(y))

        x += y

        return x
        
class Invertible1x1Convolution(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, bias=False)

        W = torch.linalg.qr(torch.FloatTensor(channels, channels).normal_())[0]

        if torch.det(W) < 0:
            W[:, 0] = -1 * W[:, 0]
        
        W = W.view(channels, channels, 1)
        self.conv.weight.data = W

    def forward(self, x: torch.Tensor, reverse: bool = False):
        batch_size, _, n_of_groups = x.size()

        W = self.conv.weight.squeeze()

        if reverse:
            if not hasattr(self, 'W_inverse'):
                W_inverse = W.float().inverse()
                W_inverse = torch.autograd.Variable(W_inverse[..., None])
                self.W_inverse = W_inverse
            return F.conv1d(x, self.W_inverse, bias=False, stride=1, padding=1)
        else:
            # Feed Forward
            log_det_W = batch_size * n_of_groups * torch.logdet(W)
            z = self.conv(z)
            return z, log_det_W
        
class ConvFlow(nn.Module):
    def __init__(self, in_channels: int, filter_channels: int, kernel_size: int, n_layers: int, num_bins: int=10, tail_bound: float=5.0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.half_channels = in_channels // 2

        self.pre = nn.Conv1d(in_channels=self.half_channels, out_channels=filter_channels, kernel_size=1)
        self.convs = DDSConv(n_layers=n_layers, channels=filter_channels, kernel_size=kernel_size)
        self.proj = nn.Conv1d(in_channels=filter_channels, out_channels=self.half_channels * (num_bins * 3 -1), kernel_size=1)

        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x: torch.Tensor, reverse: bool = False, g: Optional[torch.Tensor] = None):
        x0, x1 = torch.split(x, [self.half_channels] * 2, dim=1)
        h = self.pre(x0)
        h = self.convs(h, g=g)
        h = self.proj(h)

        b, c, t = x0.shape
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)

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

        x = torch.cat([x0, x1], dim=1)
        logdet = torch.sum(logabsdet, dim=[1,2])
        if not reverse:
            return x, logdet
        return x