import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.ffn import LayerNorm
from typing import Optional
import math
from src.utils.affine_coupling import Log, ElementwiseAffine, Flip
from src.utils.convolution import ConvFlow
from src.utils.convolution import DDSConv

class StochasticDurationPredictor(nn.Module):
    def __init__(self, channels: int, filter_channels: int, kernel_size: int, dropout_rate: float, n_flows: int = 4, gin_channels: Optional[int] = None) -> None:
        super().__init__()
        self.channels = channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = Log()
        self.flows = nn.ModuleList()
        self.flows.append(ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(ConvFlow(in_channels=2, filter_channels=channels, kernel_size=kernel_size, n_layers=3))
            self.flows.append(Flip())

        self.post_pre = nn.Conv1d(in_channels=1, out_channels=channels, kernel_size=1)
        self.post_proj = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.post_convs = DDSConv(n_layers=3, channels=channels, kernel_size=kernel_size, dropout_rate=dropout_rate)
        
        self.post_flows = nn.ModuleList()
        self.post_flows.append(ElementwiseAffine(channels=2))
        for i in range(n_flows):
            self.post_flows.append(ConvFlow(in_channels=2, filter_channels=channels, kernel_size=kernel_size, n_layers=3))
            self.post_flows.append(Flip())

        self.pre = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.proj = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.convs = DDSConv(n_layers=3, channels=channels, kernel_size=kernel_size, dropout_rate=dropout_rate)

        if gin_channels is not None and gin_channels != 0:
            self.cond = nn.Conv1d(in_channels=gin_channels, out_channels=channels, kernel_size=1)

    def forward(self, x: torch.Tensor, w: Optional[torch.Tensor] = None, reverse: bool = False, noise_scale: float = 1.0, g: Optional[torch.Tensor] = None):
        x = torch.detach(x)
        x = self.pre(x)

        if g is not None:
            g = torch.detach(g)
            x += self.cond(g)

        x = self.convs(x)
        x = self.proj(x)

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = torch.zeros((x.size(0)))
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w)
            h_w = self.post_proj(h_w)

            e_q = torch.rand((w.size(0), 2, w.size(2))).to(device=x.device, dtype=x.dtype)
            z_q = e_q

            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q)
                logdet_tot_q += logdet_q
            
            z_u, z1 = torch.split(z_q, [1,1], 1)
            u = torch.sigmoid(z_u)
            z0 = (w - u)
            logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)), [1,2])
            logq = torch.sum(-0.5 * (math.log(2*math.pi) + (e_q**2)), [1,2]) - logdet_tot_q

            logdet_tot = 0
            z0, logdet = self.log_flow(z0)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)

            for flow in flows:
                z, logdet = flow(z, reverse=reverse)
                logdet_tot = logdet_tot + logdet

            nll = torch.sum(0.5 * (math.log(2*math.pi) + (z**2)), [1,2]) - logdet_tot
            return nll + logq # [b]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]
            z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
            for flow in flows:
                z = flow(z, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw

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


        