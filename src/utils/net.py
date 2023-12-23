import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class WN(nn.Module):
    def __init__(self, hidden_channels: int, kernel_size: int, n_layers: int, dilation_rate: int = 2, gin_channels: Optional[int] = None, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.gin_channels = gin_channels

        self.n_layers = n_layers

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()

        self.dropout = nn.Dropout(p=dropout_rate)

        if gin_channels is not None and gin_channels != 0:
            cond_layer = nn.Conv1d(in_channels=gin_channels, out_channels=2*hidden_channels*n_layers, kernel_size=1)
            self.cond_layer = nn.utils.parametrizations.weight_norm(cond_layer, name='weight')

        for i in range(n_layers):
            dilation = dilation_rate ** i

            padding = int((kernel_size * dilation - dilation) / 2)

            in_layer = nn.Conv1d(in_channels=hidden_channels, out_channels=2*hidden_channels, kernel_size=kernel_size, dilation=dilation, padding=padding)
            in_layer = nn.utils.parametrizations.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = nn.Conv1d(hidden_channels, res_skip_channels, kernel_size=1)
            res_skip_layer = nn.utils.parametrizations.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x: torch.Tensor,  g: Optional[torch.Tensor] = None):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)

            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            acts = self.dropout(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, :self.hidden_channels, :]
                x = (x + res_acts)
                output += res_skip_acts[:, self.hidden_channels:, :]
            else:
                output += res_skip_acts

        return output
    
    def remove_weight_norm(self):
        if self.gin_channels is not None:
            nn.utils.remove_weight_norm(self.cond_layer)
        for layer in self.in_layers:
            nn.utils.remove_weight_norm(layer)
        for layer in self.res_skip_layers:
            nn.utils.remove_weight_norm(layer)

@torch.jit.script
def fused_add_tanh_sigmoid_multiply(a: torch.Tensor, b: torch.Tensor, n_channels: torch.Tensor):
    n_channels_int = n_channels[0]
    in_act = a + b
    return torch.tanh(in_act[:, :n_channels_int, :]) * F.sigmoid(in_act[:, n_channels_int:, :])