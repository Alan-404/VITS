import torch
import torch.nn as nn
import torch.nn.functional as F

class DDSConv(nn.Module):
    def __init__(self, n_layers: int, channels: int, kernel_size: int, dropout_rate: float) -> None:
        super().__init__()
        self.convolutions = nn.ModuleList([DDSConvLayer(i, channels, kernel_size, dropout_rate) for i in range(n_layers)])

    def forward(self, x: torch.Tensor):
        for conv in self.convolutions:
            x = conv(x)
        return x

class DDSConvLayer(nn.Module):
    def __init__(self, index: int, channels: int, kernel_size: int, dropout_rate: float) -> None:
        super().__init__()
        dilation = kernel_size ** index
        padding = (kernel_size * dilation - dilation) // 2

        self.sep_conv = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels, padding=padding)
        self.norm_1 = nn.LayerNorm(normalized_shape=channels)
        self.conv_1x1 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.norm_2 = nn.LayerNorm(normalized_shape=channels)

        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, x: torch.Tensor):
        res = x

        x = self.sep_conv(x)
        x = F.gelu(self.norm_1(x))
        x = self.conv_1x1(x)
        x = F.gelu(self.norm_2(x))
        x = self.dropout(x)

        return x + res

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