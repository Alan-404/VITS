import torch
import torch.nn as nn

class ElementwiseAffine(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.m = nn.Parameter(torch.zeros(channels, 1))
        self.logs = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor, reverse: bool = False):
        if reverse == False:
            y = self.m + torch.exp(self.logs) * x
            y = y * mask
            logdet = torch.sum(self.logs * mask, [1,2])
            return y, logdet
        else:
            x = (x - self.m) * torch.exp(-self.logs) * mask
            return x