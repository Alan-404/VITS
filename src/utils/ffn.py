import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape=dim)
    def forward(self, x: torch.Tensor):
        x = x.transpose(-1, -2)
        x = self.norm(x)
        x = x.transpose(-1, -2)
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.hidden_layer = nn.Conv1d(in_channels=d_model, out_channels=4*d_model, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.output_layer = nn.Conv1d(in_channels=4*d_model, out_channels=d_model, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor):
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.dropout_layer(x)
        x = self.output_layer(x)
        return x