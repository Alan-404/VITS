import torch


def fused_add_tanh_sigmoid_multiply(a: torch.Tensor, b: torch.Tensor, n_channels: torch.Tensor):
    n_channels_int = n_channels[0]
    in_act = a + b
    return torch.tanh(in_act[:, :n_channels_int, :]) * torch.sigmoid(in_act[:, n_channels_int:, :])