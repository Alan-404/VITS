import torch
import torch.nn as nn
from src.modules.encoder import PriorEncoder, PosteriorEncoder
from src.modules.decoder import Generator
from src.utils.affine_coupling import ResidualCouplingBlock
from typing import List

class VITS(nn.Module):
    def __init__(self, 
                 phoneme_size: int, 
                 n: int, 
                 d_model: int, 
                 heads: int, 
                 n_mels: int, 
                 n_layers: int,
                 hidden_channels: int, 
                 kernel_size: int,
                 dilation_rate: int,
                 upsample_rates: List[int],
                 upsample_kernel_sizes: List[int],
                 upsample_initial_channel: int,
                 resblock_kernel_sizes: List[int],
                 resblock_dilation_sizes: List[List[int]],
                 eps: float, 
                 dropout_rate: float) -> None:
        super().__init__()
        self.d_model = d_model

        self.prior_encoder = PriorEncoder(
            phoneme_size=phoneme_size,
            n=n,
            d_model=d_model,
            heads=heads,
            eps=eps,
            dropout_rate=dropout_rate
        )

        self.flow = ResidualCouplingBlock(
            channels=d_model,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            n_layers=n_layers
        )
        self.posterior_encoder = PosteriorEncoder(
            n_layers=n_layers,
            in_channels=n_mels,
            out_channels=d_model,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate
        )

        self.decoder = Generator(
            n_mel_channels=d_model,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            upsample_initial_channel=upsample_initial_channel,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x, x_mean, x_logs = self.prior_encoder(x)

        y, y_mean, y_logs = self.posterior_encoder(y)
        z = self.flow(y)

        signal = self.decoder(z)
        return signal