import torch
import torch.nn as nn

from typing import Optional, List

from model.vits import VITS
from model.modules.encoder import PosteriorEncoder
from model.utils.masking import generate_mask
from model.modules.mas.search import find_path
from processing.processor import VITSProcessor

class VITSModule(nn.Module):
    def __init__(self,
                 processor: VITSProcessor,
                 n_mel_channels: int,
                 d_model: int = 192,
                 n_blocks: int = 6,
                 n_heads: int = 2,
                 kernel_size: int = 3,
                 hidden_channels: int = 192,
                 upsample_initial_channel: int = 512,
                 upsample_rates: List[int] = [8,8,2,2],
                 upsample_kernel_sizes: List[int] = [16,16,4,4],
                 resblock_kernel_sizes: List[int] = [3,7,11],
                 resblock_dilation_sizes: List[List[int]] = [[1,3,5], [1,3,5], [1,3,5]],
                 dropout_p: float = 0.1,
                 segment_size: Optional[int] = 8192,
                 n_speakers: Optional[int] = None,
                 gin_channels: Optional[int] = None) -> None:
        super().__init__()

        self.processor = processor

        self.posterior_encoder = PosteriorEncoder(
            n_mel_channels=n_mel_channels,
            hidden_channels=d_model,
            out_channels=hidden_channels,
            kernel_size=5,
            n_layers=16,
            dilation_rate=1,
            gin_channels=gin_channels
        )

        self.vits = VITS(
            token_size=len(processor.dictionary),
            d_model=d_model,
            n_blocks=n_blocks,
            n_heads=n_heads,
            kernel_size=kernel_size,
            hidden_channels=hidden_channels,
            upsample_initial_channel=upsample_initial_channel,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            dropout_p=dropout_p,
            segment_size=segment_size,
            n_speakers=n_speakers,
            gin_channels=gin_channels
        )

    def forward(self, x: torch.Tensor, mels: torch.Tensor, x_lengths: Optional[torch.Tensor] = None, mel_lengths: Optional[torch.Tensor] = None, sid: Optional[torch.Tensor] = None):
        # (Optional) Embedding Speaker
        g = None
        if sid is not None:
            g = self.vits.speaker_embedding(sid).unsqueeze(-1) # (batch_size, gin_channels, 1)
        
        # Extract Posterior Distribution
        z_mask = None
        if mel_lengths is not None:
            z_mask = generate_mask(mel_lengths)
        z, m_q, logs_q = self.posterior_encoder(mels, z_mask, g)

        # Main VITS Flow
        o, l_length, sliced_indexes, x_mask, z_p, m_p, logs_p = self.vits(
            x=x,
            z=z,
            x_lengths=x_lengths,
            z_mask=z_mask,
            g=g
        )

        return o, l_length, sliced_indexes, x_mask, z_mask, z, z_p, m_p, logs_p, m_q, logs_q