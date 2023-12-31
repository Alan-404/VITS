import torch
import torch.nn as nn
from src.modules.encoder import PriorEncoder, PosteriorEncoder
from src.modules.decoder import Generator
from src.utils.affine_coupling import ResidualCouplingBlock
from src.modules.duration import StochasticDurationPredictor, DurationPredictor
from src.utils.masking import generate_mask
from src.utils.mas import monotonic_alignment_search_batch
from typing import Optional
from typing import List
import math

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
                 use_sdp: bool,
                 dropout_rate: float,
                 num_speakers: Optional[int] = None,
                 gin_channels: Optional[int]=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.use_sdp = use_sdp

        if num_speakers is not None and num_speakers not in [0,1] and gin_channels is not None and gin_channels != 0:
            self.speaker_embedding = nn.Embedding(num_embeddings=num_speakers, embedding_dim=gin_channels)

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

        if use_sdp:
            self.dp = StochasticDurationPredictor(channels=d_model, filter_channels=hidden_channels, kernel_size=3, dropout_rate=0.5, gin_channels=gin_channels)
        else:
            self.dp = DurationPredictor(in_channels=d_model, filter_channels=hidden_channels, kernel_size=3, dropout_rate=0.5, gin_channels=gin_channels)

        self.decoder = Generator(
            n_mel_channels=d_model,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            upsample_initial_channel=upsample_initial_channel,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, x_lengths: Optional[torch.Tensor] = None, y_lengths: Optional[torch.Tensor] = None, g: Optional[torch.Tensor] = None):
        x_mask = (generate_mask(x_lengths) == False)
        x, x_mean, x_logs = self.prior_encoder(x, x_mask)

        y_mask = (generate_mask(y_lengths) == False)
        y, y_mean, y_logs = self.posterior_encoder(y, y_mask)
        z = self.flow(y, g=g)

        with torch.no_grad():
            s_p_sq_r = torch.exp(-2 * x_logs)
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - x_logs, [1], keepdim=True) # [b, 1, t_s]
            neg_cent2 = torch.matmul(-0.5 * (z ** 2).transpose(1, 2), s_p_sq_r) # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent3 = torch.matmul(z.transpose(1, 2), (x_mean * s_p_sq_r)) # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent4 = torch.sum(-0.5 * (x_mean ** 2) * s_p_sq_r, [1], keepdim=True) # [b, 1, t_s]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
            attn = monotonic_alignment_search_batch(neg_cent, x_lengths, y_lengths)
        
        w = torch.sum(attn, dim=2).unsqueeze(1).type(torch.FloatTensor)

        if self.use_sdp:
            l_length = self.dp(x, w, g=g) / torch.sum(x_mask)
        else:
            logw_ = torch.log(w + 1e-6)
            logw = self.dp(x, g=g)
            l_length = torch.sum((logw - logw_)**2, [1,2]) / torch.sum(x_mask)

        attn = attn.transpose(1, 2)
        x_mean = torch.matmul(attn, x_mean.transpose(1, 2)).transpose(1, 2)
        x_logs = torch.matmul(attn, x_logs.transpose(1, 2)).transpose(1, 2)

        signal = self.decoder(z)

        return signal, l_length, attn, (y, z, x_mean, x_logs, y_mean, y_logs)