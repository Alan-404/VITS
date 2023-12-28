import torch
import torch.nn as nn
import torch.nn.functional as F

from ignite.engine import Engine, Events


from src.vits import VITS
from src.modules.discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator

from preprocessing.processor import VITSProcessor

processor = VITSProcessor(
    phoneme_path='./phonemes/phoneme.json'
)

model = VITS(
    phoneme_size=len(processor.dictionary.get_itos()),
    n=6,
    d_model=192,
    heads=2,
    hidden_channels=128,
    kernel_size=3,
    n_mels=80,
    n_layers=4,
    upsample_rates=[8,8,2,2],
    upsample_kernel_sizes=[16,16,4,4],
    upsample_initial_channel=512,
    resblock_kernel_sizes=[3,7,11],
    resblock_dilation_sizes=[[1,3,5], [1,3,5], [1,3,5]],
    eps=1e-5,
    dilation_rate=2,
    use_sdp=True,
    dropout_rate=0.1
)

multi_period_discriminator = MultiPeriodDiscriminator()
multi_scale_discriminator = MultiScaleDiscriminator()