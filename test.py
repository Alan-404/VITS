#%%
from src.utils.affine_coupling import ResidualCouplingBlock
from src.vits import VITS
import torch
import torchsummary
# %%
model = VITS(
    phoneme_size=99,
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
    dropout_rate=0.1
)
# %%
phonemes = torch.tensor([[1,4,7,8]])
mel = torch.rand((1, 80, 140))
# %%
out = model(phonemes, mel)
# %%
out
# %%
len(out)
# %%
out.shape
# %%
from src.modules.flow import ConvFlow
# %%
import torch
# %%
layer = ConvFlow(
    in_channels=192,
    filter_channels=128,
    kernel_size=3,
    n_layers=4,
)
# %%
a = torch.rand((1, 192, 147))
# %%
out = layer(a)
# %%
out[0]
# %%
rev = layer(out[0], reverse=False)
# %%
rev
# %%
a
# %%
