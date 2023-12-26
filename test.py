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
import torch
from src.modules.flow import ConvFlow
# %%
layer = ConvFlow(
    in_channels=192,
    filter_channels=128,
    kernel_size=3,
    n_layers=4
)
# %%
a = torch.rand((1, 192, 140))
# %%
out, _ = layer(a)
# %%
out.shape
# %%
out
# %%
rev = layer(out, reverse=True)
# %%
rev
# %%
a
# %%
out
# %%
import torch
# %%
a = torch.rand((1, 50, 140))
# %%
a.gather(dim=0)
# %%
