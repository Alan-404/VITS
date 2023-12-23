#%%
import torch
import torchsummary
from src.modules.duration import DurationPredictor
from src.modules.encoder import PosteriorEncoder
# %%
layer = PosteriorEncoder(n_layers=3, in_channels=80, out_channels=192, hidden_channels=128, kernel_size=3)
# %%
a = torch.rand((1, 80, 140))
# %%
out = layer(a)
# %%
out
# %%
