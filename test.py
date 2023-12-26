#%%
from src.vits import VITS
import torch
import torchsummary
import math
from preprocessing.processor import VITSProcessor
#%%
processor = VITSProcessor(
    phoneme_path='./phonemes/phoneme.json'
)
# %%

# %%
model = VITS(
    phoneme_size=len(processor.dictionary),
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
#%%
text = "hiện nay vị trí của bàn thờ thường được đặt trong phòng riêng ở tầng trên cùng của nhà"
path = "C:\src\dataset\InfoRe_Technology/00000.wav"
#%%
tokens = processor.text2digit(text)
# %%
tokens
#%%
mel = processor.log_mel_spectrogram(processor.load_audio(path))
# %%
mel.shape
# %%
x = tokens.unsqueeze(0)
y = mel.unsqueeze(0)
#%%
# signal = model(x, y)

# %%
x.shape
# %%
y.shape
# %%
hidden, x_mean, x_logs = model.prior_encoder(x)
# %%
hidden.shape
# %%
z, y_mean, y_logs = model.posterior_encoder(y)
# %%
z_p = model.flow(z)
# %%
with torch.no_grad():
    s_p_sq_r = torch.exp(-2 * x_logs)
# %%
s_p_sq_r.shape
# %%
neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - x_logs, [1], keepdim=True)
# %%
neg_cent1.shape
# %%
neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r)
# %%
neg_cent3 = torch.matmul(z_p.transpose(1, 2), (x_mean * s_p_sq_r))
# %%
neg_cent4 = torch.sum(-0.5 * (x_mean ** 2) * s_p_sq_r, [1], keepdim=True)
# %%
neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
# %%
neg_cent.shape
# %%
neg_cent = torch.detach(neg_cent)
# %%
neg_cent.shape
# %%
from src.utils.mas import monotonic_alignment_search
# %%
out = monotonic_alignment_search(neg_cent[0].transpose(0,1))
# %%
out.shape
# %%
out[2].shape
# %%
out
# %%
out = torch.tensor(out).unsqueeze(0)
# %%
out.shape
# %%
w = torch.sum(out)
# %%
w = w.unsqueeze(0)
# %%
w.shape
# %%

# %%
dur_out = model.dp(hidden, w=w)
# %%
x.size()
# %%
