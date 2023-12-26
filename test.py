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
path = "D:\datasets/tts\infore_tech/audio/00000.wav"
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
# %%
out, w = model(x, y)
# %%
dur_out = model.dp(out, w)
# %%
dur_out
# %%
