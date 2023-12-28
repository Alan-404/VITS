#%%
from src.vits import VITS
import torch
from preprocessing.processor import VITSProcessor
from src.utils.masking import generate_mask
#%%
processor = VITSProcessor(
    phoneme_path='./phonemes/phoneme.json'
)
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
text = [
    "hiện nay vị trí của bàn thờ thường được đặt trong phòng riêng ở tầng trên cùng của nhà",
    "để cậu nhỏ hướng lên hay hướng xuống khi mặc quần lót"
]
paths = [
    "D:\datasets/tts\infore_tech/audio/00000.wav",
    "D:\datasets/tts\infore_tech/audio/00001.wav"
]
#%%
