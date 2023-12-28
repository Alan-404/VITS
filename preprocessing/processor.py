import json
from typing import Any, Optional
import numpy as np
import re
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from pydub import AudioSegment
from torchtext.vocab import Vocab, vocab as create_vocab

MAX_AUDIO_VALUE = 32768

class VITSProcessor:
    def __init__(self, phoneme_path: Optional[str] = None, sample_rate: int = 22050, n_mel_channels: int = 80, fft_size: int = 1024, window_size: int = 1024, hop_length: int = 256, fmax: float = 8000.0, fmin: float = 0.0, htk: bool = True, bos_token: str = "<s>", eos_token: str = "</s>", space_token: str = "|", padding_token: str = "<pad>") -> None:
        # Text
        self.vowels = []
        self.consonants = []
        self.compound_consonants = []
        self.double_vowels = []
        self.triple_vowels = []
        self.marks = []

        self.break_tokens = [".", ","]

        self.dictionary = self.load_phonemes(phoneme_path, padding_token, bos_token, eos_token, space_token)

        self.space_str = space_token

        if self.dictionary is not None:
            self.padding_token = self.dictionary.__getitem__(padding_token)
            self.bos_token = self.dictionary.__getitem__(bos_token)
            self.eos_token = self.dictionary.__getitem__(eos_token)
            self.space_token = self.dictionary.__getitem__(space_token)
        

        # Audio
        self.sample_rate = sample_rate
        self.n_mel_channels = n_mel_channels
        self.fft_size = fft_size
        self.window_size = window_size
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.htk = htk

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=fft_size,
            win_length=window_size,
            hop_length=hop_length,
            f_min=fmin,
            f_max=fmax,
            n_mels=n_mel_channels
        )

    def load_audio(self, path: str):
        audio = AudioSegment.from_file(path).set_frame_rate(self.sample_rate).get_array_of_samples()
        signal = torch.tensor(audio) / MAX_AUDIO_VALUE
        signal = self.standard_normalize(signal)
        return signal
    
    def clean_text(self, text: str):
        text = text.lower()
        text = re.sub("\s\s+", " ", text)

        return text
    
    def spectral_normalize(self, x, C=1, clip_val=1e-5):
        return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)
    
    def standard_normalize(self, signal: torch.Tensor, eps: float = 1e-7):
        return (signal - signal.mean()) / torch.sqrt(signal.var() + eps)
    
    def spec_normalize(self, mel: torch.Tensor, clip_val: float = 1e-5, C: int = 1):
        return torch.log(torch.clamp(mel, min=clip_val) * C)

    def log_mel_spectrogram(self, signal: torch.Tensor):
        mel_spec = self.mel_transform(signal)

        log_mel = self.spec_normalize(mel_spec)

        return log_mel
    
    def generate_mask(self, lengths: np.ndarray, max_len: Optional[int] = None):
        if max_len is None:
            max_len = np.max(lengths)

        mask = []
        for length in lengths:
            mask.append(torch.tensor(np.array([True] * length + [False] * (max_len - length))).type(torch.bool))

        return torch.stack(mask)

    def load_phonemes(self, path: str, padding_token: str, bos_token: str, eos_token: str, space_token: str):
        data = json.load(open(path, encoding='utf-8'))

        self.vowels = data['vowel']
        self.consonants = data['consonant']
        self.compound_consonants = data['compound_consonant']
        self.double_vowels = data['double_vowel']
        self.triple_vowels = data['triple_vowel']
        self.marks = data['mark']

        dictionary = dict()
        count = 0
        for key in data.keys():
            for item in data[key]:
                count += 1
                dictionary[item] = count

        return Vocab(vocab=create_vocab(
            dictionary,
            specials=[padding_token, bos_token, eos_token, space_token]
        ))
    
    
    def text2digit(self, text: str):
        phonemes = self.text2phonemes(text)
        # digits = [self.bos_token] + self.dictionary(phonemes) + [self.eos_token]
        digits = self.dictionary(phonemes)
        return torch.tensor(digits)
    
    def text2phonemes(self, text: str):
        text = self.clean_text(text)
        words = text.split(" ")
        phonemes = []

        for index, word in enumerate(words):
            phonemes += self.word2phonemes(word)
            if index != len(words) - 1:
                phonemes.append(self.space_str)
        
        return phonemes
    
    def find_single_char(self, char: str):
        if char in self.consonants or char in self.vowels or char in self.marks:
            return [char]
        return None
    
    def word2phonemes(self, text: str,  n_grams: int = 3):
        if len(text) == 1:
            return self.find_single_char(text)
        phonemes = []
        start = 0
        if len(text) - 1 < n_grams:
            n_grams = len(text) - 1
        num_steps = n_grams
        while start < len(text):
            found = True
            item = text[start:start + num_steps]

            if item in self.marks:
                phonemes.append(item)
            elif item in self.triple_vowels:
                phonemes.append(item)
            elif item in self.double_vowels:
                phonemes.append(item)
            elif item in self.compound_consonants:
                phonemes.append(item)
            elif item in self.consonants:
                phonemes.append(item)
            elif item in self.vowels:
                phonemes.append(item)
            else:
                found = False

            if found:
                start += num_steps
                if len(text[start:]) < n_grams:
                    num_steps = len(text[start:])
                else:
                    num_steps = n_grams
            else:
                num_steps -= 1

        return phonemes
    
    def __call__(self, sentences: list, max_len: Optional[int] = None) -> Any:
        digits = []
        lengths = []

        for sentence in sentences:
            digit = self.text2digit(sentence)
            digits.append(digit)
            lengths.append(len(digit))

        if max_len is None:
            max_len = np.max(lengths)

        tokens = []
        for index, digit in enumerate(digits):
            tokens.append(F.pad(digit, (0, max_len - lengths[index]), mode='constant', value=self.padding_token))
        
        return torch.stack(tokens), torch.tensor(lengths)
    
    def mel_spectrogize(self, signals: list, max_len: Optional[int] = None, return_attention_mask: bool = False):
        if max_len is None:
            max_len = np.max([len(signal) for signal in signals])

        mels = []
        mel_lengths = []

        for signal in signals:
            signal_length = len(signal)

            log_mel = self.log_mel_spectrogram(F.pad(signal, (0, max_len - signal_length), mode='constant', value=0.0))

            mels.append(log_mel)
            mel_lengths.append((signal_length // self.hop_length) + 1)

        return torch.stack(mels), torch.tensor(mel_lengths)
