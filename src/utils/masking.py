import torch
import numpy as np

def generate_mask(lengths: torch.Tensor):
    max_length = torch.max(lengths)

    mask = []
    for length in lengths:
        mask.append(torch.tensor(np.array([True] * length + [False] * (max_length - length))))

    return torch.stack(mask)