import numpy as np
import torch

def monotonic_alignment_search_batch(hidden: torch.Tensor):
    paths = []
    for item in hidden:
        paths.append(monotonic_alignment_search(item))
    return torch.stack(paths)

def monotonic_alignment_search(value: torch.Tensor):
    value = value.transpose(0,1)
    t_x, t_y = value.shape # [text_length, letent_variable_length]
    path = np.zeros([t_x, t_y])
    # A cache to store the log-likelihood for the most likely alignment so far.
    Q = float('-inf') * np.ones([t_x, t_y])
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            if y == 0: # Base case. If y is 0, the possible x value is only 0.
                Q[x, 0] = value[x, 0]
            else:
                if x == 0:
                    v_prev = float('-inf')
                else:
                    v_prev = Q[x-1, y-1]
                v_cur = Q[x, y-1]
                Q[x, y] = value[x, y] + max(v_prev, v_cur)
    # Backtrack from last observation.
    index = t_x - 1
    for y in range(t_y - 1, -1, -1):
        path[index, y] = 1
        if index != 0 and (index == y or Q[index, y-1] < Q[index-1, y-1]):
            index = index - 1
    return path