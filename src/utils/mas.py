import numpy as np
import torch

def monotonic_alignment_search_batch(hidden: torch.Tensor, x_lengths: torch.Tensor, y_lengths: torch.Tensor):
    hidden = hidden.transpose(1,2)
    paths = []
    for idx, item in enumerate(hidden):
        x_len = x_lengths[idx]
        y_len = y_lengths[idx]

        x_num_pad = item.shape[0] - x_len
        y_num_pad = item.shape[1] - y_len

        path = monotonic_alignment_search(item[:x_lengths[idx], :y_lengths[idx]])
        path = torch.concat([path, torch.zeros(x_num_pad, path.shape[1])], dim=0)
        path = torch.concat([path, torch.zeros(path.shape[0], y_num_pad)], dim=1)

        paths.append(path)
        
    return torch.stack(paths).type(torch.FloatTensor)

def monotonic_alignment_search(value: torch.Tensor):
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
    return torch.tensor(path)