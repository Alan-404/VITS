import numpy as np

def monotonic_alignment_search(latent_value: np.ndarray):
    t_x, t_y = latent_value.shape

    Q = float('-inf') * np.ones([t_x, t_y])
    path = np.zeros([t_x, t_y])

    for y in range(t_y):
        for x in range(np.max(0, t_x + y - t_y), np.min(t_x, y+1)):
            if y == 0:
                Q[x, 0] = latent_value[x, 0]
            else:
                if x == 0:
                    v_prev = float('-inf')
                else:
                    v_prev = Q[x-1, y-1]
                
                v_current = Q[x -1, y-1]
                Q[x, y] = latent_value[x, y] + np.max(v_prev, v_current)

    index = t_x - 1
    for y in range(t_y - 1, -1, -1):
        path[index, y] = 1
        if index != 0 and (index == y or Q[index, y-1] < Q[index-1, y-1]):
            index -= 1

    return path