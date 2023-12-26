import re
import torch


def find_best_ckpt(policy_ckpt_dir):
    best_episodes = list(policy_ckpt_dir.glob("policy_best*[0-9].pt"))
    best_episode = 0
    for episode in best_episodes:
        episode = int(re.findall(r'\d+', episode.name)[0])
        if episode > best_episode:
            best_episode = episode
    best_ckpt = f"policy_best{best_episode}.pt"

    return best_ckpt


def find_last_ckpt(policy_ckpt_dir):
    final_episode = list(policy_ckpt_dir.glob("policy_final.pt"))
    if final_episode:
        return "policy_final.pt"
    last_episodes = list(policy_ckpt_dir.glob("policy*[0-9].pt"))
    last_episode = 0
    for episode in last_episodes:
        episode = int(re.findall(r'\d+', episode.name)[0])
        if episode > last_episode:
            last_episode = episode
    last_ckpt = f"policy{last_episode}.pt"

    return last_ckpt


def index_to_cell(index, n):
    i = index % n
    j = index // n
    return (i, j)


def cells_to_index(cells, n):
    if cells.ndim == 2:
        idx = cells[:, 1] * n + cells[:, 0]
    elif cells.ndim == 3:
        idx = cells[:, :, 1] * n + cells[:, :, 0]
        
    return idx.long()


def states_to_cell(states, linspace_x, linspace_y):
    # Use digitize to find the indices of the cells that contains given states
    buckets_x = torch.bucketize(states[:, 0], linspace_x) - 1
    buckets_y = torch.bucketize(states[:, 1], linspace_y) - 1
    cells = torch.stack([buckets_x, buckets_y], dim=1)
    # Bucketize returns -1 for states equal to 0 so we overwrite it
    mask_x = cells[:, 0] == -1
    cells[:, 0][mask_x] = 0
    mask_y = cells[:, 1] == -1
    cells[:, 1][mask_y] = 0

    return cells


def reverse_min_max(x, min, max, min_data, max_data):
    # Scale the data
    x_scaled = (x - min_data) / (max_data - min_data)
    # Reverse scaling
    x_reversed = x_scaled * (max - min) + min

    return x_reversed



