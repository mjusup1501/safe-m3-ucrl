from collections import deque, namedtuple
import random
from typing import Tuple

import torch
from torch.utils.data import Dataset


class DynamicsDataset(Dataset):
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


Transition = namedtuple("Transition", ("states", "mu", "actions", "next_states", "next_mu", "reward"))


def concat_inputs(
    states: torch.Tensor,
    mu: torch.Tensor,
    actions: torch.Tensor,
):
    return torch.cat(
        [states, mu.reshape(1, -1).expand(states.shape[0], -1), actions], dim=1
    )


def concat_inputs_(
    states: torch.Tensor,
    mu: torch.Tensor,
):
    return torch.cat([states, mu.reshape(1, -1).expand(states.shape[0], -1)], dim=1)


def normalize_states(
    states: torch.Tensor, state_space: Tuple[float, float], polar: bool = False
) -> torch.Tensor:
    # min-max normalization
    states_ = (states - state_space[0]) / (state_space[1] - state_space[0])
    if not polar:
        return states_
    else:
        states_ = 0.5 * (states_ + 1.0)
        return torch.cat(
            (
                torch.sin(2 * torch.pi * states_),
                torch.cos(2 * torch.pi * states_),
            ),
            dim=1,
        )


def normalize_mus(
    mus: torch.Tensor,
    gmm: bool = False,
) -> torch.Tensor:
    if len(mus.shape) == 1:
        mus = mus.reshape(1, -1)
    if gmm:
        fixed_states = (
            torch.arange(0, mus.shape[1], 1).reshape(1, -1).detach().to(mus.device)
            / mus.shape[1]
        ) + 0.5 / mus.shape[1]
        with torch.no_grad():
            p = mus / mus.sum(dim=1, keepdim=True)
            avg = fixed_states @ p.T
            std_dev = torch.sqrt(
                ((fixed_states - avg.T) ** 2 * p).sum(dim=1, keepdim=True)
            )
        return torch.cat((avg.T, std_dev), dim=1)
    return mus / mus.sum(dim=1, keepdim=True)


def normalize_actions(
    actions: torch.Tensor, action_space: Tuple[float, float]
) -> torch.Tensor:
    return (actions - (action_space[1] + action_space[0]) / 2) / (
        (action_space[1] - action_space[0]) / 2
    )


def denormalize_states(
    states: torch.Tensor, state_space: Tuple[float, float]
) -> torch.Tensor:
    return (states * (state_space[1] - state_space[0]) / 2) + (
        state_space[1] + state_space[0]
    ) / 2


def denormalize_actions(
    actions: torch.Tensor, action_space: Tuple[float, float]
) -> torch.Tensor:
    return (actions * (action_space[1] - action_space[0]) / 2) + (
        action_space[1] + action_space[0]
    ) / 2


def normalize_inputs(
    states: torch.Tensor,
    mu: torch.Tensor,
    actions: torch.Tensor,
    state_space: Tuple[float, float],
    action_space: Tuple[float, float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states_ = normalize_states(states, state_space)
    mu_ = normalize_mus(mu)
    actions_ = normalize_actions(actions, action_space)
    return (states_, mu_, actions_)
