from typing import Union
import torch
import numpy as np


def entropy(mu: Union[torch.Tensor, np.ndarray]):
    if isinstance(mu, np.ndarray):
        mu = torch.tensor(mu)
    if len(mu.shape) > 1:
        mu = mu.flatten()
    mu_ = mu.clone()
    return -torch.sum(torch.log(mu_ + 1e-30) * mu_)


def entropic_constraint(mu: torch.Tensor, c: float):
    return entropy(mu) - c


def max_entropy(mu_dim: int):
    return torch.log(torch.tensor(mu_dim))


def kld(true: torch.Tensor, pred: torch.Tensor):
    true = true / true.sum(dim=1)
    pred = pred / pred.sum(dim=1)
    return torch.sum(true * (torch.log(true + 1e-30) - torch.log(pred + 1e-30)), dim=1)
