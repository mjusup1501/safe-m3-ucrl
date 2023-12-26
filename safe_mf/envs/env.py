from abc import ABC, abstractmethod
from typing import Callable, Tuple, Optional
import torch
from safe_mf.models.policy import MFPolicy


class Env(ABC):
    def __init__(
        self,
        state_dim: int,
        mu_dim: int,
        action_dim: int,
        device=torch.device,
    ) -> None:
        self.state_dim = state_dim
        self.mu_dim = mu_dim
        self.action_dim = action_dim
        self.device = device

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def reward(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        mu: torch.Tensor,
        safety: bool,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def integrated_reward(
        self,
        mu: torch.Tensor,
        actions: torch.Tensor,
        constraint_function: Callable,
        step: Optional[int] = None,
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def step(
        self, 
        policy: MFPolicy, 
        step: Optional[int] = None, 
        known_dynamics: bool=True
    ) -> Tuple[torch.Tensor, float]:
        raise NotImplementedError
