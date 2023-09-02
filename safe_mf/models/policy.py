import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from safe_mf.utils.distributions import shifted_uniform

from safe_mf.utils.data import (
    normalize_mus,
    normalize_states,
)


class RandomPolicy(nn.Module):
    def __init__(
        self,
        action_dim: int,
        action_space: Tuple[float, float] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()

        self.action_dim = action_dim
        self.action_space = action_space
        self.device = device

    def forward(self, states: torch.Tensor, mu: torch.Tensor, explore: bool) -> torch.Tensor:
        """Computes the action given the state and the mean field

        Args:
            mu (torch.Tensor): [mu_dim]
            explore (bool): for consistency with MFPolicy

        Returns:
            torch.Tensor: [mu_dim * action_dim]
        """
        size = (states.shape[0], self.action_dim)
        low = self.action_space[0]
        high = self.action_space[1]
     
        return shifted_uniform(low, high, size, device=self.device)


class OptimalSwarm1DPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, states: torch.Tensor, mu: torch.Tensor, explore: Optional[float] = None):
        return 2 * torch.pi * torch.cos(2 * torch.pi * states)


class MFPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        mu_dim: int,
        action_dim: int,
        hallucinated_control: bool,
        hidden_dims: List[int],
        state_space: Tuple[float, float],
        action_space: Tuple[float, float],
        gmm: bool,
        polar: bool,
    ) -> None:
        super().__init__()
        assert len(hidden_dims) > 0
        self.mu_dim = mu_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hallucinated_control = hallucinated_control
        self.state_space = state_space
        self.action_space = action_space
        self.gmm = gmm
        self.polar = polar

        final_dim = state_dim + action_dim if hallucinated_control else action_dim
        s_dim = 2 * state_dim if self.polar else state_dim
        m_dim = 2 if self.gmm else self.mu_dim
        self.model = [nn.Linear(s_dim + m_dim, hidden_dims[0])]
        dims = hidden_dims + [final_dim]
        for i in range(len(dims) - 1):
            self.model += [nn.LeakyReLU(), nn.Linear(dims[i], dims[i + 1])]
        self.model += [nn.Tanh()]
        self.model = nn.Sequential(*self.model)
        self.reset_parameters()


    def reset_parameters(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=math.sqrt(2))
                nn.init.zeros_(layer.bias)


    def forward(
        self, states: torch.Tensor, mu: torch.Tensor, explore: Optional[float] = None
    ) -> torch.Tensor:
        """Computes the actions given the mean field

        Args:
            states (torch.Tensor): [m, state_dim]
            mu (torch.Tensor): [n, mu_dim]

        Returns:
            torch.Tensor: [m, act_dim] or [n, act_dim + state_dim]
        """
        if states.shape[0] == mu.shape[0]:
            inputs = torch.cat(
                (
                    normalize_states(states, self.state_space, polar=self.polar),
                    normalize_mus(mu, gmm=self.gmm),
                ),
                dim=1,
            )
        elif mu.shape[0] == 1:
            inputs = torch.cat(
                (
                    normalize_states(states, self.state_space, polar=self.polar),
                    normalize_mus(mu, gmm=self.gmm).expand(states.shape[0], -1),
                ),
                dim=1,
            )
        else:
            norm_states = normalize_states(states, self.state_space, polar=self.polar)
            norm_states = norm_states.repeat(mu.shape[0], 1, 1)
            norm_mu = normalize_mus(mu, gmm=self.gmm)
            norm_mu = norm_mu.unsqueeze(1).repeat(1, norm_states.shape[1], 1)
            inputs = torch.cat((norm_states, norm_mu), dim=2)
            inputs = inputs.reshape(-1, inputs.shape[2])
        
        return self.model(inputs)