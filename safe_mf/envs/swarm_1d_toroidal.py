
import math
from typing import Any, Callable, Mapping, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import torch
import wandb

from safe_mf.envs.env import Env
from safe_mf.models.ensemble import UnknownDynamics
from safe_mf.models.policy import MFPolicy
from torch.distributions import Normal
from safe_mf.utils.entropy import entropy, max_entropy
from safe_mf.utils.distributions import shifted_uniform


class Swarm1DToroidal(Env):
    def __init__(
        self,
        mu_dim: int,
        delta: float,
        control_std: float,
        constraint_lipschitz: float,
        barrier_lambda: float = 1,
        dynamics_cfg: Mapping[str, Any] = None,
        constraint_function: Callable = None,
        state_space: Tuple[float, float] = (0.0, 1.0),
        action_space: Tuple[float, float] = (-10.0, 10.0),
        device: torch.device = torch.device("cpu"),
        dynamics_ckpt: str = None,
        exec_type: str = "train",
        num_agents: int = 1,
        reward_type: str = "current_mu",
    ) -> None:
        super().__init__(state_dim=1, mu_dim=mu_dim, action_dim=1, device=device)
        self.state_space = state_space
        self.action_space = action_space
        self.delta = delta
        self.control_std = control_std * math.sqrt(self.delta) if control_std else None
        self.constraint_lipschitz = constraint_lipschitz
        self.barrier_lambda = barrier_lambda
        self.num_intervals = self.mu_dim
        self.exec_type = exec_type
        self.num_agents = num_agents
        self.reward_type = reward_type
        self.constraint_function = constraint_function
        if self.control_std is None:
            self.normal_control = 0.0
        else:
            self.normal_control = Normal(loc=0.0, scale=self.control_std)

        with torch.no_grad():
            self.cell_centers_1d = (
                torch.arange(0, self.num_intervals, 1, device=self.device)
                .reshape(1, -1)
                / self.num_intervals
            ) + 0.5 / self.num_intervals
            self.cell_centers = torch.cartesian_prod(*[self.cell_centers_1d.squeeze(0)] * self.state_dim)
            extended_cell_centers_1d = (
                torch.arange(
                    -1 * self.num_intervals, 
                    2 * self.num_intervals, 
                    1, device=self.device
                ).reshape(1, -1)
                / self.num_intervals
            ) + 0.5 / self.num_intervals
            self.upper = extended_cell_centers_1d + (0.5 / self.num_intervals)
            self.lower = extended_cell_centers_1d - (0.5 / self.num_intervals)
            self.upper[0][-1] += 10 ** 4 
            self.lower[0][0] -= 10 ** 4

        if dynamics_ckpt is not None:
            self.dynamics = torch.load(dynamics_ckpt, map_location=self.device)
        elif dynamics_cfg is not None:
            self.dynamics = UnknownDynamics(
                **dynamics_cfg,
                state_dim=self.state_dim,
                mu_dim=self.mu_dim,
                action_dim=self.action_dim,
                state_space=self.state_space,
                action_space=self.action_space,
                device=self.device,
            )
        else:
            self.dynamics = None

        self.reset()

    def reset(self) -> None:
        with torch.no_grad():
            self.mu = torch.ones(size=(self.mu_dim,), device=self.device).reshape(1, -1) / self.mu_dim
            self.mu = torch.nn.functional.normalize(self.mu, p=1)
            # Representative agent
            self.ra_states = shifted_uniform(
                                low=self.state_space[0], 
                                high=self.state_space[1],
                                size=(self.num_agents, self.state_dim), 
                                device=self.device
                                )

        if self.constraint_function is not None and self.dynamics is not None:
            self.acc_uncertainty = 0.0
        else:
            self.acc_uncertainty = None


    def get_next_states(self, current_states, actions, mf_transition: bool=False):
        next_states = current_states + actions * self.delta 
        if mf_transition:
            pass
        else:
            # We use it to train neural nets on the extended states space and mod it afterwards
            noise = self.normal_control.sample(next_states.shape).to(self.device)
            next_states += noise
        next_states = (
            torch.remainder(next_states, self.state_space[1] - self.state_space[0])
            + self.state_space[0]
        )

        return next_states


    def get_hallucinated_next_states(self, means, actions, mf_transition: bool=False):
        next_states = means + actions
        if mf_transition:
            pass
        else:
            # We use it to train neural nets on the extended states space and mod it afterwards
            noise = self.normal_control.sample(next_states.shape).to(self.device)
            next_states += noise
        next_states = (
            torch.remainder(next_states, self.state_space[1] - self.state_space[0])
            + self.state_space[0]
        )

        return next_states


    def _true_dynamics(
        self,
        states: torch.Tensor,  # [?, state_dim]
        mu: torch.Tensor,  # [n, mu_dim]
        policy: MFPolicy,
        exploration: Optional[float] = None,
        grid_actions: Optional[torch.Tensor] = None,
        mf_transition: bool = False
    ) -> torch.Tensor:
        act_dim = self.action_dim
        actions = policy(states.reshape(-1, 1), mu, exploration)
        true_actions = actions[:, :act_dim].reshape(1, -1)
        next_states = self.get_next_states(states.reshape(1, -1), true_actions, mf_transition)

        return true_actions, next_states


    def _approximate_dynamics(
        self,
        states: torch.Tensor,  # [?, state_dim]
        mu: torch.Tensor,  # [1, mu_dim]
        policy: MFPolicy,
        exploration: Optional[float] = None,
        grid_actions: Optional[torch.Tensor] = None,
        mf_transition: bool = False
    ) -> torch.Tensor:
        act_dim = self.action_dim
        actions = policy(states.reshape(-1, 1), mu, exploration)
        true_actions = actions[:, :act_dim].reshape(1, -1)
        hallucinated_actions = actions[:, act_dim:].reshape(1, -1)
        means, epi, alea = self.dynamics(
            states.reshape(-1, self.state_dim).detach(),
            mu.detach(), 
            true_actions.reshape(-1, self.action_dim)
        )
        epi = torch.sqrt(epi)
        epi = torch.diagonal(epi, dim1=1, dim2=2)
        hallucinated_actions = self.dynamics.beta * epi * hallucinated_actions.reshape(-1, self.state_dim)
        next_states = self.get_hallucinated_next_states(means.reshape(1, -1), hallucinated_actions.reshape(1, -1), mf_transition)

        return actions, next_states


    def positional_reward(self, states: torch.Tensor):
        s_ = 2 * torch.pi * states
        return 2 * torch.pi**2 * (torch.sin(s_) - torch.cos(s_) ** 2) + 2 * torch.sin(s_)


    def reward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        mu: torch.Tensor,
        safety: bool,
    ) -> torch.Tensor:
        """
        IMPORTANT: SHAPES MUST MATCH
        """
        assert states.shape == actions.shape and actions.shape == mu.shape
        if safety:
            return self.positional_reward(states) - 0.5 * actions**2
        else:
            return (
                self.positional_reward(states)
                - 0.5 * actions**2
                - torch.log(torch.clamp(mu, min=1e-30))
            )
        

    def integrated_reward(
        self,
        current_mu: torch.Tensor,
        actions: torch.Tensor,
        next_mu: torch.Tensor,
        step: Optional[int] = None,
        constraint_function: Callable = None,
    ) -> float:
        if self.reward_type == 'next_mu':
            mu = next_mu
        else:
            mu = current_mu
        main_reward = torch.sum(
            mu * self.reward(
                self.cell_centers.reshape(-1),
                actions.reshape(-1),
                mu.reshape(-1),
                safety=constraint_function is not None,
            )
        )
        if constraint_function is None:
            return main_reward
        else:
            constraint_value = constraint_function(current_mu)
            if self.dynamics is None:
                barrier = self.barrier_lambda * torch.nan_to_num(
                    torch.log(constraint_value),
                    nan=-100,
                    neginf=-100,
                )
            else:
                acc_uncertainty = self.acc_uncertainty
                barrier = self.barrier_lambda * torch.nan_to_num(
                     torch.log(
                        constraint_value
                        - self.constraint_lipschitz * acc_uncertainty
                    ),
                    nan=-100,
                    neginf=-100,
                )
            return main_reward + barrier


    def step(
        self,
        policy: MFPolicy,
        step: Optional[int] = None,
        known_dynamics: bool = True,
        exploration: Optional[float] = None,
        policy_training: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dynamics = self._true_dynamics if known_dynamics else self._approximate_dynamics
        ra_current_states = self.ra_states.clone()
        current_mu = self.mu.clone()

        grid_actions, grid_next_states = dynamics(
            self.cell_centers,
            current_mu,
            policy,
            exploration,
            mf_transition=True
        )
        true_grid_actions = grid_actions.reshape(self.mu_dim, -1)[:, :self.action_dim].reshape(1, -1)
        # Collect data for eval, i.e., when not training policy
        if not policy_training:
            with torch.no_grad():
                ra_actions, ra_next_states = dynamics(
                    ra_current_states,
                    current_mu,
                    policy,
                    exploration,
                    grid_actions,
                    mf_transition=False
                )
                # Reshaping needed for the case when we only have one agent
                ra_actions = ra_actions.reshape(self.num_agents, -1)
                ra_true_actions = ra_actions[:, :self.action_dim]
                ra_next_states = ra_next_states.reshape(self.num_agents, -1)
                self.ra_states = ra_next_states
       
        if self.acc_uncertainty is not None and not known_dynamics:
            self.acc_uncertainty += self.dynamics.expected_deviation(
                self.cell_centers,
                current_mu,
                true_grid_actions
            )

        if self.exec_type == "eval":
            next_mu = self._step_deterministic(ra_next_states, current_mu)
        else:
            next_mu = self._step_probabilistic(grid_next_states, current_mu)
        integrated_reward = self.integrated_reward(
            current_mu, true_grid_actions, next_mu, step, self.constraint_function
        )
        self.mu = next_mu

        if not policy_training:
            return (
                    ra_current_states, None, 
                    ra_next_states, current_mu, None, 
                    next_mu, ra_true_actions, integrated_reward
                )
        else:
            return integrated_reward


    def _apply_erf(
        self, mean: torch.Tensor
    ) -> torch.Tensor:
        return 0.5 * (
            torch.erf((self.upper - mean.T) / (self.control_std * math.sqrt(2)))
            - torch.erf((self.lower - mean.T) / (self.control_std * math.sqrt(2)))
        )


    def _apply_cdf(
        self, mean: torch.Tensor
    ) -> torch.Tensor:
        return self.normal_control.cdf(self.upper - mean.T) - self.normal_control.cdf(self.lower - mean.T)


    def _step_probabilistic(
        self,
        next_states: torch.Tensor,
        current_mu: torch.Tensor,
    ) -> torch.Tensor:
        """
        ERF based
        """
        probs = self._apply_cdf(next_states)
        multiplied_probs = (probs.T * current_mu).sum(dim=-1)
        next_mu = multiplied_probs.reshape(-1, self.mu_dim).sum(dim=0)
        next_mu = torch.nn.functional.normalize(next_mu.reshape(1, -1), p=1).squeeze()

        return next_mu.reshape(1, -1)


    def _step_deterministic(
        self,
        next_states: torch.Tensor,
        current_mu: torch.Tensor,
    ) -> torch.Tensor:
        """
        REQUIRES CPU
        """
        next_states = next_states.cpu()
        next_mu = torch.histogram(
            next_states.flatten(),
            bins=self.num_intervals,
            range=self.state_space,
        )[0].detach()
        next_mu = next_mu.to(self.device)
        next_mu = torch.nn.functional.normalize(next_mu, p=1)
        next_states = next_states.to(self.device)

        return next_mu.reshape(1, -1)