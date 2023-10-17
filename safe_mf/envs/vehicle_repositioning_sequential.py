import math
from typing import Any, Callable, Mapping, Optional, Tuple

import torch
import torchist


from safe_mf.envs.env import Env
from safe_mf.models.ensemble import UnknownDynamics
from safe_mf.models.policy import MFPolicy
from safe_mf.utils.utils import states_to_cell, cells_to_index, index_to_cell
from safe_mf.utils.entropy import kld
from torch.distributions import Normal
from safe_mf.utils.distributions import TruncatedNormal, shifted_uniform


class VehicleRepositioningSequential(Env):
    def __init__(
        self,
        mu_dim: int,
        target_mu: torch.Tensor,
        demand_matrix: torch.Tensor,
        control_std: float,
        constraint_lipschitz: float,
        barrier_lambda: float = 0.001,
        demand_move: bool = False,
        dynamics_cfg: Mapping[str, Any] = None,
        constraint_function: Callable = None,
        state_space: Tuple[float, float] = (0.0, 1.0),
        action_space: Tuple[float, float] = (-1.0, 1.0),
        device: torch.device = torch.device("cpu"),        
        dynamics_ckpt: str = None,
        exec_type: str = "train",
        num_agents: int = 1,
        reward_type: str = 'current_mu'
    ) -> None:
        super().__init__(state_dim=2, mu_dim=mu_dim, action_dim=2, device=device)
        if target_mu.ndim == 1:
            self.target_mu = target_mu.unsqueeze(0)
        else:
            self.target_mu = target_mu.unsqueeze(1)
        self.demand_matrix = demand_matrix
        # When all rows in the demand matrix are 0 we want agents to stay in the current cell
        # The logic is necessary for representative agent move in get_demand_states()
        self.demand_move_matrix = demand_matrix.clone()
        mask = torch.all(self.demand_matrix == 0, dim=-1).nonzero()
        if self.demand_move_matrix.ndim == 2:
            self.demand_move_matrix[mask[:, 0], mask[:, 0]] = 1.
        else:
            self.demand_move_matrix[mask[:, 0], mask[:, 1], mask[:, 1]] = 1.
        self.state_space = state_space
        self.action_space = action_space
        self.constraint_lipschitz = constraint_lipschitz
        self.barrier_lambda = barrier_lambda
        self.demand_move = demand_move
        self.num_intervals = int(math.sqrt(self.mu_dim))
        self.control_std = control_std if control_std else None
        self.exec_type = exec_type
        self.num_agents = num_agents
        self.reward_type = reward_type
        self.constraint_function = constraint_function
        self.linspace_x = torch.linspace(self.state_space[0], self.state_space[1], self.num_intervals + 1, device=device)
        self.linspace_y = torch.linspace(self.state_space[0], self.state_space[1], self.num_intervals + 1, device=device)
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

        padding = torch.tensor([10 ** 4], device=self.device).unsqueeze(0)
        upper = self.cell_centers_1d + (0.5 / self.num_intervals)
        lower = self.cell_centers_1d - (0.5 / self.num_intervals)
        first = lower[0][0].reshape(padding.shape)
        last = upper[0][-1].reshape(padding.shape)
        self.lower = torch.cat([-padding, lower, last], dim=1)
        self.upper = torch.cat([first, upper, padding], dim=1)

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
            self.acc_uncertainty = torch.tensor(0.0, device=self.device)
        else:
            self.acc_uncertainty = None


    def move_demand_to_destinations(self, destination_idx):
        # Choose demand state uniformly from a cell
        idx_x, idx_y = index_to_cell(destination_idx, self.num_intervals)
        x = shifted_uniform(self.linspace_x[idx_x], self.linspace_x[idx_x + 1], device=self.device)
        y = shifted_uniform(self.linspace_y[idx_y], self.linspace_y[idx_y + 1], device=self.device)
        demand_states = torch.stack([x, y], dim=1)

        return demand_states


    def get_demand_states(self, states, ra_move=False, step: Optional[int] = None):
        if self.demand_move and ra_move:
            if self.demand_move_matrix.ndim == 2:
                demand_move_matrix = self.demand_move_matrix
            else:
                demand_move_matrix = self.demand_move_matrix[step]
            cells = states_to_cell(states, self.linspace_x, self.linspace_y)
            origins_idx = cells_to_index(cells, self.num_intervals)
            transition_probs = demand_move_matrix[origins_idx, :]
            destinations_idx = torch.multinomial(transition_probs, 1).squeeze(1)
            demand_states = self.move_demand_to_destinations(destinations_idx)
        else:
            cells = states_to_cell(states, self.linspace_x, self.linspace_y)
            origins_idx = cells_to_index(cells, self.num_intervals)
            destinations_idx = origins_idx
            demand_states = self.move_demand_to_destinations(destinations_idx)

        return demand_states


    def get_next_states(self, current_states, actions, mf_transition: bool=False):
        # Unlike swarm environment, here we don't have delta * actions meaning that
        # the next state can be obtained using the identical logic in true and
        # approximated dynamics
        next_states = current_states + actions 
        next_states = torch.clamp(next_states, self.state_space[0], self.state_space[1])
        if mf_transition:
            return next_states
        else:
            with torch.no_grad():
                zeros = torch.zeros_like(next_states)
                std = torch.ones_like(next_states) * self.control_std
                lower_bound = self.state_space[0] - next_states
                upper_bound = self.state_space[1] - next_states
                tnorm = TruncatedNormal(loc=zeros, scale=std, a=lower_bound, b=upper_bound)
                noise = tnorm.sample()
                del zeros, std, lower_bound, upper_bound
                noisy_next_states = next_states + noise

            return noisy_next_states
    

    def get_demand_mu(self, mu, step: Optional[int] = None):
        if self.demand_move:
            if self.demand_matrix.ndim == 2:
                demand_matrix = self.demand_matrix
            else:
                demand_matrix = self.demand_matrix[step]
            if self.target_mu.ndim == 2:
                target_mu = self.target_mu
            else:
                target_mu = self.target_mu[step]
            one = torch.tensor(1, device=self.device)
            mu_ = torch.log(mu + 1e-30)
            target_mu_ = torch.log(target_mu + 1e-30)
            p = torch.min(one, torch.exp(target_mu_ - mu_))
            demand_move = (mu * p) @ demand_matrix
            pending_supply = mu * (1 - p)
            mu = demand_move + pending_supply
            mu = torch.nn.functional.normalize(mu, p=1)

        return mu

    def _true_dynamics(
        self,
        states: torch.Tensor,  # [?, state_dim]
        mu: torch.Tensor,  # [1, mu_dim]
        policy: MFPolicy,
        exploration: Optional[float],
        grid_actions: Optional[torch.Tensor] = None,
        mf_transition: bool = False
    ) -> torch.Tensor:
        act_dim = self.action_dim
        actions = policy(states, mu, exploration)
        true_actions = actions[:, :act_dim]
        next_states = self.get_next_states(states, true_actions, mf_transition)

        return true_actions, next_states


    def _approximate_dynamics(
        self,
        states: torch.Tensor,  # [?, state_dim]
        mu: torch.Tensor,  # [1, mu_dim]
        policy: MFPolicy,
        exploration: Optional[float],
        grid_actions: Optional[torch.Tensor] = None,
        mf_transition: bool = False
    ) -> torch.Tensor:
        actions = policy(states, mu, exploration)
        act_dim = self.action_dim
        true_actions = actions[:, :act_dim]
        hallucinated_actions = actions[:, act_dim:]
        means, epi, alea = self.dynamics(states, mu, true_actions)
        epi = torch.diagonal(epi, dim1=1, dim2=2)
        hallucinated_actions = self.dynamics.beta * torch.sqrt(
            epi
        ) * hallucinated_actions.reshape(-1, self.state_dim)
        next_states = self.get_next_states(means, hallucinated_actions, mf_transition)
        
        return actions, next_states
    
    def reward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """A dummy reward needed to be aligned with the Gym interface"""
        pass


    def integrated_reward(
        self,
        current_mu: torch.Tensor,
        actions: torch.Tensor,
        next_mu: torch.Tensor,
        step: Optional[int] = None,
        constraint_function: Callable = None,
    ) -> float:
        if self.target_mu.ndim == 2:
            target_mu = self.target_mu
        else:
            target_mu = self.target_mu[step]
        if self.reward_type == 'next_mu':
            main_reward = -kld(target_mu, next_mu)
        else:
            main_reward = -kld(target_mu, current_mu)
        if constraint_function is None:
            return main_reward
        else:
            constraint_value = constraint_function(current_mu)
            if self.dynamics is None:
                barrier = self.barrier_lambda * torch.nan_to_num(
                    torch.log(constraint_value),
                    nan=-10,
                    neginf=-10,
                )
            else:
                acc_uncertainty = self.acc_uncertainty
                barrier = self.barrier_lambda * torch.nan_to_num(
                     torch.log(
                        constraint_value
                        - self.constraint_lipschitz * acc_uncertainty
                    ),
                    nan=-10,
                    neginf=-10,
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
        ra_current_states = self.ra_states
        current_mu = self.mu

        demand_mu = self.get_demand_mu(current_mu, step) #.detach().clone())
        grid_demand_states = self.get_demand_states(self.cell_centers, ra_move=False, step=step)
        grid_actions, grid_next_states = dynamics(
                grid_demand_states,
                demand_mu,
                policy,
                exploration,
                mf_transition=True
            )
        true_grid_actions = grid_actions[:, :self.action_dim]
        # Collect data for eval, i.e., when not training policy
        if not policy_training:
            with torch.no_grad():
                ra_demand_states = self.get_demand_states(ra_current_states, ra_move=True, step=step)
                ra_actions, ra_next_states = dynamics(
                    ra_demand_states,
                    demand_mu,
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
                grid_demand_states,
                demand_mu,
                true_grid_actions
            )
        if self.exec_type == "eval":
            next_mu = self._step_deterministic(ra_next_states, demand_mu)
        else:
            next_mu = self._step_probabilistic(grid_next_states, demand_mu)
        integrated_reward = self.integrated_reward(
            current_mu, true_grid_actions, next_mu, step, self.constraint_function
        )
        self.mu = next_mu

        if not policy_training:
            return (
                    ra_current_states, ra_demand_states, 
                    ra_next_states, current_mu, demand_mu, 
                    next_mu, ra_true_actions, integrated_reward
                )
        else:
            return integrated_reward
            

    def _step_probabilistic(
        self,
        next_states: torch.Tensor,
        current_mu: torch.Tensor,
    ) -> torch.Tensor:
        """
        ERF based
        """
        joint_probs = torch.zeros(size=[self.mu_dim for _ in range(2)], device=self.device)
        probs = []
        for i in range(self.state_dim):
            next_states_projection = next_states[:, i:i+1]
            probs.append(self._apply_truncated_cdf(next_states_projection))
        cols = 0
        for i in range(self.num_intervals):
            joint_probs[:, cols:cols + self.num_intervals] = probs[0][:, i:i + 1] * probs[1]
            cols += self.num_intervals
        multiplied_probs = (joint_probs.T * current_mu).sum(dim=-1)
        next_mu = multiplied_probs.reshape(-1, self.mu_dim).sum(dim=0)

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
        next_mu = torchist.histogramdd(
            next_states,
            bins=self.num_intervals,
            low=self.state_space[0],
            upp=self.state_space[1],
        )
        # We need to transpose next_mu to match our shape
        next_mu = next_mu.T
        next_mu = next_mu.to(self.device).float().reshape(1, -1)
        next_mu = torch.nn.functional.normalize(next_mu, p=1)
        next_states = next_states.to(self.device)

        return next_mu


    def _apply_truncated_cdf(
        self, next_states_projection: torch.Tensor
    ) -> torch.Tensor:
        cdf = (
                self.normal_control.cdf(self.upper - next_states_projection) 
                - self.normal_control.cdf(self.lower - next_states_projection)
            )
        w = cdf[:, 1:-1] / cdf[:, 1:-1].sum(dim=1, keepdim=True)
        cdf = cdf[:, 1:-1] + (cdf[:, 0].unsqueeze(1) + cdf[:, -1].unsqueeze(1)) * w
        cdf /= cdf.sum(dim=1, keepdim=True)
        
        return cdf


    def _apply_erf(
            self, lower: torch.Tensor, mean: torch.Tensor, upper: torch.Tensor, noise
        ) -> torch.Tensor:
            upper[0][-1] += 10 ** 4 
            lower[0][0] -= 10 ** 4
            return 0.5 * (
                torch.erf((upper - mean) / (noise * math.sqrt(2)))
                - torch.erf((lower - mean) / (noise * math.sqrt(2)))
            )


    def _apply_cdf(
        self, lower: torch.Tensor, mean: torch.Tensor, upper: torch.Tensor, noise
    ) -> torch.Tensor:
        upper[0][-1] += 10 ** 4 
        lower[0][0] -= 10 ** 4

        return self.normal.cdf(upper - mean) - self.normal.cdf(lower - mean)