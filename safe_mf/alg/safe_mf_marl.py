import os

from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch
import wandb
from copy import deepcopy

from safe_mf.envs.vehicle_repositioning_sequential import VehicleRepositioningSequential
from safe_mf.envs.swarm_1d import Swarm1D
from safe_mf.models.gradient_descent import GradientDescent
from safe_mf.models.policy import MFPolicy, RandomPolicy, OptimalSwarm1DPolicy
from safe_mf.utils.entropy import entropic_constraint, max_entropy, entropy as entropy_func
from utils.utils import find_best_ckpt, find_last_ckpt


class SafeMFMARL:
    def __init__(
        self,
        env_name: str,
        mu_dim: int,
        control_std: float,
        constraint_lipschitz: float,
        max_entropy_ratio: float,
        barrier_lambda: float,
        dynamics_type: str,
        dynamics: Mapping[str, Any],
        solver: Mapping[str, Any],
        device: str,
        log_dir: str,
        input_data_path: str = None,
        demand_move: bool = None,
        delta: float = None,
        dynamics_ckpt: str = None,
        policy_ckpt: str = None,
        exec_type: str = "train",
        num_agents: int = 1,
        reward_type: str = 'current_mu'
    ) -> None:
        self.env_name = env_name
        if max_entropy_ratio is not None:
            self.constraint_function = lambda mu: entropic_constraint(
                mu, max_entropy_ratio * max_entropy(mu_dim)
            )
        else:
            self.constraint_function = None
        self.device = torch.device(device)
        self.dynamics_type = dynamics_type.lower()
        self.solver_cfg = solver
        self.hallucinated_control = self.dynamics_type == "unknown"
        self.num_agents = num_agents
        self.log_dir = log_dir
        self.policy_ckpt_dir = log_dir / "checkpoints" / "policy"
        self.dynamics_ckpt_dir = log_dir / "checkpoints" / "dynamics"
        self.results_dir = log_dir / "data" / exec_type
        if exec_type == 'eval':
            self.results_dir = self.results_dir / f'{self.num_agents}-ra'
        os.makedirs(self.policy_ckpt_dir, exist_ok=True)
        os.makedirs(self.dynamics_ckpt_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        if policy_ckpt is not None:
            if policy_ckpt == 'policy_best.pt':
                assert self.hallucinated_control is True, "Best policy exists only in the unknown environment"
                policy_ckpt = find_best_ckpt(self.policy_ckpt_dir)
            if policy_ckpt == 'policy.pt':
                policy_ckpt = find_last_ckpt(self.policy_ckpt_dir)
            self.policy_ckpt = self.policy_ckpt_dir / policy_ckpt
        else:
            self.policy_ckpt = None
        if dynamics_ckpt is not None:
            self.dynamics_ckpt = self.dynamics_ckpt_dir / dynamics_ckpt
        else:
            self.dynamics_ckpt = None
        self.exec_type = exec_type

        if self.env_name == "vehicle_repositioning_sequential":
            assert input_data_path is not None, "input_data_path must be specified for vehicle repositioning environment"
            input_data_path = Path(input_data_path)
            target_mu = np.load(input_data_path / 'target_mu.npy')
            target_mu = torch.from_numpy(target_mu).float().to(device)
            demand_matrix = np.load(input_data_path / 'demand_matrix.npy')
            demand_matrix = torch.from_numpy(demand_matrix).float().to(device)
            self.env = VehicleRepositioningSequential(
                mu_dim,
                target_mu,
                demand_matrix,
                control_std,
                constraint_lipschitz,
                barrier_lambda,
                demand_move,
                dynamics if self.hallucinated_control else None,
                self.constraint_function,
                device=device,
                dynamics_ckpt=self.dynamics_ckpt,
                exec_type=self.exec_type,
                num_agents=num_agents,
                reward_type=reward_type
            )
        elif self.env_name == "swarm-1d":
            self.env = Swarm1D(
                mu_dim,
                delta,
                control_std,
                constraint_lipschitz,
                barrier_lambda,
                dynamics if self.hallucinated_control else None,
                self.constraint_function,
                device=device,
                dynamics_ckpt=self.dynamics_ckpt,
                exec_type=self.exec_type,
                num_agents=num_agents,
                reward_type=reward_type
            )
        self.model = GradientDescent(
            self.env, self.device, self.hallucinated_control, **self.solver_cfg
        )

        self.count_evaluations = 0


    def _warmup(
        self,
        n_transitions: int,
    ):
        if n_transitions < 2:
            return
        policy = RandomPolicy(
            self.env.action_dim,
            self.env.action_space,
            self.device,
        )
        episode_current_states, episode_current_mus, episode_true_actions, episode_next_states = [], [], [], []
        self.env.reset()
        for step in range(n_transitions + 1):
            current_states, demand_states, next_states, current_mu, demand_mu, next_mu, true_actions, integrated_reward \
                = self.env.step(policy, step, known_dynamics=True, policy_training=False)
            episode_current_states.append(demand_states)
            episode_current_mus.append(demand_mu.repeat_interleave(self.num_agents, dim=0))
            episode_true_actions.append(true_actions)
            episode_next_states.append(next_states)

        episode_current_states = torch.cat(episode_current_states, dim=0)
        episode_current_mus = torch.cat(episode_current_mus, dim=0)
        episode_true_actions = torch.cat(episode_true_actions, dim=0)
        episode_next_states = torch.cat(episode_next_states, dim=0)

        self.env.dynamics.train(episode_current_states, episode_current_mus, episode_true_actions, episode_next_states)


    def _compute_entropy(self, mu: torch.Tensor) -> torch.Tensor:
        entropy = entropy_func(mu)

        return entropy.unsqueeze(0)
    

    def _compute_constraint_violation(self, mu: torch.Tensor) -> torch.Tensor:
        constraint_violation = self.constraint_function(mu)

        return constraint_violation.unsqueeze(0)
        

    def _evaluate(
        self,
        horizon: int,
        policy: MFPolicy,
        train_dynamics: bool,
        n_repeats: int = 10,
    ):
        visualization_mus, visualization_demand_mus = [], []
        visualization_integrated_rewards = []
        visualization_constraint_violations, visualization_entropies = [], []
        visualization_ra_states, visualization_ra_demand_states = [], []
        train_current_states, train_next_states = [], []
        train_current_mus, train_true_actions = [], []
        with torch.no_grad():
            policy.eval()
            for step in range(n_repeats):
                self.env.reset()
                episode_current_states, episode_demand_states, episode_next_states = [], [], []
                episode_current_mus, episode_demand_mus = [], [] 
                episode_true_actions, episode_integrated_rewards = [], []
                episode_constraint_violations, episode_entropies = [], []
                for _ in range(horizon):
                    current_states, demand_states, next_states, current_mu, \
                    demand_mu, next_mu, true_actions, integrated_reward = self.env.step(
                        policy, step, known_dynamics=True, policy_training=False
                    )
                    episode_current_states.append(current_states)
                    episode_demand_states.append(demand_states)
                    episode_next_states.append(next_states)
                    episode_current_mus.append(current_mu) 
                    episode_demand_mus.append(demand_mu)
                    episode_true_actions.append(true_actions)
                    episode_integrated_rewards.append(integrated_reward.unsqueeze(0))
                    episode_entropies.append(self._compute_entropy(current_mu))
                    if self.constraint_function is not None:
                        episode_constraint_violations.append(self._compute_constraint_violation(current_mu))

                if self.env_name == "vehicle_repositioning_sequential":
                    train_current_states.extend(episode_demand_states)
                    train_current_mus.extend(episode_demand_mus)            
                else:
                    train_current_states.extend(episode_current_states)
                    train_current_mus.extend(episode_current_mus)        
                train_true_actions.extend(episode_true_actions)
                train_next_states.extend(episode_next_states)

                # Postprocessing for visualization
                # Add the final mu to the list
                episode_current_mus.append(next_mu)
                visualization_mus.append(torch.stack(episode_current_mus).squeeze(1).cpu().numpy())

                if self.env_name == "vehicle_repositioning_sequential":
                    # Add dummy demand state to align the dimensions
                    episode_demand_mus.append(next_mu)
                    visualization_demand_mus.append(torch.stack(episode_demand_mus).squeeze(1).cpu().numpy())

                # Add dummy reward to align the dimensions
                integrated_reward = torch.zeros_like(integrated_reward)
                episode_integrated_rewards.append(integrated_reward.unsqueeze(0))
                visualization_integrated_rewards.append(torch.stack(episode_integrated_rewards).cpu().numpy())
 
                # Compute entropy after the last step
                episode_entropies.append(self._compute_entropy(next_mu))
                visualization_entropies.append(torch.stack(episode_entropies).cpu().numpy())
                if self.constraint_function is not None:
                    episode_constraint_violations.append(self._compute_constraint_violation(next_mu))
                    visualization_constraint_violations.append(torch.stack(episode_constraint_violations).cpu().numpy())

                if self.exec_type == 'eval':
                    episode_current_states.append(next_states)
                    visualization_ra_states.append(torch.stack(episode_current_states).cpu().numpy())
                    if self.env_name == "vehicle_repositioning_sequential":
                        episode_demand_states.append(next_states)
                        visualization_ra_demand_states.append(torch.stack(episode_demand_states).cpu().numpy())
      
        np.save(self.results_dir / f"integrated_rewards{self.count_evaluations}", visualization_integrated_rewards)
        np.save(self.results_dir / f"entropies{self.count_evaluations}", visualization_entropies)

        if self.constraint_function is not None:
            np.save(self.results_dir / f"constraint_violations{self.count_evaluations}", visualization_constraint_violations)

        visualization_mus = np.stack(visualization_mus)
        np.save(self.results_dir / f"mu_trajectories{self.count_evaluations}", visualization_mus)

        if self.env_name == "vehicle_repositioning_sequential":
            visualization_demand_mus = np.stack(visualization_demand_mus)
            np.save(self.results_dir / f"demand_mu_trajectories{self.count_evaluations}", visualization_demand_mus)

        if self.exec_type == "eval":
            visualization_ra_states = np.stack(visualization_ra_states)
            np.save(self.results_dir / f"ra_states{self.count_evaluations}", visualization_ra_states)
            if self.env_name == "vehicle_repositioning_sequential":
                visualization_ra_demand_states = np.stack(visualization_ra_demand_states)
                np.save(self.results_dir / f"ra_demand_states{self.count_evaluations}", visualization_ra_demand_states)

        self.count_evaluations += 1

        if train_dynamics:
            current_states = torch.cat(train_current_states, dim=0).to(self.device)
            current_mus = torch.cat(train_current_mus, dim=0).to(self.device)
            current_mus = current_mus.repeat_interleave(self.num_agents, dim=0)
            true_actions = torch.cat(train_true_actions, dim=0).to(self.device)
            next_states = torch.cat(train_next_states, dim=0).to(self.device)
            self.env.dynamics.train(current_states, current_mus, true_actions, next_states)

        total_reward = sum(sum(visualization_integrated_rewards))
        wandb.log({"avg_episode_reward": total_reward / n_repeats})
        wandb.log({"avg_step_reward": total_reward / (n_repeats * horizon)})
        total_entropy = sum(sum(visualization_entropies))
        wandb.log({"avg_step_entropy": total_entropy / (n_repeats * horizon)})
        if self.constraint_function is not None:
            min_constraint_violation = np.inf
            num_constraint_violations = 0
            for min_constraint_violations in visualization_constraint_violations:
                min_constraint_violation = min(min_constraint_violation, min(min_constraint_violations))
                num_constraint_violations += sum(min_constraint_violations < 0)
            wandb.log({"min_constraint_violation": min_constraint_violation})  
            wandb.log({"avg_episode_constraint_violations": num_constraint_violations / n_repeats})  

        return total_reward / n_repeats
    

    def _run_optimal_policy(self, horizon: int, n_repeats: int = 10):
        opt_policy = OptimalSwarm1DPolicy()

        _ = self._evaluate(horizon, opt_policy, n_repets=n_repeats, train_dynamics=False)

    def _run_known_dynamics(self, horizon: int, policy_epochs: int, n_repeats: int = 10):
        opt_policy = self.model.train(policy_epochs, horizon)

        _ = self._evaluate(horizon, opt_policy, n_repeats=n_repeats, train_dynamics=False)

        torch.save(opt_policy, self.policy_ckpt_dir / "policy_final.pt")

    def _run_unknown_dynamics(
        self,
        horizon: int,
        policy_epochs: int,
        warmup_steps: int,
        n_episodes: int,
        n_repeats: int = 10,
    ):
        self._warmup(warmup_steps)
        opt_policy = self.model.policy
        best_policy = opt_policy
        best_reward = -torch.inf
        for i in range(n_episodes):
            avg_reward = self._evaluate(
                horizon,
                opt_policy,
                train_dynamics=True,
                n_repeats=n_repeats,
            )
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_policy = deepcopy(opt_policy)
                torch.save(best_policy, f"{self.policy_ckpt_dir}/policy_best{i}.pt")
            torch.save(opt_policy, f"{self.policy_ckpt_dir}/policy{i}.pt")
            torch.save(self.env.dynamics, f"{self.dynamics_ckpt_dir}/dynamics{i}.pt")
            opt_policy = self.model.train(policy_epochs, horizon)

        torch.save(best_policy, f"{self.policy_ckpt_dir}/policy_final.pt")
        torch.save(self.env.dynamics, f"{self.dynamics_ckpt_dir}/dynamics_final.pt")


    def run(
        self,
        n_episodes: int,
        horizon: int,
        policy_epochs: int,
        warmup_steps: int,
        n_repeats: int = 4,
    ) -> None:
        if self.exec_type == "train":
            if self.dynamics_type == "known":
                self._run_known_dynamics(horizon, policy_epochs, n_repeats)
            elif self.dynamics_type == "optimal_policy":
                self._run_optimal_policy(horizon, n_repeats)
            else:
                self._run_unknown_dynamics(
                    horizon,
                    policy_epochs,
                    warmup_steps,
                    n_episodes,
                    n_repeats
                )
        elif self.exec_type == "eval":
            opt_policy = torch.load(self.policy_ckpt, map_location=self.device)
            _ = self._evaluate(horizon, opt_policy, n_repeats=n_repeats, train_dynamics=False)
