from copy import deepcopy
from typing import Any, Mapping

import torch
from tqdm import tqdm
import wandb
from safe_mf.envs.env import Env

from safe_mf.models.policy import MFPolicy

class GradientDescent:
    def __init__(
        self,
        env: Env,
        device: torch.device,
        hallucinated_control: bool,
        policy: Mapping[str, Any],
        action_std: float,
        polar: bool,
        gmm: bool,
        reset_params_every_episode: bool = True,
        patience: int = 1,
        min_improvement: int = 0.005
    ) -> None:
        super().__init__()
        self.env = env
        self.state_dim = self.env.state_dim
        self.mu_dim = self.env.mu_dim
        self.action_dim = self.env.action_dim
        self.device = device
        self.hallucinated_control = hallucinated_control
        self.state_space = self.env.state_space
        self.action_space = self.env.action_space
        self.action_std = action_std
        self.polar = polar
        self.gmm = gmm
        self.reset_params_every_episode = reset_params_every_episode
        self.patience = patience
        self.min_improvement = min_improvement

        self.policy = MFPolicy(
            self.state_dim,
            self.mu_dim,
            self.action_dim,
            self.hallucinated_control,
            policy["hidden_dims"],
            self.state_space,
            self.action_space,
            self.gmm,
            self.polar,
        ).to(self.device)

        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=policy["lr"],
            weight_decay=policy["weight_decay"],
        )

    def train(
        self,
        policy_epochs: int,
        horizon: int,
    ) -> MFPolicy:
        if self.reset_params_every_episode:
            self.policy.reset_parameters()
        best_model = deepcopy(self.policy.state_dict())
        best_loss = -torch.inf
        early_stopper = EarlyStopper(patience=self.patience, min_improvement=self.min_improvement)
        for epoch in tqdm(range(policy_epochs)):
            self.policy.eval()
            self.policy_optimizer.zero_grad()
            self.env.reset()
            rewards = []
            for step in range(horizon): 
                reward = self.env.step(
                    self.policy,
                    step,
                    known_dynamics=not self.hallucinated_control,
                    action_std = self.action_std * (1.0 - epoch / policy_epochs),
                    policy_training=True
                )
                rewards.append(reward)
            self.policy.train()
            loss = -torch.stack(rewards).sum()
            current_loss = -loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0, norm_type=2)
            self.policy_optimizer.step()
            # Early stopping
            if (epoch - 1) % 100 == 0:
                if early_stopper.early_stop(current_loss, best_loss):
                    break
            if current_loss >= best_loss:
                best_model = deepcopy(self.policy.state_dict())
                best_loss = current_loss
            if epoch % 1 == 0:
                wandb.log({"training_episode_reward": current_loss})

               
        self.policy.load_state_dict(best_model)

        return self.policy

class EarlyStopper:
    def __init__(self, patience=1, min_improvement=0):
        self.patience = patience
        self.min_improvement = min_improvement
        self.counter = 0
        self.early_stop_loss = -torch.inf

    def early_stop(self, loss, best_loss):
        if self.early_stop_loss > 0:
            requirement = self.early_stop_loss * (1 + self.min_improvement)
        else:
            requirement = self.early_stop_loss * (1 - self.min_improvement)
        if loss > requirement:
            self.early_stop_loss = best_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False