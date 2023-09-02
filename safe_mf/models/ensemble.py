from copy import deepcopy
import math
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from safe_mf.utils.data import (
    concat_inputs,
    normalize_inputs,
    DynamicsDataset,
)
from safe_mf.utils.stats_tracker import StatsTracker


class EnsembleMember(nn.Module):
    def __init__(
        self,
        state_dim: int,
        mu_dim: int,
        action_dim: int,
        hidden_dims: List[int],
    ) -> None:
        super().__init__()
        self.mu_dim = mu_dim
        total_dim = state_dim + mu_dim + action_dim

        dims = [total_dim] + hidden_dims
        self.model_core = []
        for i in range(len(dims) - 1):
            self.model_core += [nn.Linear(dims[i], dims[i + 1]), nn.LeakyReLU()]
        self.model_core = nn.Sequential(*self.model_core)

        self.mean = nn.Linear(hidden_dims[-1], state_dim)
        self.logvar = nn.Linear(hidden_dims[-1], state_dim)

        self.max_logvar = nn.parameter.Parameter(torch.ones((1, state_dim)))
        self.min_logvar = nn.parameter.Parameter(-1 * torch.ones((1, state_dim)))

        self.init_layers()


    def init_layers(self) -> None:
        for layer in self.model_core:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=math.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.mean.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.mean.bias)
        nn.init.xavier_uniform_(self.logvar.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.logvar.bias)


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Computes the action given the state and the mean field

        Args:
            states (torch.Tensor): [n, state_dim]
            mu (torch.Tensor): [mu_dim]

        Returns:
            torch.Tensor: means [n, action_dim]
            torch.Tensor: vars [n, action_dim]
        """

        outs = self.model_core(inputs)
        means = self.mean(outs)
        logvars = self.logvar(outs)
        logvars_ = self.max_logvar - nn.Softplus()(self.max_logvar - logvars)
        logvars__ = self.min_logvar + nn.Softplus()(logvars_ - self.min_logvar)

        return means, torch.exp(logvars__)


class UnknownDynamics(nn.Module):
    def __init__(
        self,
        beta: float,
        ensemble_nets: int,
        lr: float,
        state_dim: int,
        mu_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        dynamics_epochs: int,
        batch_size: int,
        holdout: float,
        reset_params_every_episode: bool = True,
        buffer_size: float = 10_000,
        state_space: Tuple[float, float] = None,
        action_space: Tuple[float, float] = None,
        device: torch.device = torch.device("cpu"),
        weight_decay: float = 0.0005
    ) -> None:
        super().__init__()
        self.beta = beta
        self.device = device
        self.ensemble_nets = ensemble_nets
        self.state_dim = state_dim
        self.mu_dim = mu_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.state_space = state_space
        self.action_space = action_space
        self.lr = lr
        self.dynamics_epochs = dynamics_epochs
        self.batch_size = batch_size
        self.holdout = holdout
        self.reset_params_every_episode = reset_params_every_episode
        self.weight_decay = weight_decay

        self.models = nn.ModuleList(
            [
                EnsembleMember(state_dim, mu_dim, action_dim, hidden_dims)
                for _ in range(self.ensemble_nets)
            ]
        ).to(device)

        (
            self.saved_states,
            self.saved_mus,
            self.saved_actions,
            self.saved_next_states,
        ) = ([], [], [], [])

        self.optimizers = [
            torch.optim.Adam(m.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            for m in self.models
        ]


    def reset_parameters(self) -> None:
        for m in self.models:
            m.init_layers()


    def forward(
        self,
        states: torch.Tensor,  # [n, state_dim]
        mu: torch.Tensor,  # [1, mu_dim]
        actions: torch.Tensor,  # [n, action_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stats_tracker = StatsTracker()

        norm_states, norm_mu, norm_actions = normalize_inputs(
            states, mu, actions, self.state_space, self.action_space
        )
        inputs = concat_inputs(norm_states, norm_mu, norm_actions)

        means, vars = [], []
        for m in self.models:
            mean, var = m(inputs)
            means.append(mean)
            vars.append(var)

        stacked_means = torch.stack(means)
        out_means = stacked_means.mean(dim=0)
        out_epistemic = torch.matmul(
            (stacked_means - out_means).unsqueeze(-1),
            (stacked_means - out_means).unsqueeze(-1).permute((0, 1, 3, 2)),
        ).sum(dim=0) / (self.ensemble_nets - 1)
        out_aleatoric = (torch.stack(vars)).mean(dim=0)

        return (
            states + stats_tracker.denormalize_diffs(out_means),
            out_epistemic * stats_tracker.diff_std**2,
            out_aleatoric * stats_tracker.diff_std**2,
        )


    def expected_deviation(
        self, fixed_states: torch.Tensor, mu: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        _, covariances, _ = self.forward(
            fixed_states.reshape(self.mu_dim, -1), mu, actions.reshape(self.mu_dim, -1)
        )
        return (
            torch.sqrt(
                torch.sum(torch.diagonal(covariances, dim1=1, dim2=2), dim=1)
            ).flatten()
            * mu
        ).sum()


    def train(
        self,
        states: torch.Tensor,
        mus: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
    ):
        if self.reset_params_every_episode:
            self.reset_parameters()

        stats_tracker = StatsTracker()
        stats_tracker.update_diffs(next_states - states)

        self.saved_states.append(states)
        self.saved_mus.append(mus)
        self.saved_actions.append(actions)
        self.saved_next_states.append(next_states)
        if len(self.saved_states) > self.buffer_size:
            self.saved_states.pop(0)
            self.saved_mus.pop(0)
            self.saved_actions.pop(0)
            self.saved_next_states.pop(0)
        if self.batch_size == 8 and len(self.saved_states) >= 50:
            self.batch_size = 16

        states = torch.cat(self.saved_states, dim=0)
        mus = torch.cat(self.saved_mus, dim=0)
        actions = torch.cat(self.saved_actions, dim=0)
        next_states = torch.cat(self.saved_next_states, dim=0)

        norm_diffs = stats_tracker.normalize_diffs(next_states - states)

        norm_states, norm_mus, norm_actions = normalize_inputs(
            states,
            mus,
            actions,
            self.state_space,
            self.action_space,
        )
        inputs = torch.cat([norm_states, norm_mus, norm_actions], dim=1)
        nll_loss = nn.GaussianNLLLoss(reduction="sum")
        mse_loss = nn.MSELoss(reduction="sum")

        split_point = int(self.holdout * len(inputs))
        for m, opt in zip(self.models, self.optimizers):
            idx = torch.randperm(len(inputs))
            bidx = torch.poisson(torch.ones(split_point)).to(int).to(self.device)

            train_loader = DataLoader(
                DynamicsDataset(
                    torch.repeat_interleave(inputs[idx][:split_point], bidx, dim=0),
                    torch.repeat_interleave(norm_diffs[idx][:split_point], bidx, dim=0),
                ),
                shuffle=True,
                batch_size=self.batch_size,
            )
            val_loader = DataLoader(
                DynamicsDataset(
                    inputs[idx][split_point:], norm_diffs[idx][split_point:]
                ),
                batch_size=self.batch_size,
            )
            best_val_loss = torch.inf
            best_train_loss = torch.inf
            best_model = deepcopy(m.state_dict())
            early_stopper = EarlyStopper(patience=100, min_improvement=0.005)
            for t in tqdm(range(self.dynamics_epochs)):
                train_loss = 0.0
                m.train()
                for batch_inputs, batch_norm_diffs in train_loader:
                    out_means, out_vars = m(batch_inputs)
                    loss = nll_loss(
                        out_means.reshape(-1),
                        batch_norm_diffs.reshape(-1),
                        out_vars.reshape(-1),
                    )
                    opt.zero_grad()
                    (
                        loss + 0.001 * (m.max_logvar.sum() - m.min_logvar.sum())
                    ).backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0, norm_type=2)
                    opt.step()
                    train_loss += loss.item()
                val_loss = 0.0
                m.eval()
                for batch_inputs, batch_norm_diffs in val_loader:
                    with torch.no_grad():
                        out_means, out_vars = m(batch_inputs)
                        loss = mse_loss(
                            out_means.reshape(-1),
                            batch_norm_diffs.reshape(-1),
                        )
                        val_loss += loss.item()
                if early_stopper.early_stop(val_loss):             
                    break
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = deepcopy(m.state_dict())
            m.load_state_dict(best_model)


class EarlyStopper:
    def __init__(self, patience=1, min_improvement=0):
        self.patience = patience
        self.min_improvement = min_improvement
        self.counter = 0
        self.min_val_loss = torch.inf

    def early_stop(self, val_loss):
        if val_loss < self.min_val_loss * (1 - self.min_improvement):
            self.min_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False