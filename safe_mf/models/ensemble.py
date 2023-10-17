from copy import deepcopy
import math
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from safe_mf.utils.data import (
    concat_inputs,
    concat_inputs_without_mu,
    normalize_inputs,
    DynamicsDataset,
)


class EnsembleMember(nn.Module):
    def __init__(
        self,
        state_dim: int,
        mu_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        exclude_mu: Optional[bool] = False,
        min_var: Optional[float] = 1e-3,
        max_var: Optional[float] = 10,
    ) -> None:
        super().__init__()
        self.mu_dim = mu_dim
        if exclude_mu:
            total_dim = state_dim + action_dim
        else:
            total_dim = state_dim + mu_dim + action_dim

        dims = [total_dim] + hidden_dims
        self.model_core = []
        for i in range(len(dims) - 1):
            self.model_core += [nn.Linear(dims[i], dims[i + 1]), nn.BatchNorm1d(dims[i + 1]), nn.LeakyReLU()]
        self.model_core = nn.Sequential(*self.model_core)

        self.mean = nn.Linear(hidden_dims[-1], state_dim)
        self.logvar = nn.Linear(hidden_dims[-1], state_dim)

        # Code below sets the variance in a range [min_var, max_var - min_var]
        self.max_logvar = nn.parameter.Parameter(np.log(max_var - min_var) * torch.ones((1, state_dim)))
        self.min_logvar = nn.parameter.Parameter(np.log(min_var) * torch.ones((1, state_dim)))

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
        exclude_mu: Optional[bool] = False,
        adversarial_epsilon: Optional[float] = None,
        reset_params_every_episode: bool = True,
        buffer_size: float = 10_000,
        state_space: Tuple[float, float] = None,
        action_space: Tuple[float, float] = None,
        extended_state_space: Tuple[float, float] = None,
        device: torch.device = torch.device("cpu"),
        weight_decay: float = 0.0005,
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
        self.extended_state_space = extended_state_space
        self.lr = lr
        self.dynamics_epochs = dynamics_epochs
        self.batch_size = batch_size
        self.holdout = holdout
        self.exclude_mu = exclude_mu
        self.adversarial_epsilon = adversarial_epsilon
        self.reset_params_every_episode = reset_params_every_episode
        self.weight_decay = weight_decay
        self._diff_mean = None
        self._diff_sq = None
        self._diff_count = 0

        if extended_state_space is None:
            self.min_var = 0.001**2
            self.max_var = 1
        else:
            self.min_var = 0.001**2
            self.max_var = 1

        self.models = nn.ModuleList(
            [
                EnsembleMember(
                    state_dim, mu_dim, action_dim, hidden_dims, 
                    self.exclude_mu, self.min_var, self.max_var
                )
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

    @property
    def diff_std(self):
        return torch.sqrt(self._diff_sq / (self._diff_count - 1))

    @property
    def diff_mean(self):
        return self._diff_mean

    def update_diffs(self, diffs: torch.Tensor):
        """Assumes initial batch to be larger than 1"""
        if self.diff_mean is None:
            self._diff_mean = diffs.detach().mean(dim=0)
            self._diff_sq = ((diffs - self.diff_mean) ** 2).detach().sum(dim=0)
            self._diff_count = diffs.shape[0]
        else:
            m = diffs.detach().mean(dim=0)
            sq = ((diffs - m) ** 2).detach().sum(dim=0)
            c = diffs.shape[0]

            new_count = self._diff_count + c
            delt = m - self._diff_mean
            new_mean = self._diff_mean + delt * (c / new_count)
            new_diff_sq = (
                self._diff_sq + sq + delt**2 * (self._diff_count * c / new_count)
            )
            self._diff_mean = new_mean
            self._diff_sq = new_diff_sq
            self._diff_count = new_count

    def normalize_diffs(self, diffs: torch.Tensor):
        return (diffs - self.diff_mean) / self.diff_std

    def denormalize_diffs(self, diffs: torch.Tensor):
        return diffs * self.diff_std + self.diff_mean


    def forward(
        self,
        states: torch.Tensor,  # [n, state_dim]
        mu: torch.Tensor,  # [1, mu_dim]
        actions: torch.Tensor,  # [n, action_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = self.normalize_inputs(states, mu, actions, self.exclude_mu)

        means, vars = [], []
        for m in self.models:
            mean, var = m(inputs)
            means.append(mean)
            vars.append(var)

        stacked_means = torch.stack(means)
        stacked_means = self.denormalize_diffs(stacked_means)
        out_means = stacked_means.mean(dim=0)
        out_epistemic = torch.matmul(
            (stacked_means - out_means).unsqueeze(-1),
            (stacked_means - out_means).unsqueeze(-1).permute((0, 1, 3, 2)),
        ).sum(dim=0) / (self.ensemble_nets - 1)
        vars = torch.stack(vars) * self.diff_std ** 2
        out_aleatoric = vars.mean(dim=0)

        if self.extended_state_space is None:
            states = torch.clamp(states + out_means, self.state_space[0], self.state_space[1])
        else:
            states = torch.clamp(states + out_means, self.extended_state_space[0], self.extended_state_space[1])
        out_epistemic = torch.clamp(out_epistemic, self.min_var, self.max_var)
        out_aleatoric = torch.clamp(out_aleatoric, self.min_var, self.max_var)

        return (
            states,
            out_epistemic,
            out_aleatoric,
        )


    def normalize_inputs(
            self, states: torch.Tensor, mu: torch.Tensor, 
            actions: torch.Tensor, exclude_mu: Optional[bool] = False
        ) -> torch.Tensor:
        if self.extended_state_space is None:
            norm_states, norm_mu, norm_actions = normalize_inputs(
                states, mu, actions, self.state_space, self.action_space, exclude_mu
            )
        else:
            norm_states, norm_mu, norm_actions = normalize_inputs(
                states, mu, actions, self.extended_state_space, self.action_space, exclude_mu
            )
        if exclude_mu:
            inputs = concat_inputs_without_mu(norm_states, norm_actions)
        else:
            inputs = concat_inputs(norm_states, norm_mu, norm_actions)

        return inputs


    def expected_deviation(
        self, fixed_states: torch.Tensor, mu: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        _, vars, _ = self.forward(
            fixed_states.reshape(self.mu_dim, -1), mu, actions.reshape(self.mu_dim, -1)
        )
        return (
            torch.sqrt(
                torch.sum(torch.diagonal(vars, dim1=1, dim2=2), dim=1)
            ).flatten()
            * mu
        ).sum()


    def create_adversarial_samples(
        self,
        m: EnsembleMember,
        z: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        nll_loss = nn.GaussianNLLLoss(reduction="sum")
        z.requires_grad = True
        means, vars = m(z)
        loss = nll_loss(means.reshape(-1), y.reshape(-1), vars.reshape(-1))
        m.zero_grad()
        loss.backward()
        grad = z.grad.data
        grad_sign = grad.sign()
        z_perturbed = (z + self.adversarial_epsilon * grad_sign).detach()
        if self.exclude_mu:
            offset = 0
        else:
            offset = self.mu_dim
        states_perturbed = z_perturbed[:, :self.state_dim]
        if not self.exclude_mu:
            mus_perturbed = z_perturbed[:, self.state_dim:self.state_dim + offset]
            mus_perturbed = torch.nn.functional.normalize(torch.clamp(mus_perturbed, 0.0, 1.0), p=1, dim=1)
        actions_perturbed = z_perturbed[:, self.state_dim + offset:]
        if self.extended_state_space is None:
            states_perturbed = torch.clamp(states_perturbed, self.state_space[0], self.state_space[1])
        else:
            states_perturbed = torch.clamp(states_perturbed, self.extended_state_space[0], self.extended_state_space[1])
        actions_perturbed = torch.clamp(actions_perturbed, -1.0, 1.0)
        if not self.exclude_mu:
            z_perturbed = torch.cat([states_perturbed, mus_perturbed, actions_perturbed], dim=1)
        else:
            z_perturbed = torch.cat([states_perturbed, actions_perturbed], dim=1)

        return z_perturbed


    def train(
        self,
        states: torch.Tensor,
        mus: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
    ):
        if self.reset_params_every_episode:
            self.reset_parameters()
        self.update_diffs(next_states - states)
        self.saved_states.append(states)
        if not self.exclude_mu:
            self.saved_mus.append(mus)
        self.saved_actions.append(actions)
        self.saved_next_states.append(next_states)
        if len(self.saved_states) > self.buffer_size:
            self.saved_states.pop(0)
            if not self.exclude_mu:
                self.saved_mus.pop(0)
            self.saved_actions.pop(0)
            self.saved_next_states.pop(0)
        num_samples = len(self.saved_states) * len(self.saved_states[0]) 
        if self.batch_size <= 8 and num_samples >= 1_500:
            self.batch_size = 16
        elif self.batch_size <= 16 and num_samples >= 3_000:
            self.batch_size = 32
        elif self.batch_size <= 32 and num_samples >= 6_000:
            self.batch_size = 64
        elif self.batch_size <= 64 and num_samples >= 12_000:
            self.batch_size = 128
        elif self.batch_size <= 128 and num_samples >= 24_000:
            self.batch_size = 256
        elif self.batch_size <= 256 and num_samples >= 48_000:
            self.batch_size = 512

        states = torch.cat(self.saved_states, dim=0)
        if not self.exclude_mu:
            mus = torch.cat(self.saved_mus, dim=0)
        actions = torch.cat(self.saved_actions, dim=0)
        next_states = torch.cat(self.saved_next_states, dim=0)
        norm_diffs = self.normalize_diffs(next_states - states)
        inputs = self.normalize_inputs(states, mus, actions, self.exclude_mu)
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
                drop_last=True
            )
            val_loader = DataLoader(
                DynamicsDataset(
                    inputs[idx][split_point:], norm_diffs[idx][split_point:]
                ),
                batch_size=self.batch_size,
                drop_last=True
            )
            best_val_loss = torch.inf
            best_train_loss = torch.inf
            best_model = deepcopy(m.state_dict())
            early_stopper = EarlyStopper(patience=30, min_improvement=0.005)
            for t in tqdm(range(self.dynamics_epochs)):
                train_loss = 0.0
                m.train()
                for batch_inputs, batch_norm_diffs in train_loader:
                    if self.adversarial_epsilon is not None:
                        batch_inputs_perturbed = self.create_adversarial_samples(
                            m,
                            batch_inputs.clone(),
                            batch_norm_diffs,
                        )
                    else:
                        batch_inputs_perturbed = None
                    out_means, out_vars = m(batch_inputs)
                    loss = nll_loss(
                        out_means.reshape(-1),
                        batch_norm_diffs.reshape(-1),
                        out_vars.reshape(-1),
                    )
                    if batch_inputs_perturbed is not None:
                        out_means_perturbed, out_vars_perturbed = m(batch_inputs_perturbed)
                        loss += nll_loss(
                            out_means_perturbed.reshape(-1),
                            batch_norm_diffs.reshape(-1),
                            out_vars_perturbed.reshape(-1),
                        )
                    # Regularization penalty for min_logvar and max_logvar
                    loss += 0.001 * (m.max_logvar.sum() - m.min_logvar.sum())
                    opt.zero_grad()
                    loss.backward()
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