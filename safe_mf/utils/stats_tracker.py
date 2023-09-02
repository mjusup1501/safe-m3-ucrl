import torch


class StatsTracker(object):
    _instance = None

    # singleton
    def __new__(self):
        if self._instance is None:
            self._instance = super(StatsTracker, self).__new__(self)
            self._state_mean = None
            self._state_sq = None
            self._state_count = 0

            self._mu_mean = None
            self._mu_sq = None
            self._mu_count = 0

            self._action_mean = None
            self._action_sq = None
            self._action_count = 0

            self._diff_mean = None
            self._diff_sq = None
            self._diff_count = 0
        return self._instance

    def reset(self):
        self._state_mean = None
        self._state_sq = None
        self._state_count = 0

        self._mu_mean = None
        self._mu_sq = None
        self._mu_count = 0

        self._action_mean = None
        self._action_sq = None
        self._action_count = 0

        self._diff_mean = None
        self._diff_sq = None
        self._diff_count = 0

    @property
    def state_std(self):
        return torch.sqrt(self._state_sq / (self._state_count - 1))

    @property
    def mu_std(self):
        return torch.sqrt(self._mu_sq / (self._mu_count - 1))

    @property
    def action_std(self):
        return torch.sqrt(self._action_sq / (self._action_count - 1))

    @property
    def diff_std(self):
        return torch.sqrt(self._diff_sq / (self._diff_count - 1))

    @property
    def state_mean(self):
        return self._state_mean

    @property
    def mu_mean(self):
        return self._mu_mean

    @property
    def action_mean(self):
        return self._action_mean

    @property
    def diff_mean(self):
        return self._diff_mean

    def update_states(self, states: torch.Tensor):
        """Assumes initial batch to be larger than 1"""
        if self.state_mean is None:
            self._state_mean = states.detach().mean(dim=0)
            self._state_sq = ((states - self.state_mean) ** 2).detach().sum(dim=0)
            self._state_count = states.shape[0]
        else:
            m = states.detach().mean(dim=0)
            sq = ((states - m) ** 2).detach().sum(dim=0)
            c = states.shape[0]

            new_count = self._state_count + c
            delt = m - self._state_mean
            new_mean = self._state_mean + delt * (c / new_count)
            new_state_sq = (
                self._state_sq + sq + delt**2 * (self._state_count * c / new_count)
            )
            self._state_mean = new_mean
            self._state_sq = new_state_sq
            self._state_count = new_count

    def update_mus(self, mus: torch.Tensor):
        """Assumes initial batch to be larger than 1"""
        if self.mu_mean is None:
            self._mu_mean = mus.detach().mean(dim=0)
            self._mu_sq = ((mus - self.mu_mean) ** 2).detach().sum(dim=0)
            self._mu_count = mus.shape[0]
        else:
            m = mus.detach().mean(dim=0)
            sq = ((mus - m) ** 2).detach().sum(dim=0)
            c = mus.shape[0]

            new_count = self._mu_count + c
            delt = m - self._mu_mean
            new_mean = self._mu_mean + delt * (c / new_count)
            new_mu_sq = self._mu_sq + sq + delt**2 * (self._mu_count * c / new_count)
            self._mu_mean = new_mean
            self._mu_sq = new_mu_sq
            self._mu_count = new_count

    def update_actions(self, actions: torch.Tensor):
        """Assumes initial batch to be larger than 1"""
        if self.action_mean is None:
            self._action_mean = actions.detach().mean(dim=0)
            self._action_sq = ((actions - self.action_mean) ** 2).detach().sum(dim=0)
            self._action_count = actions.shape[0]
        else:
            m = actions.detach().mean(dim=0)
            sq = ((actions - m) ** 2).detach().sum(dim=0)
            c = actions.shape[0]

            new_count = self._action_count + c
            delt = m - self._action_mean
            new_mean = self._action_mean + delt * (c / new_count)
            new_action_sq = (
                self._action_sq + sq + delt**2 * (self._action_count * c / new_count)
            )
            self._action_mean = new_mean
            self._action_sq = new_action_sq
            self._action_count = new_count

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

    def normalize_states(self, states: torch.Tensor):
        return (states - self.state_mean) / self.state_std

    def normalize_mus(self, mus: torch.Tensor):
        return (mus - self.mu_mean) / self.mu_std

    def normalize_actions(self, actions: torch.Tensor):
        return (actions - self.action_mean) / self.action_std

    def normalize_diffs(self, diffs: torch.Tensor):
        return (diffs - self.diff_mean) / self.diff_std

    def denormalize_states(self, states: torch.Tensor):
        return states * self.state_std + self.state_mean

    def denormalize_mus(self, mus: torch.Tensor):
        return mus * self.mu_std + self.mu_mean

    def denormalize_actions(self, actions: torch.Tensor):
        return actions * self.action_std + self.action_mean

    def denormalize_diffs(self, diffs: torch.Tensor):
        return diffs * self.diff_std + self.diff_mean
