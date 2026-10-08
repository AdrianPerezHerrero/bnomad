"""One model-owned transition posterior for shared A/Gamma policies."""

import torch

from ._parameter_posterior import make_pair_prior


class TransitionHistory:
    def __init__(self, owner, name, count=1):
        self.owner, self.name, self.count = owner, name, count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [getattr(self.owner, self.name)] * len(range(*index.indices(self.count)))
        return getattr(self.owner, self.name)

    def __setitem__(self, index, value):
        pass

    def __len__(self):
        return self.count

    def __iter__(self):
        return iter([getattr(self.owner, self.name)] * self.count)

    def append(self, value):
        self.count += 1

    def copy(self):
        return [getattr(self.owner, self.name)] * self.count


class TransitionPolicy:
    def __init__(self, A, Gamma, degrees, matrix_policy, noise_policy):
        self.scope = 'shared'
        self.prior = make_pair_prior(A, Gamma, degrees, matrix_policy, noise_policy)
        self.posterior = self.prior
        self.A, self.Gamma = self.prior.get_mean(), self.prior.get_scale()

    def attach(self, emission):
        count = len(emission.A)
        emission.transition_policy = self
        emission.A = TransitionHistory(self, 'A', count)
        emission.Gamma = TransitionHistory(self, 'Gamma', count)
        emission.A_def = self.prior.get_mean()
        emission.Gamma_def = self.prior.get_scale()
        emission.internal_params = self.posterior

    def kl(self):
        return self.posterior.kl_divergence(self.prior)

    def convert(self, target):
        method = {'numpy': 'to_numpy', 'torch': 'to_torch',
                  'cpu': 'to_cpu', 'cuda': 'to_cuda'}[target]
        getattr(self.prior, method)()
        if self.posterior is not self.prior:
            getattr(self.posterior, method)()
        self.A = self.posterior.get_mean()
        self.Gamma = self.posterior.get_scale(final=True)

    def update(self, emissions, responsibilities, *, online=False, current_index=None):
        observations, latents = [], []
        cov_y = torch.zeros_like(self.Gamma)
        cov_x = torch.zeros_like(self.Gamma)
        cov_cross = torch.zeros_like(self.Gamma)
        mass = 0.0
        for cluster, emission in enumerate(emissions):
            positions = [len(emission.indexes) - 1] if online else range(1, len(emission.indexes))
            for position in positions:
                if position < 1:
                    continue
                index = emission.indexes[position]
                if online and index != current_index:
                    continue
                weight = float(responsibilities[cluster] if online else responsibilities[index, cluster])
                if weight <= 0:
                    continue
                root = weight ** 0.5
                mean = emission.f_star_sm[position + 1].reshape(-1, 1)
                previous = emission.f_star_sm[position].reshape(-1, 1)
                covariance = emission.cov_f_sm[position + 1]
                previous_cov = emission.cov_f_sm[position]
                filtered = emission.cov_f[position]
                predicted = self.A @ filtered @ self.A.T + self.Gamma
                gain = torch.cholesky_solve(self.A @ filtered.T,
                                            emission._chol_spd(predicted)).T
                observations.append(root * mean)
                latents.append(root * previous)
                cov_y += weight * covariance
                cov_x += weight * previous_cov
                cov_cross += weight * covariance @ gain.T
                mass += weight
        if not mass:
            return
        base = self.posterior if online else self.prior
        self.posterior = base.posterior(
            mass, torch.cat(observations, dim=1), torch.cat(latents, dim=1),
            cov_y, cov_x, cov_cross)
        self.A, self.Gamma = self.posterior.get_mean(), self.posterior.get_scale(final=True)
        for emission in emissions:
            emission.internal_params = self.posterior
            if emission.var:
                emission.var[-1] = torch.diag(self.Gamma).sqrt().reshape(-1, 1)
