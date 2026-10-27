"""Pooled conjugate MFA: shared loadings/noise and cluster-specific intercepts.

For each feature row the joint regression coefficients are [mu_1,...,mu_K,C].
A single normal/inverse-gamma posterior retains intercept/loading correlations;
cluster emissions expose its [mu_k,C] marginals for existing LDS conditioning.
"""
import torch

from ._factor_analysis import FactorAnalysisPosterior, refresh_factor_observation
from ._observation import ObservationHistory
from ..dynamics._parameter_posterior import ConstrainedPairPosterior


class SharedFactorAnalysisPolicy:
    scope = 'shared'
    factor_analysis = True

    def __init__(self, C, Sigma, degrees, mean_precision=1., loading_precision=1.):
        self.initial_C = torch.as_tensor(C, dtype=torch.float64).clone()
        self.initial_Sigma = torch.as_tensor(Sigma, dtype=torch.float64).clone()
        self.degrees = degrees
        self.mean_precision = mean_precision
        self.loading_precision = loading_precision
        self.intercepts = 0
        self.prior = self._prior(0)
        self.posterior = self.prior
        self.C, self.Sigma = self.initial_C.clone(), self.initial_Sigma.clone()

    def _prior(self, count):
        C = self.initial_C
        mean = torch.cat((C.new_zeros((len(C), count)), C), dim=1)
        precision = torch.diag(C.new_tensor(
            [self.mean_precision]*count + [self.loading_precision]*C.shape[1]))
        return ConstrainedPairPosterior(mean, precision, self.degrees,
                                        self.initial_Sigma, noise='diagonal')

    def attach(self, emission):
        # Preserve the local mean on proposal copies/reordering. A new cluster
        # starts at its own original mean prior, not an existing cluster's slot.
        previous_owner = getattr(emission, 'observation_policy', None)
        local = emission.observation_params
        mean = torch.as_tensor(local.m_mean).clone()
        mean[:, 1:] = self.C
        inverse = torch.cholesky_inverse(torch.linalg.cholesky(
            torch.as_tensor(local.m_r_cov)))
        # Shared loading uncertainty comes from the global posterior. Keep the
        # existing local intercept covariance/cross-covariance where possible.
        global_inverse = torch.cholesky_inverse(torch.linalg.cholesky(self.posterior.m_r_cov))
        if self.intercepts == 0 or not getattr(previous_owner, 'factor_analysis', False):
            inverse[0, 1:] = 0
            inverse[1:, 0] = 0
            inverse[1:, 1:] = global_inverse[self.intercepts:, self.intercepts:]
        # Once fitted, existing marginal cross-covariances must stay intact.
        emission.observation_params = FactorAnalysisPosterior(
            mean, torch.linalg.inv(inverse), self.posterior.n0, self.Sigma, noise='diagonal')
        emission.observation_policy = self
        emission.C = ObservationHistory(self, 'C', emission)
        emission.Sigma = ObservationHistory(self, 'Sigma', emission)
        emission.C_def, emission.Sigma_def = self.initial_C, self.initial_Sigma
        refresh_factor_observation(emission)
        variance = self.Sigma.diag().sqrt().reshape(-1, 1)
        if emission.y_var:
            emission.y_var[-1] = variance
        else:
            emission.y_var = [variance]
        emission.gp.assign_alpha_ini(self.Sigma, emission.Gamma[0])

    def kl(self):
        return self.posterior.kl_divergence(self.prior)

    def convert(self, target):
        method = {'numpy': 'to_numpy', 'torch': 'to_torch',
                  'cpu': 'to_cpu', 'cuda': 'to_cuda'}[target]
        getattr(self.prior, method)()
        if self.posterior is not self.prior:
            getattr(self.posterior, method)()
        for name in ('initial_C', 'initial_Sigma'):
            value = getattr(self, name)
            if target == 'numpy':
                value = torch.as_tensor(value).detach().cpu().numpy()
            else:
                value = torch.as_tensor(value)
                if target in ('cpu', 'cuda'):
                    value = getattr(value, target)()
            setattr(self, name, value)
        self.C = self.posterior.m_mean[:, self.intercepts:]
        self.Sigma = self.posterior.get_scale()

    def update(self, emissions, responsibilities, *, online=False, current_index=None):
        if online:
            raise ValueError('Shared MFA currently requires offline conjugate_batch updates')
        count, latent = len(emissions), self.C.shape[1]
        dimension = count + latent
        prior = self._prior(count)
        data, design = [], []
        latent_cov = self.C.new_zeros((dimension, dimension))
        mass = 0.
        for cluster, emission in enumerate(emissions):
            for position, index in enumerate(emission.indexes):
                weight = float(responsibilities[index, cluster])
                if weight <= 0:
                    continue
                root = weight**.5
                observation = emission.y_train[position].reshape(-1, 1) - emission.direct_observation_offset
                column = self.C.new_zeros((dimension, 1))
                column[cluster] = root
                column[count:] = root * emission.f_star_sm[position+1].reshape(-1, 1)
                data.append(root * observation)
                design.append(column)
                latent_cov[count:, count:] += weight * emission.cov_f_sm[position+1]
                mass += weight
        if not mass:
            return
        Y, U = torch.cat(data, 1), torch.cat(design, 1)
        precision = prior.m_r_cov + U @ U.T + latent_cov
        precision = .5*(precision+precision.T)
        information = prior.m_mean @ prior.m_r_cov + Y @ U.T
        mean = torch.cholesky_solve(information.T, torch.linalg.cholesky(precision)).T
        error, delta = Y - mean @ U, mean-prior.m_mean
        residual = (error.square().sum(1) + ((mean @ latent_cov)*mean).sum(1)
                    + ((delta @ prior.m_r_cov)*delta).sum(1))
        scale = torch.diag((prior.n0*prior.scale.diag()+residual)/(prior.n0+mass))
        self.prior = prior
        self.posterior = ConstrainedPairPosterior(mean, precision, prior.n0+mass,
                                                   scale, noise='diagonal')
        self.intercepts = count
        self.C, self.Sigma = mean[:, count:], scale
        inverse = torch.cholesky_inverse(torch.linalg.cholesky(precision))
        for cluster, emission in enumerate(emissions):
            slots = [cluster] + list(range(count, dimension))
            covariance = inverse[slots][:, slots]
            emission.observation_params = FactorAnalysisPosterior(
                mean[:, slots], torch.linalg.inv(covariance), self.posterior.n0,
                scale, noise='diagonal')
            emission.observation_policy = self
            emission.C = ObservationHistory(self, 'C', emission)
            emission.Sigma = ObservationHistory(self, 'Sigma', emission)
            refresh_factor_observation(emission)
            if emission.y_var:
                emission.y_var[-1] = scale.diag().sqrt().reshape(-1, 1)
