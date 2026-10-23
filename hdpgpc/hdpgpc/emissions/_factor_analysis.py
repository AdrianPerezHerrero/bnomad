"""Affine factor-analysis observations for the native HDP-GPC LDS.

There is no assignment engine here. Each cluster learns [intercept, C] with
normal/inverse-gamma rows; states, proposals and HDP weights remain native.
"""
from dataclasses import dataclass

import numpy as np
import torch

from ._fixed_observation import AffineGaussianObservation, _chol_cov
from ..dynamics._parameter_posterior import ConstrainedPairPosterior


def estimate_mfa_observation(observations, grid, *, components, max_samples=400,
                             random_state=0, sampling='uniform', noise_floor=1e-6):
    """Label-free spectral initialization, not a separately fitted mixture.

    Scale principal directions to a unit factor prior, subtracting the mean
    residual noise from their variances. All cluster parameters subsequently
    learn in HDPGPC; this calibration only fixes a common prior center.
    """
    from ._pca import estimate_pca_observation
    estimate = estimate_pca_observation(observations, grid, components=components,
        max_beats=max_samples, random_state=random_state, sampling=sampling,
        noise_floor=noise_floor)
    scales = np.sqrt(np.maximum(estimate.latent_variances
        - np.diag(estimate.residual_covariance).mean(), noise_floor))
    return AffineGaussianObservation(estimate.components*scales,
        estimate.residual_covariance, estimate.grid, estimate.mean, np.eye(components))


class FactorAnalysisPosterior(ConstrainedPairPosterior):
    """Joint Gaussian intercept/loading rows conditional on diagonal noise.

    b_j | s_j ~ N(m_j, s_j K^-1), s_j ~ IG(n/2+1, n*scale_j/2).
    The intercept is a parameter, never an additional evolving LDS state.
    """
    affine_intercept = True

    @classmethod
    def prior(cls, C, Sigma, degrees, mean_precision=1., loading_precision=1.):
        C = torch.as_tensor(C, dtype=torch.float64)
        mean = torch.cat((C.new_zeros((len(C), 1)), C), dim=1)
        precision = torch.diag(C.new_tensor([mean_precision] + [loading_precision] * C.shape[1]))
        return cls(mean, precision, degrees, Sigma, noise='diagonal')

    def get_mean(self):
        return self.m_mean[:, 1:]

    def posterior(self, n_k, y1, y2, cov, cov_, cov_cross,
                  sse_matrix=None, annealing=False, *, intercept_weights=None):
        if sse_matrix is not None:
            raise ValueError('Factor analysis observes the original feature space directly')
        if intercept_weights is None:
            # Native sequential updates contain one sqrt(responsibility)-scaled
            # column. Unweighted full-data legacy calls contain unit columns.
            intercept_weights = y2.new_full((y2.shape[1],), float(n_k / y2.shape[1]) ** .5)
        roots = torch.as_tensor(intercept_weights, dtype=y2.dtype, device=y2.device).reshape(1, -1)
        if roots.shape[1] != y2.shape[1] or not torch.allclose(roots.square().sum(), roots.new_tensor(n_k)):
            raise ValueError('Intercept weights must preserve the observation responsibility mass')
        augmented = torch.cat((roots, y2), dim=0)
        latent_cov = cov_.new_zeros((len(cov_) + 1, len(cov_) + 1))
        latent_cov[1:, 1:] = cov_
        cross = cov_cross.new_zeros((len(cov_cross), cov_cross.shape[1] + 1))
        cross[:, 1:] = cov_cross
        precision = self.m_r_cov + augmented @ augmented.T + latent_cov
        precision = .5*(precision+precision.T)
        information = self.m_mean @ self.m_r_cov + y1 @ augmented.T + cross
        mean = torch.cholesky_solve(information.T, torch.linalg.cholesky(precision)).T
        error = y1-mean @ augmented
        delta = mean-self.m_mean
        # Only diagonal residual moments are needed for inverse-gamma rows.
        # Avoid forming a feature-by-feature scatter matrix for image data.
        residual = (error.square().sum(1) + cov.diag()
            - 2*(mean*cross).sum(1) + ((mean @ latent_cov)*mean).sum(1)
            + ((delta @ self.m_r_cov)*delta).sum(1))
        scale = torch.diag((self.n0*self.scale.diag()+residual)/(self.n0+n_k))
        return type(self)(mean, precision, self.n0+n_k, scale, noise=self.noise)

    def posterior_natural_gradient(self, n_k, y1, y2, cov, cov_, cov_cross,
                                   sse_matrix=None, annealing=False, step_size=.2):
        root = step_size ** .5
        return self.posterior(step_size*n_k, root*y1, root*y2,
                              step_size*cov, step_size*cov_, step_size*cov_cross,
                              sse_matrix, annealing)

    def observation(self, grid, baseline, latent_covariance):
        inverse = torch.cholesky_inverse(torch.linalg.cholesky(self.m_r_cov))
        a = self.n0 / 2 + 1
        b = self.n0 * torch.diag(self.scale) / 2
        array = lambda x: torch.as_tensor(x).detach().cpu().numpy().copy()
        return FactorAnalysisEmission(
            C=array(self.get_mean()), Sigma=array(self.scale),
            observation_grid=array(grid),
            offset=array(baseline).reshape(-1) + array(self.m_mean[:, 0]),
            latent_covariance=array(latent_covariance),
            column_covariance=array(inverse),
            expected_precision=array(a / b),
            expected_log_precision=array(torch.digamma(b.new_tensor(a)) - b.log()))


@dataclass(frozen=True)
class FactorAnalysisEmission(AffineGaussianObservation):
    """Immutable affine snapshot including loading and intercept uncertainty.

    condition() is the Gaussian variational state coordinate update using
    E[noise^-1 B B.T]. The LDS supplies its prior/smoothed state moments.
    """
    column_covariance: np.ndarray
    expected_precision: np.ndarray
    expected_log_precision: np.ndarray

    def __post_init__(self):
        # Diagonal feature noise needs a positivity check, not an O(d^3)
        # eigendecomposition whenever a sequential posterior is refreshed.
        C = np.asarray(self.C, dtype=float)
        if C.ndim != 2 or min(C.shape) < 1:
            raise ValueError('Factor loadings must be a nonempty matrix')
        for name, shape in (('C', C.shape), ('Sigma', (len(C), len(C))),
                ('observation_grid', (len(C), 1)), ('offset', (len(C),)),
                ('latent_covariance', (C.shape[1], C.shape[1]))):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError('Invalid factor observation ' + name)
            if name == 'Sigma' and (not np.array_equal(value, np.diag(np.diag(value)))
                                    or (np.diag(value) <= 0).any()):
                raise ValueError('Factor noise must be positive and diagonal')
            if name == 'latent_covariance' and (not np.allclose(value, value.T)
                    or np.linalg.eigvalsh(value).min() <= 0):
                raise ValueError('Factor state prior must be positive definite')
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        for name, shape in (
                ('column_covariance', (self.latent_dim+1, self.latent_dim+1)),
                ('expected_precision', (self.observation_dim,)),
                ('expected_log_precision', (self.observation_dim,))):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError('Invalid factor-analysis ' + name)
            if name == 'expected_precision' and (value <= 0).any():
                raise ValueError('Expected precision must be positive')
            if name == 'column_covariance':
                if not np.allclose(value, value.T) or np.linalg.eigvalsh(value).min() <= 0:
                    raise ValueError('Column covariance must be positive definite')
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def _factor_tensors(self, reference):
        ref = torch.as_tensor(reference)
        make = lambda value: torch.as_tensor(value.copy(), dtype=ref.dtype, device=ref.device)
        return (make(self.column_covariance), make(self.expected_precision),
                make(self.expected_log_precision))

    def encode_likelihood(self, observation, *, noise=None):
        y = torch.as_tensor(observation).reshape(-1, 1)
        C, _, offset = self._tensors(y, None)
        V, tau, _ = self._factor_tensors(y)
        precision = C.T @ (tau[:, None] * C) + self.observation_dim * V[1:, 1:]
        information = C.T @ (tau[:, None] * (y-offset)) - self.observation_dim * V[1:, :1]
        return .5*(precision+precision.T), information

    def condition_batch(self, prior_mean, prior_covariance, observations, *, responsibilities, noise=None):
        mean = torch.as_tensor(prior_mean).reshape(-1, 1)
        values = torch.as_tensor(observations, dtype=mean.dtype, device=mean.device).reshape(-1, self.observation_dim)
        weights = torch.as_tensor(responsibilities, dtype=mean.dtype, device=mean.device).reshape(-1)
        if len(weights) != len(values) or not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError('Invalid factor observation responsibilities')
        C, _, offset = self._tensors(mean, None)
        V, tau, _ = self._factor_tensors(mean)
        H, _ = self.encode_likelihood(values[0])
        information = C.T @ (tau[:, None]*(values.T-offset)) - self.observation_dim*V[1:, :1]
        prior_precision = torch.cholesky_inverse(_chol_cov(torch.as_tensor(prior_covariance).to(mean)))
        unique, inverse = torch.unique(weights, return_inverse=True)
        covariance = torch.cholesky_inverse(_chol_cov(prior_precision[None] + unique[:, None, None]*H))
        covariance = covariance[inverse]
        natural = prior_precision @ mean + weights[None]*information
        return covariance @ natural.T.unsqueeze(-1), covariance

    def expected_log_likelihood_batch(self, observations, latent_mean, latent_covariance, *, noise=None):
        means = torch.as_tensor(latent_mean).reshape(-1, self.latent_dim)
        values = torch.as_tensor(observations, dtype=means.dtype, device=means.device).reshape(-1, self.observation_dim)
        covariance = torch.as_tensor(latent_covariance).to(means).reshape(-1, self.latent_dim, self.latent_dim)
        C, _, offset = self._tensors(means, None)
        V, tau, log_tau = self._factor_tensors(means)
        residual = values-offset.T-means @ C.T
        quadratic = (residual.square()*tau).sum(-1)
        quadratic += torch.einsum('ij,nji->n', C.T @ (tau[:, None]*C), covariance)
        u = torch.cat((means.new_ones((len(means), 1)), means), 1)
        uncertain = torch.einsum('ni,ij,nj->n', u, V, u)
        uncertain = uncertain + torch.einsum('ij,nji->n', V[1:, 1:], covariance)
        return .5*(log_tau.sum()-self.observation_dim*np.log(2*np.pi)
                   -quadratic-self.observation_dim*uncertain)

    def expected_log_likelihood(self, observation, latent_mean, latent_covariance, *, noise=None):
        return self.expected_log_likelihood_batch(observation, latent_mean, latent_covariance)[0]

    def predictive_moments(self, latent_mean, latent_covariance, *, noise=None):
        mean = torch.as_tensor(latent_mean).reshape(-1, 1)
        cov = torch.as_tensor(latent_covariance).to(mean)
        C, Sigma, offset = self._tensors(mean, noise)
        V, _, _ = self._factor_tensors(mean)
        u = torch.cat((mean.new_ones((1, 1)), mean))
        uncertainty = (u.T @ V @ u).squeeze() + torch.trace(V[1:, 1:] @ cov)
        return offset+C@mean, C@cov@C.T + Sigma + torch.diag(torch.diag(Sigma)*uncertainty)

    def state_dict(self):
        state = super().state_dict()
        state.update(type='factor_analysis', column_covariance=self.column_covariance.copy(),
                     expected_precision=self.expected_precision.copy(),
                     expected_log_precision=self.expected_log_precision.copy())
        return state

    @classmethod
    def from_state_dict(cls, state):
        if state.get('type') != 'factor_analysis':
            raise ValueError('Expected factor_analysis observation state')
        return cls(state['C'], state['Sigma'], state['grid'], state['offset'],
                   state['latent_covariance'], state['column_covariance'],
                   state['expected_precision'], state['expected_log_precision'])


def refresh_factor_observation(emission, posterior=None):
    """Attach a cluster-local snapshot to existing scoring and filtering layers."""
    if not getattr(emission, 'factor_analysis', False):
        return
    posterior = emission.observation_params if posterior is None else posterior
    snapshot = posterior.observation(emission.direct_observation_grid,
                                    emission.direct_observation_offset,
                                    emission.ini_cov_def)
    emission.fixed_observation_model = snapshot
    emission.gp.fixed_observation_model = snapshot
    emission.__dict__.pop('_factor_local_moments', None)


def observation_prior(emission, C, Sigma):
    """Construct the original FA prior through the native parameter lifecycle."""
    strength = emission.lds_pair_policies['observation'][1].strength
    return FactorAnalysisPosterior.prior(C, Sigma, emission.free_deg_MNIV*strength,
        emission.factor_mean_precision, emission.factor_loading_precision)
