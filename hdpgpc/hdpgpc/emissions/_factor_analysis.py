"""Standalone conjugate variational factor-analysis emission, without GP/HDP state.

Each observed row of B=[mu,W] has an independent Gaussian posterior. Continuous
factors are conditional on this component, not averaged across mixture labels.
Noise is a diagonal *variance*, treated as a point parameter. All local scores
are normalized variational evidence, including factor prior and entropy once.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Integral

import torch
from torch.distributions import MultivariateNormal, kl_divergence


@dataclass(frozen=True)
class FactorAnalysisPosterior:
    """Component-conditioned factors: mean (N,r), shared covariance (r,r).

    log_evidence has shape (N,). The covariance is observation-independent for
    this model; do not allocate N identical copies. Moments describe the state
    of the component at calculation time and are not updated in place.
    """
    mean: torch.Tensor
    covariance: torch.Tensor
    log_evidence: torch.Tensor


@dataclass(frozen=True)
class FactorAnalysisStatistics:
    """Weighted moments of u=[1,z]: occupancy, F, g and sum_i r_i h_i^2.

    Shapes are (), (r+1,r+1), (d,r+1), (d,). These moments can be added across
    minibatches before one batch update; adding posteriors would repeat priors.
    """
    occupancy: torch.Tensor
    factor_second_moment: torch.Tensor
    observation_factor: torch.Tensor
    observation_second_moment: torch.Tensor


@dataclass(frozen=True)
class FactorAnalysisMoments:
    """Compact observation moments: Cov(h)=factor@factor.T+diag(diagonal).

    These are exact moments. With parameter uncertainty, the posterior
    predictive distribution is generally not Gaussian. No d-by-d array is
    constructed. mean/diagonal have shape (d,), factor has shape (d,r).
    """
    mean: torch.Tensor
    factor: torch.Tensor
    diagonal: torch.Tensor

    @property
    def variance(self):
        return self.factor.square().sum(-1) + self.diagonal


class FactorAnalysisEmission:
    """One learnable Bayesian FA component for observations shaped (N,d).

    h=mu+Wz+epsilon, z~N(0,I), epsilon~N(0,diag(noise)). The prior is on
    augmented rows b_j=[mu_j,W_j]. Prior means broadcast to (d,r+1); precision
    is scalar times I, a diagonal (r+1,) vector, a common (r+1,r+1) matrix,
    or row-specific (d,r+1,r+1) matrices. Covariances/precisions must be SPD.

    Initialization is the proper prior. With a zero loading mean and independent
    centroid/loading priors, CAVI has a zero-loading stationary point: a future
    mixture initializer should call set_posterior with seeded nonzero loadings.
    This class deliberately does not choose clusters or run a training schedule.

    Noise can be overridden per operation, allowing a later model-level owner
    to supply one shared diagonal vector. Shared noise updates pool expected
    residual *sums* and occupancies before division, never sequential averages.
    Constructor/update inputs are detached and copied; no autograd optimization
    or process-wide dtype/RNG changes are performed.
    """

    def __init__(self, observation_dim, n_factors, *, loading_prior_mean=0.,
                 loading_prior_precision=1., noise=1., noise_floor=1e-8,
                 dtype=torch.float64, device='cpu'):
        for name, value in (('observation_dim', observation_dim), ('n_factors', n_factors)):
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if n_factors > observation_dim:
            raise ValueError('n_factors cannot exceed observation_dim')
        if dtype not in (torch.float32, torch.float64):
            raise ValueError('dtype must be torch.float32 or torch.float64')
        if not math.isfinite(float(noise_floor)) or noise_floor <= 0:
            raise ValueError('noise_floor must be finite and positive')
        self.observation_dim, self.n_factors = int(observation_dim), int(n_factors)
        self.dtype, self.device = dtype, torch.device(device)
        self.noise_floor = float(noise_floor)
        d, p = self.observation_dim, self.n_factors + 1
        self._prior_mean = self._broadcast(loading_prior_mean, (d, p), 'loading_prior_mean')
        precision = self._tensor(loading_prior_precision)
        if precision.ndim == 0 or precision.shape == (p,):
            precision = torch.eye(p, dtype=dtype, device=self.device) * precision
        self._prior_precision = self._broadcast(precision, (d, p, p), 'loading_prior_precision')
        prior_chol = self._spd(self._prior_precision, 'loading_prior_precision')
        self._prior_precision = .5 * (self._prior_precision + self._prior_precision.mT)
        covariance = torch.cholesky_solve(self._eye(p).expand(d, p, p), prior_chol)
        self._noise = self._noise_value(noise)
        self._occupancy = self._tensor(0.)
        self.set_posterior(self._prior_mean, covariance)

    def _tensor(self, value):
        return torch.as_tensor(value, dtype=self.dtype, device=self.device).detach().clone()

    def _broadcast(self, value, shape, name):
        try:
            result = torch.broadcast_to(self._tensor(value), shape).clone()
        except RuntimeError as error:
            raise ValueError(f'{name} must broadcast to {shape}') from error
        if not bool(torch.isfinite(result).all()):
            raise ValueError(f'{name} must be finite')
        return result

    @staticmethod
    def _spd(matrix, name):
        if (not bool(torch.isfinite(matrix).all())
                or not torch.allclose(matrix, matrix.mT, atol=1e-12, rtol=1e-7)):
            raise ValueError(f'{name} must be finite and symmetric')
        # Proper Gaussian precisions and I+E[W^T Psi^-1 W] need no heuristic
        # jitter. Reject invalid inputs instead of silently changing the model.
        try:
            return torch.linalg.cholesky(.5 * (matrix + matrix.mT))
        except torch.linalg.LinAlgError as error:
            raise ValueError(f'{name} must be positive definite') from error

    def _eye(self, dimension):
        return torch.eye(dimension, dtype=self.dtype, device=self.device)

    def _noise_value(self, value):
        variance = self._broadcast(value, (self.observation_dim,), 'noise')
        if not bool((variance > 0).all()):
            raise ValueError('noise must contain positive variances')
        return variance.clamp_min(self.noise_floor)

    def _observations(self, values):
        values = torch.as_tensor(values, dtype=self.dtype, device=self.device).detach()
        if (values.ndim != 2 or values.shape[1] != self.observation_dim
                or not bool(torch.isfinite(values).all())):
            raise ValueError(f'observations must be finite with shape (N,{self.observation_dim})')
        return values

    def _weights(self, weights, count):
        result = self._tensor(weights)
        if (result.shape != (count,) or not bool(torch.isfinite(result).all())
                or bool(((result < 0) | (result > 1)).any())):
            raise ValueError('responsibilities must have shape (N,) and lie in [0,1]')
        return result

    @staticmethod
    def _slices(count, chunk_size):
        if not isinstance(chunk_size, Integral) or isinstance(chunk_size, bool) or chunk_size < 1:
            raise ValueError('chunk_size must be a positive integer')
        return (slice(start, min(start + chunk_size, count))
                for start in range(0, count, chunk_size))

    @property
    def parameter_mean(self):
        return self._parameter_mean.clone()

    @property
    def parameter_covariance(self):
        return self._parameter_covariance.clone()

    @property
    def mean(self):
        return self._parameter_mean[:, 0].clone()

    @property
    def loadings(self):
        return self._parameter_mean[:, 1:].clone()

    @property
    def loading_covariance(self):
        return self._parameter_covariance[:, 1:, 1:].clone()

    @property
    def mean_variance(self):
        return self._parameter_covariance[:, 0, 0].clone()

    @property
    def prior_mean(self):
        return self._prior_mean.clone()

    @property
    def prior_precision(self):
        return self._prior_precision.clone()

    @property
    def noise(self):
        return self._noise.clone()

    @property
    def occupancy(self):
        return self._occupancy.clone()

    def set_posterior(self, mean, covariance):
        """Install a proper Gaussian initialization/posterior, retaining the prior."""
        d, p = self.observation_dim, self.n_factors + 1
        mean = self._broadcast(mean, (d, p), 'parameter mean')
        covariance = self._broadcast(covariance, (d, p, p), 'parameter covariance')
        chol = self._spd(covariance, 'parameter covariance')
        covariance = .5 * (covariance + covariance.mT)
        self._parameter_mean, self._parameter_covariance = mean, covariance
        self._parameter_chol = chol
        return self

    def set_noise(self, noise):
        self._noise = self._noise_value(noise)
        return self

    def _local_setup(self, noise, *, point_parameters=False):
        variance = self._noise if noise is None else self._noise_value(noise)
        mean, covariance = self._parameter_mean, self._parameter_covariance
        weighted_w = mean[:, 1:] / variance[:, None]
        precision = self._eye(self.n_factors) + mean[:, 1:].T @ weighted_w
        correction = mean.new_zeros(self.n_factors)
        uncertainty_chol = None
        if not point_parameters:
            weighted_cov = (covariance / variance[:, None, None]).sum(0)
            precision = precision + weighted_cov[1:, 1:]
            correction = weighted_cov[1:, 0]
            uncertainty_chol = self._spd(weighted_cov, 'weighted parameter covariance')
        # Weighted Gram products are symmetric analytically, but independent
        # reductions can leave tiny skew in entries whose true value is zero.
        precision = .5 * (precision + precision.mT)
        chol = self._spd(precision, 'factor precision')
        factor_cov = torch.cholesky_solve(self._eye(self.n_factors), chol)
        factor_cov = .5 * (factor_cov + factor_cov.mT)
        constant = (self.observation_dim * math.log(2 * math.pi)
                    + variance.log().sum() + 2 * chol.diagonal().log().sum())
        return variance, weighted_w, correction, chol, factor_cov, constant, uncertainty_chol

    def _local_chunk(self, values, setup, *, evidence=True):
        variance, weighted_w, correction, chol, _, constant, uncertainty_chol = setup
        centered = values - self._parameter_mean[:, 0]
        information = centered @ weighted_w - correction
        mean = torch.cholesky_solve(information.T, chol).T
        if not evidence:
            return mean, None
        # Complete the square without subtracting two large quadratic forms:
        # c-a^T S a = ||h-Bbar*u||_Psi^-1^2 + ||m||^2
        #             + u^T (sum_j V_j/psi_j) u, u=[1,m].
        residual = centered - mean @ self._parameter_mean[:, 1:].T
        energy = (residual.square() / variance).sum(1) + mean.square().sum(1)
        if uncertainty_chol is not None:
            augmented = torch.cat((mean.new_ones((len(mean), 1)), mean), 1)
            energy = energy + (augmented @ uncertainty_chol).square().sum(1)
        return mean, -.5 * (constant + energy)

    @torch.no_grad()
    def posterior_factors(self, observations, *, noise=None, chunk_size=1024):
        """Return q(z|this component) and its optimized variational evidence."""
        values = self._observations(observations)
        setup = self._local_setup(noise)
        means = values.new_empty((len(values), self.n_factors))
        evidence = values.new_empty(len(values))
        for part in self._slices(len(values), chunk_size):
            means[part], evidence[part] = self._local_chunk(values[part], setup)
        return FactorAnalysisPosterior(means, setup[4], evidence)

    @torch.no_grad()
    def local_evidence(self, observations, *, noise=None, chunk_size=1024):
        """Normalized optimized local bound; includes z prior and entropy once."""
        values, setup = self._observations(observations), self._local_setup(noise)
        evidence = values.new_empty(len(values))
        for part in self._slices(len(values), chunk_size):
            _, evidence[part] = self._local_chunk(values[part], setup)
        return evidence

    @torch.no_grad()
    def marginal_log_prob(self, observations, *, noise=None, chunk_size=1024):
        """Exact Gaussian density at posterior-mean B, ignoring B uncertainty.

        This plug-in marginal is a diagnostic, not the variational assignment
        evidence or an exact Bayesian posterior predictive density.
        """
        values = self._observations(observations)
        setup = self._local_setup(noise, point_parameters=True)
        result = values.new_empty(len(values))
        for part in self._slices(len(values), chunk_size):
            _, result[part] = self._local_chunk(values[part], setup)
        return result

    def _check_posterior(self, posterior, count):
        mean, cov = self._tensor(posterior.mean), self._tensor(posterior.covariance)
        if mean.shape != (count, self.n_factors) or cov.shape != (self.n_factors, self.n_factors):
            raise ValueError('factor posterior has incompatible dimensions')
        if not bool(torch.isfinite(mean).all()):
            raise ValueError('factor posterior mean must be finite')
        self._spd(cov, 'factor posterior covariance')
        return mean, cov

    @torch.no_grad()
    def sufficient_statistics(self, observations, responsibilities, *, posterior=None,
                              noise=None, chunk_size=1024):
        """Collect F, g, h^2 once with fractional weights (never gate at 1)."""
        values = self._observations(observations)
        weights = self._weights(responsibilities, len(values))
        setup = self._local_setup(noise) if posterior is None else None
        means, cov = (None, setup[4]) if posterior is None else self._check_posterior(posterior, len(values))
        p = self.n_factors + 1
        second = values.new_zeros((p, p))
        cross = values.new_zeros((self.observation_dim, p))
        square = values.new_zeros(self.observation_dim)
        for part in self._slices(len(values), chunk_size):
            mean = self._local_chunk(values[part], setup, evidence=False)[0] if means is None else means[part]
            augmented = torch.cat((mean.new_ones((len(mean), 1)), mean), 1)
            weight = weights[part]
            second += augmented.T @ (weight[:, None] * augmented)
            cross += values[part].T @ (weight[:, None] * augmented)
            square += (weight[:, None] * values[part].square()).sum(0)
        occupancy = weights.sum()
        second[1:, 1:] += occupancy * cov
        # Enforce F = sum_i r_i E[u_i u_i^T] symmetry before dividing by
        # small pixel variances amplifies harmless accumulation roundoff.
        second = .5 * (second + second.T)
        return FactorAnalysisStatistics(occupancy, second, cross, square)

    def _statistics(self, stats):
        p, d = self.n_factors + 1, self.observation_dim
        parts = tuple(self._tensor(value) for value in
                      (stats.occupancy, stats.factor_second_moment,
                       stats.observation_factor, stats.observation_second_moment))
        if (tuple(value.shape for value in parts) != ((), (p, p), (d, p), (d,))
                or not all(bool(torch.isfinite(value).all()) for value in parts)
                or bool(parts[0] < 0) or bool((parts[3] < 0).any())
                or not torch.allclose(parts[1], parts[1].T)
                or not torch.allclose(parts[1][0, 0], parts[0])):
            raise ValueError('incompatible or invalid factor-analysis sufficient statistics')
        # Statistics may also come from external/minibatch accumulation.
        # Canonicalize only AFTER validating their shape, finiteness and
        # symmetry; invalid inputs still fail rather than being repaired.
        return parts[0], .5 * (parts[1] + parts[1].T), parts[2], parts[3]

    @torch.no_grad()
    def update_from_statistics(self, statistics, *, noise=None):
        """Batch conjugate q(B) update from the original prior, never q_old(B)."""
        occupancy, second, cross, _ = self._statistics(statistics)
        variance = self._noise if noise is None else self._noise_value(noise)
        precision = self._prior_precision + second / variance[:, None, None]
        chol = self._spd(precision, 'parameter posterior precision')
        information = (self._prior_precision @ self._prior_mean.unsqueeze(-1)
                       + (cross / variance[:, None]).unsqueeze(-1))
        mean = torch.cholesky_solve(information, chol).squeeze(-1)
        p = self.n_factors + 1
        covariance = torch.cholesky_solve(self._eye(p).expand(self.observation_dim, p, p), chol)
        covariance = .5 * (covariance + covariance.mT)
        self.set_posterior(mean, covariance)
        self._occupancy = occupancy
        return self

    @torch.no_grad()
    def parameter_kl(self, *, per_dimension=False):
        """KL(q(B)||p(B)), counting the joint centroid/loading row once."""
        posterior = MultivariateNormal(self._parameter_mean, scale_tril=self._parameter_chol)
        prior = MultivariateNormal(self._prior_mean, precision_matrix=self._prior_precision)
        rows = kl_divergence(posterior, prior)
        return rows if per_dimension else rows.sum()

    @torch.no_grad()
    def expected_residual(self, observations, *, posterior=None, noise=None, chunk_size=1024):
        """Return E[(h_j-b_j u)^2] with BOTH z and parameter uncertainty."""
        values = self._observations(observations)
        if posterior is None:
            posterior = self.posterior_factors(values, noise=noise, chunk_size=chunk_size)
        means, cov = self._check_posterior(posterior, len(values))
        result = torch.empty_like(values)
        w = self._parameter_mean[:, 1:]
        latent_variance = ((w @ cov) * w).sum(1)
        uncertain_variance = (self._parameter_covariance[:, 1:, 1:] * cov.T).sum((1, 2))
        for part in self._slices(len(values), chunk_size):
            mean = means[part]
            u = torch.cat((mean.new_ones((len(mean), 1)), mean), 1)
            # Row-wise Cholesky evaluates u^T V_j u as a nonnegative square.
            param_variance = torch.einsum('np,dpq->ndq', u, self._parameter_chol).square().sum(-1)
            result[part] = ((values[part] - u @ self._parameter_mean.T).square()
                            + param_variance + latent_variance + uncertain_variance)
        return result

    @torch.no_grad()
    def noise_statistics(self, statistics):
        """Expected residual sums and occupancy, suitable for shared pooling."""
        occupancy, second, cross, square = self._statistics(statistics)
        mean = self._parameter_mean
        residual = (square - 2 * (mean * cross).sum(1)
                    + ((mean @ second) * mean).sum(1)
                    + (self._parameter_covariance * second.T).sum((1, 2)))
        scale = square.abs() + 2 * (mean * cross).abs().sum(1) + ((mean @ second) * mean).abs().sum(1)
        if bool((residual < -128 * torch.finfo(self.dtype).eps * scale.clamp_min(1.)).any()):
            raise FloatingPointError('Expected residual scatter is negative; check sufficient statistics')
        return residual.clamp_min(0), occupancy

    @staticmethod
    @torch.no_grad()
    def estimate_shared_noise(statistics, *, noise_floor=1e-8, previous_noise=None):
        """Pool (residual_sum, occupancy) across components before dividing.

        All pairs must use the same observation units/dimension, dtype and device.
        For zero total occupancy retain previous_noise, or fail if absent.
        """
        if not math.isfinite(float(noise_floor)) or noise_floor <= 0:
            raise ValueError('noise_floor must be finite and positive')
        pairs = list(statistics)
        if not pairs:
            raise ValueError('at least one noise-statistics pair is required')
        residual, mass = pairs[0]
        residual, mass = residual.detach().clone(), mass.detach().clone()
        if residual.ndim != 1 or mass.ndim != 0:
            raise ValueError('noise statistics must have shapes (d,) and ()')
        for value, count in pairs:
            if (value.shape != residual.shape or count.ndim != 0
                    or value.dtype != residual.dtype or value.device != residual.device
                    or count.dtype != mass.dtype or count.device != mass.device
                    or not bool(torch.isfinite(value).all() and torch.isfinite(count))
                    or bool((value < 0).any() or count < 0)):
                raise ValueError('invalid or incompatible noise statistics')
        for value, count in pairs[1:]:
            residual += value
            mass += count
        if bool(mass == 0):
            if previous_noise is None:
                raise ValueError('cannot estimate noise without positive occupancy')
            previous = torch.as_tensor(previous_noise, dtype=residual.dtype, device=residual.device)
            if (previous.shape != residual.shape or not bool(torch.isfinite(previous).all())
                    or not bool((previous > 0).all())):
                raise ValueError('previous_noise must contain positive finite variances')
            return previous.clamp_min(noise_floor).clone()
        return (residual / mass).clamp_min(noise_floor)

    def update_noise(self, statistics):
        """Single-component convenience; mixtures should pool all components."""
        return self.set_noise(self.estimate_shared_noise(
            [self.noise_statistics(statistics)], noise_floor=self.noise_floor,
            previous_noise=self._noise))

    @torch.no_grad()
    def marginal_moments(self, *, noise=None):
        """Plug-in Gaussian moments at posterior-mean B; omit B uncertainty."""
        variance = self._noise if noise is None else self._noise_value(noise)
        return FactorAnalysisMoments(self.mean, self.loadings, variance.clone())

    @torch.no_grad()
    def predictive_moments(self, latent_mean=None, latent_covariance=None, *, noise=None):
        """Exact B-integrated moments for a Gaussian factor independent of B.

        Defaults describe a new z~N(0,I). A supplied factor is a mean-field
        variational reconstruction, not an exact conditional predictive density.
        """
        r = self.n_factors
        mean = self._parameter_mean.new_zeros(r) if latent_mean is None else self._tensor(latent_mean)
        covariance = self._eye(r) if latent_covariance is None else self._tensor(latent_covariance)
        if mean.shape != (r,) or covariance.shape != (r, r) or not bool(torch.isfinite(mean).all()):
            raise ValueError('latent moments have incompatible dimensions or nonfinite values')
        factor = self._parameter_mean[:, 1:] @ self._spd(covariance, 'latent covariance')
        u = torch.cat((mean.new_ones(1), mean))
        second = torch.outer(u, u)
        second[1:, 1:] += covariance
        diagonal = (self._parameter_covariance * second.T).sum((1, 2))
        diagonal += self._noise if noise is None else self._noise_value(noise)
        return FactorAnalysisMoments(self._parameter_mean @ u, factor, diagonal)

    def copy(self):
        """Independent copy on the same device, with no tensor aliases."""
        return deepcopy(self)

    def state_dict(self):
        """Versioned JSON-compatible state; no callbacks, pickle or caches."""
        return {'type': 'factor_analysis', 'version': 1,
                'observation_dim': self.observation_dim, 'n_factors': self.n_factors,
                'dtype': str(self.dtype).removeprefix('torch.'), 'noise_floor': self.noise_floor,
                'loading_prior_mean': self._prior_mean.cpu().tolist(),
                'loading_prior_precision': self._prior_precision.cpu().tolist(),
                'parameter_mean': self._parameter_mean.cpu().tolist(),
                'parameter_covariance': self._parameter_covariance.cpu().tolist(),
                'noise': self._noise.cpu().tolist(), 'occupancy': float(self._occupancy)}

    @classmethod
    def from_state_dict(cls, state, *, device='cpu', dtype=None):
        if state.get('type') != 'factor_analysis' or state.get('version') != 1:
            raise ValueError('unsupported factor-analysis state type or version')
        saved_dtype = {'float32': torch.float32, 'float64': torch.float64}.get(state.get('dtype'))
        if saved_dtype is None:
            raise ValueError('unsupported saved dtype')
        result = cls(state['observation_dim'], state['n_factors'],
                     loading_prior_mean=state['loading_prior_mean'],
                     loading_prior_precision=state['loading_prior_precision'],
                     noise=state['noise'], noise_floor=state['noise_floor'],
                     dtype=saved_dtype if dtype is None else dtype, device=device)
        result.set_posterior(state['parameter_mean'], state['parameter_covariance'])
        occupancy = result._tensor(state['occupancy'])
        if occupancy.ndim != 0 or not bool(torch.isfinite(occupancy)) or bool(occupancy < 0):
            raise ValueError('saved occupancy must be finite and nonnegative')
        result._occupancy = occupancy
        return result
