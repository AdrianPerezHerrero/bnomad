"""Fixed observation models for Gaussian LDS trajectories.

An encoder is an optional initializer, not an observation likelihood. A future
nonlinear decoder must supply its own Gaussian state update and expected log
likelihood; projecting observations through an encoder alone is insufficient.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch


def _chol_cov(value):
    matrix = 0.5 * (value + value.mT)
    eye = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    scale = matrix.diagonal(dim1=-2, dim2=-1).abs().mean(-1).clamp_min(torch.finfo(matrix.dtype).eps)
    for relative in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4):
        factor, info = torch.linalg.cholesky_ex(matrix + relative * scale[..., None, None] * eye)
        if not bool(info.any()):
            return factor
        if info.ndim:
            # Successful rows keep the same jitter as their scalar counterpart.
            failed = torch.where(info != 0)[0]
            factor = factor.clone()
            for index in failed.tolist():
                factor[index] = _chol_cov(matrix[index])
            return factor
    raise torch.linalg.LinAlgError('Fixed observation covariance is not positive definite')


class FixedObservationModel(ABC):
    """Observation contract for one output on a fixed waveform grid.

    Methods use column means and full covariance matrices in float64. `condition`
    returns Gaussian posterior moments after a responsibility-weighted likelihood
    update. `expected_log_likelihood` must evaluate E_q(f)[log p(y|f)] in waveform
    space, including normalization and uncertainty; an encoder reconstruction
    error alone is not a valid implementation. Nonconjugate implementations must
    document their approximation and must not claim an exact LDS ELBO.
    """

    @property
    @abstractmethod
    def grid(self) -> np.ndarray: ...

    @property
    @abstractmethod
    def latent_dim(self) -> int: ...

    @property
    @abstractmethod
    def observation_dim(self) -> int: ...

    @abstractmethod
    def encode(self, observation):
        """Return an initial Gaussian latent approximation (mean, covariance)."""
        ...

    def encode_likelihood(self, observation, *, noise=None):
        """Return latent precision and information supplied by one observation.

        This exact information-form encoding is available for affine Gaussian
        decoders. A nonlinear decoder can instead implement its own `condition`
        approximation without pretending its encoded output is a Gaussian
        likelihood factor.
        """
        raise NotImplementedError('This observation model has no exact latent likelihood encoding')

    @abstractmethod
    def decode(self, latent):
        """Return the decoder's waveform-space conditional mean."""
        ...

    @abstractmethod
    def condition(self, prior_mean, prior_covariance, observation, *,
                  responsibility=1.0, noise=None): ...

    @abstractmethod
    def expected_log_likelihood(self, observation, latent_mean,
                                latent_covariance, *, noise=None): ...

    def expected_log_likelihood_batch(self, observations, latent_mean,
                                      latent_covariance, *, noise=None):
        values = torch.as_tensor(observations)
        return torch.stack([
            self.expected_log_likelihood(value, latent_mean, latent_covariance,
                                         noise=noise)
            for value in values])

    @abstractmethod
    def predictive_moments(self, latent_mean, latent_covariance, *, noise=None): ...

    @abstractmethod
    def state_dict(self) -> dict: ...

    @classmethod
    @abstractmethod
    def from_state_dict(cls, state): ...


@dataclass(frozen=True)
class AffineGaussianObservation(FixedObservationModel):
    """Exact fixed decoder `y | f ~ N(offset + C f, Sigma)`.

    A learned Sigma policy may pass its current covariance to the three methods;
    the decoder and offset remain fixed. `latent_covariance` is a separate LDS
    prior on f, not an encoder covariance.
    """

    C: np.ndarray
    Sigma: np.ndarray
    observation_grid: np.ndarray
    offset: np.ndarray
    latent_covariance: np.ndarray

    def __post_init__(self):
        C = np.asarray(self.C, dtype=np.float64)
        Sigma = np.asarray(self.Sigma, dtype=np.float64)
        grid = np.asarray(self.observation_grid, dtype=np.float64)
        offset = np.asarray(self.offset, dtype=np.float64).reshape(-1)
        latent_cov = np.asarray(self.latent_covariance, dtype=np.float64)
        if (C.ndim != 2 or grid.shape != (C.shape[0], 1)
                or offset.shape != (C.shape[0],)
                or Sigma.shape != (C.shape[0], C.shape[0])
                or latent_cov.shape != (C.shape[1], C.shape[1])
                or not all(np.isfinite(x).all() for x in (C, Sigma, grid, offset, latent_cov))
                or not np.allclose(Sigma, Sigma.T)
                or not np.allclose(latent_cov, latent_cov.T)
                or np.linalg.eigvalsh(Sigma).min() <= 0
                or np.linalg.eigvalsh(latent_cov).min() <= 0):
            raise ValueError('Affine observation requires finite dimensions and positive definite covariances')
        for name, value in (('C', C), ('Sigma', Sigma), ('observation_grid', grid),
                            ('offset', offset), ('latent_covariance', latent_cov)):
            copy = value.copy()
            copy.setflags(write=False)
            object.__setattr__(self, name, copy)

    @property
    def grid(self):
        return self.observation_grid

    @property
    def latent_dim(self):
        return self.C.shape[1]

    @property
    def observation_dim(self):
        return self.C.shape[0]

    def _tensors(self, reference, noise):
        ref = torch.as_tensor(reference)
        C = torch.as_tensor(self.C.copy(), dtype=ref.dtype, device=ref.device)
        Sigma = torch.as_tensor(self.Sigma.copy() if noise is None else noise,
                                dtype=ref.dtype, device=ref.device)
        offset = torch.as_tensor(self.offset.copy(), dtype=ref.dtype,
                                 device=ref.device).reshape(-1, 1)
        if Sigma.shape != (self.observation_dim, self.observation_dim):
            raise ValueError('Observation noise has the wrong shape')
        if not torch.isfinite(Sigma).all():
            raise ValueError('Observation noise must be finite')
        if torch.equal(Sigma, torch.diag(torch.diag(Sigma))) and not torch.all(torch.diag(Sigma) > 0):
            raise torch.linalg.LinAlgError('Observation noise diagonal must be positive')
        return C, Sigma, offset

    def decode(self, latent):
        value = torch.as_tensor(latent).reshape(-1, 1)
        if value.shape != (self.latent_dim, 1):
            raise ValueError('Latent mean has the wrong dimension')
        C, _, offset = self._tensors(value, None)
        return offset + C @ value

    def encode(self, observation):
        """Gaussian posterior from the calibrated zero-mean latent prior."""
        value = torch.as_tensor(observation, dtype=torch.float64)
        prior_mean = torch.zeros((self.latent_dim, 1), dtype=value.dtype, device=value.device)
        prior_cov = torch.as_tensor(self.latent_covariance.copy(), dtype=value.dtype,
                                    device=value.device)
        return self.condition(prior_mean, prior_cov, value)

    def encode_likelihood(self, observation, *, noise=None):
        """Encode one waveform as its exact latent Gaussian likelihood factor.

        For y|f ~ N(offset + C f, Sigma), the factor has precision
        H = C.T Sigma^-1 C and information h = C.T Sigma^-1 (y-offset).
        Information form also supports a rank-deficient C, where a finite
        standalone latent measurement covariance does not exist.
        """
        y = torch.as_tensor(observation).reshape(-1, 1)
        C, Sigma, offset = self._tensors(y, noise)
        if y.shape != (self.observation_dim, 1):
            raise ValueError('Observation has the wrong dimension')
        if torch.equal(Sigma, torch.diag(torch.diag(Sigma))):
            inverse_variance = 1.0 / torch.diag(Sigma)
            weighted_C = inverse_variance[:, None] * C
            weighted_y = inverse_variance[:, None] * (y - offset)
        else:
            noise_chol = _chol_cov(Sigma)
            weighted_C = torch.cholesky_solve(C, noise_chol)
            weighted_y = torch.cholesky_solve(y - offset, noise_chol)
        precision = C.T @ weighted_C
        information = C.T @ weighted_y
        return 0.5 * (precision + precision.T), information

    def encode_measurement(self, observation, *, noise=None):
        """Return the latent measurement (mean, covariance) when C has full rank.

        The LDS filter uses `encode_likelihood`, so its update also works when
        a decoder is rank deficient. For an orthonormal tied linear
        autoencoder with isotropic Sigma, this mean is its trained encoder.
        """
        precision, information = self.encode_likelihood(observation, noise=noise)
        chol, status = torch.linalg.cholesky_ex(precision)
        if bool(status.item()):
            raise ValueError('A standalone latent measurement requires full-column-rank C')
        covariance = torch.cholesky_inverse(chol)
        return covariance @ information, 0.5 * (covariance + covariance.T)

    def condition(self, prior_mean, prior_covariance, observation, *,
                  responsibility=1.0, noise=None):
        mean = torch.as_tensor(prior_mean).reshape(-1, 1)
        covariance = torch.as_tensor(prior_covariance, dtype=mean.dtype, device=mean.device)
        y = torch.as_tensor(observation, dtype=mean.dtype, device=mean.device).reshape(-1, 1)
        _, Sigma, _ = self._tensors(mean, noise)
        if y.shape != (self.observation_dim, 1) or mean.shape != (self.latent_dim, 1):
            raise ValueError('Observation or latent mean has the wrong dimension')
        weight = float(responsibility)
        if not np.isfinite(weight) or weight < 0:
            raise ValueError('responsibility must be finite and nonnegative')
        if weight == 0:
            return mean, covariance
        prior_chol = _chol_cov(covariance)
        prior_precision = torch.cholesky_inverse(prior_chol)
        encoded_precision, encoded_information = self.encode_likelihood(y, noise=Sigma)
        precision = prior_precision + weight * encoded_precision
        posterior_cov = torch.cholesky_inverse(_chol_cov(precision))
        posterior_mean = posterior_cov @ (prior_precision @ mean + weight * encoded_information)
        return posterior_mean, 0.5 * (posterior_cov + posterior_cov.T)

    def expected_log_likelihood(self, observation, latent_mean,
                                latent_covariance, *, noise=None):
        mean = torch.as_tensor(latent_mean).reshape(-1, 1)
        covariance = torch.as_tensor(latent_covariance, dtype=mean.dtype, device=mean.device)
        y = torch.as_tensor(observation, dtype=mean.dtype, device=mean.device).reshape(-1, 1)
        C, Sigma, offset = self._tensors(mean, noise)
        if y.shape != (self.observation_dim, 1):
            raise ValueError('Observation has the wrong dimension')
        residual = y - offset - C @ mean
        if torch.equal(Sigma, torch.diag(torch.diag(Sigma))):
            variance = torch.diag(Sigma)
            quadratic = torch.sum(residual.reshape(-1).square() / variance)
            trace = torch.sum(torch.sum((C @ covariance) * C, dim=1) / variance)
            logdet = torch.log(variance).sum()
        else:
            chol = _chol_cov(Sigma)
            quadratic = torch.sum(residual * torch.cholesky_solve(residual, chol))
            trace = torch.trace(torch.cholesky_solve(C @ covariance @ C.T, chol))
            logdet = 2 * torch.log(torch.diag(chol)).sum()
        return -0.5 * (quadratic + trace + logdet
                       + self.observation_dim * np.log(2 * np.pi))

    def condition_batch(self, prior_mean, prior_covariance, observations, *,
                        responsibilities, noise=None):
        """Condition independent states sharing one prior covariance and decoder.

        Repeated responsibilities reuse a factorization. The caller separates
        the initial-state prior from subsequent independent process priors.
        This method does not perform sequential parameter learning.
        """
        mean = torch.as_tensor(prior_mean).reshape(-1, 1)
        covariance = torch.as_tensor(prior_covariance, dtype=mean.dtype, device=mean.device)
        values = torch.as_tensor(observations, dtype=mean.dtype, device=mean.device)
        if values.ndim == 3 and values.shape[-1] == 1:
            values = values[..., 0]
        weights = torch.as_tensor(responsibilities, dtype=mean.dtype, device=mean.device).reshape(-1)
        if (values.ndim != 2 or values.shape != (len(weights), self.observation_dim)
                or mean.shape != (self.latent_dim, 1)
                or not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any())):
            raise ValueError('Independent batch requires matching observations and positive responsibilities')
        C, Sigma, offset = self._tensors(mean, noise)
        prior_precision = torch.cholesky_inverse(_chol_cov(covariance))
        if torch.equal(Sigma, torch.diag(torch.diag(Sigma))):
            inverse_variance = 1. / torch.diag(Sigma)
            encoded_precision = C.T @ (inverse_variance[:, None] * C)
            information = C.T @ (inverse_variance[:, None] * (values.T - offset))
        else:
            chol = _chol_cov(Sigma)
            encoded_precision = C.T @ torch.cholesky_solve(C, chol)
            information = C.T @ torch.cholesky_solve(values.T - offset, chol)
        encoded_precision = .5 * (encoded_precision + encoded_precision.T)
        if weights.requires_grad:
            unique, inverse = weights, torch.arange(len(weights), device=weights.device)
        else:
            unique, inverse = torch.unique(weights, return_inverse=True)
        precision = prior_precision[None, :, :] + unique[:, None, None] * encoded_precision[None, :, :]
        posterior = torch.cholesky_inverse(_chol_cov(precision))
        posterior = .5 * (posterior + posterior.mT)
        covariance_batch = posterior[inverse]
        natural = prior_precision @ mean + weights[None, :] * information
        means = covariance_batch @ natural.T.unsqueeze(-1)
        return means, covariance_batch

    def expected_log_likelihood_batch(self, observations, latent_mean,
                                      latent_covariance, *, noise=None):
        """Score observations with one shared or one per-observation posterior."""
        mean = torch.as_tensor(latent_mean)
        covariance = torch.as_tensor(latent_covariance, dtype=mean.dtype, device=mean.device)
        values = torch.as_tensor(observations, dtype=mean.dtype, device=mean.device)
        if values.ndim == 3 and values.shape[-1] == 1:
            values = values[..., 0]
        if values.ndim != 2 or values.shape[1] != self.observation_dim:
            raise ValueError('Batch observations have the wrong dimension')
        if mean.ndim < 3:
            mean = mean.reshape(1, -1, 1)
        if mean.shape[-2:] != (self.latent_dim, 1) or len(mean) not in (1, len(values)):
            raise ValueError('Latent means must be one state or one state per observation')
        if covariance.ndim == 2:
            covariance = covariance.unsqueeze(0)
        if covariance.shape[-2:] != (self.latent_dim, self.latent_dim) or len(covariance) not in (1, len(values)):
            raise ValueError('Latent covariances must be one state or one state per observation')
        C, Sigma, offset = self._tensors(mean, noise)
        residual = values.T - offset - (C @ mean).squeeze(-1).T
        if torch.equal(Sigma, torch.diag(torch.diag(Sigma))):
            variance = torch.diag(Sigma)
            quadratic = torch.sum(residual.square() / variance[:, None], dim=0)
            precision = C.T @ (C / variance[:, None])
            trace = torch.einsum('ij,nji->n', precision, covariance)
            logdet = torch.log(variance).sum()
        else:
            chol = _chol_cov(Sigma)
            quadratic = torch.sum(residual * torch.cholesky_solve(residual, chol), dim=0)
            precision = C.T @ torch.cholesky_solve(C, chol)
            trace = torch.einsum('ij,nji->n', precision, covariance)
            logdet = 2 * torch.log(torch.diag(chol)).sum()
        return -0.5 * (quadratic + trace + logdet
                       + self.observation_dim * np.log(2 * np.pi))

    def predictive_moments(self, latent_mean, latent_covariance, *, noise=None):
        mean = torch.as_tensor(latent_mean).reshape(-1, 1)
        covariance = torch.as_tensor(latent_covariance, dtype=mean.dtype, device=mean.device)
        C, Sigma, offset = self._tensors(mean, noise)
        return offset + C @ mean, C @ covariance @ C.T + Sigma

    def state_dict(self):
        return {'type': 'affine_gaussian', 'C': self.C.copy(),
                'Sigma': self.Sigma.copy(), 'grid': self.grid.copy(),
                'offset': self.offset.copy(),
                'latent_covariance': self.latent_covariance.copy()}

    def __deepcopy__(self, memo):
        # Immutable calibration belongs to the model, including proposal copies.
        return self

    @classmethod
    def from_state_dict(cls, state):
        if not isinstance(state, dict) or set(state) != {
                'type', 'C', 'Sigma', 'grid', 'offset', 'latent_covariance'} or state['type'] != 'affine_gaussian':
            raise ValueError('Unknown or malformed affine observation state')
        return cls(state['C'], state['Sigma'], state['grid'],
                   state['offset'], state['latent_covariance'])


class _AffineConditioner:
    """Observation-only factors held during one fixed-parameter filtering pass.

    The changing prior covariance is still factored for every state. Full noise
    uses the same jitter and solves as `AffineGaussianObservation.condition`.
    Nothing is attached to the immutable/shared calibration model.
    """

    def __init__(self, model, reference, noise):
        self.model, self.noise = model, noise
        self.C, sigma, self.offset = model._tensors(reference, noise)
        self.dtype, self.device = self.C.dtype, self.C.device
        self.inverse_variance = None
        self.noise_chol = None
        if torch.equal(sigma, torch.diag(torch.diag(sigma))):
            self.inverse_variance = 1.0 / torch.diag(sigma)
            weighted_C = self.inverse_variance[:, None] * self.C
        else:
            self.noise_chol = _chol_cov(sigma)
            weighted_C = torch.cholesky_solve(self.C, self.noise_chol)
        precision = self.C.T @ weighted_C
        self.precision = 0.5 * (precision + precision.T)

    def matches(self, model, reference, noise):
        # The enclosing pass holds Sigma fixed, and ends before its M-step.
        return (model is self.model and noise is self.noise
                and reference.dtype == self.dtype and reference.device == self.device)

    def condition(self, prior_mean, prior_covariance, observation, *, responsibility):
        mean = torch.as_tensor(prior_mean).reshape(-1, 1)
        covariance = torch.as_tensor(prior_covariance, dtype=mean.dtype, device=mean.device)
        y = torch.as_tensor(observation, dtype=mean.dtype, device=mean.device).reshape(-1, 1)
        if y.shape != (self.model.observation_dim, 1) or mean.shape != (self.model.latent_dim, 1):
            raise ValueError('Observation or latent mean has the wrong dimension')
        weight = float(responsibility)
        if not np.isfinite(weight) or weight < 0:
            raise ValueError('responsibility must be finite and nonnegative')
        if weight == 0:
            return mean, covariance
        prior_precision = torch.cholesky_inverse(_chol_cov(covariance))
        if self.inverse_variance is not None:
            weighted_y = self.inverse_variance[:, None] * (y - self.offset)
        else:
            weighted_y = torch.cholesky_solve(y - self.offset, self.noise_chol)
        information = self.C.T @ weighted_y
        precision = prior_precision + weight * self.precision
        posterior_cov = torch.cholesky_inverse(_chol_cov(precision))
        posterior_mean = posterior_cov @ (prior_precision @ mean + weight * information)
        return posterior_mean, 0.5 * (posterior_cov + posterior_cov.T)


@contextmanager
def _affine_conditioning_pass(emission):
    """Temporarily cache observation factors where an E-step holds Sigma fixed."""
    model = getattr(emission, 'fixed_observation_model', None)
    if (emission.lds_update != 'conjugate_batch' or emission.fit_gp
            or type(model) is not AffineGaussianObservation
            or emission.Sigma[-1].requires_grad):
        yield
        return
    backend = emission.gp
    missing = object()
    previous = getattr(backend, '_affine_conditioner', missing)
    conditioner = _AffineConditioner(model, emission.f_star_sm[-1], emission.Sigma[-1])
    backend._affine_conditioner = conditioner
    try:
        yield
    finally:
        if previous is missing:
            del backend._affine_conditioner
        else:
            backend._affine_conditioner = previous
