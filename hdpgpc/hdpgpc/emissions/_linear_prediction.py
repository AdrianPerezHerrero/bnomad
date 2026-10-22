"""Reconstructive, linearly predictive calibration for fixed affine emissions."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import torch

from ._fixed_observation import AffineGaussianObservation
from ._temporal import calibration_pairs


@dataclass(frozen=True)
class LinearPredictionObservationEstimate:
    """Frozen affine decoder plus raw encoder and calibration diagnostics.

    Raw codes are (y-model.offset) @ encoder. predictor uses column vectors:
    z_next = predictor @ z. It is a calibration tool, not the LDS matrix A.
    losses has columns total, normalized reconstruction, normalized prediction
    residual and normalized predictor ridge. Row zero is a PCA control; later
    rows are optimizer steps. The returned model uses the best recorded basis.
    """

    model: AffineGaussianObservation
    encoder: np.ndarray
    predictor: np.ndarray
    indices: np.ndarray
    pair_indices: np.ndarray
    pair_durations: np.ndarray
    losses: np.ndarray
    best_iteration: int
    numerical_rank: int
    whitening_dimension: int
    prediction_weight: float
    prediction_ridge: float

    def transform(self, observations):
        """Return raw normalized features; Gaussian LDS states remain uncertain."""
        values = np.asarray(observations, dtype=np.float64)
        if values.ndim == 3 and values.shape[-1] == 1:
            values = values[..., 0]
        if (values.ndim != 2 or values.shape[1] != self.model.observation_dim
                or not np.isfinite(values).all()):
            raise ValueError('Linear prediction transform requires finite matching observation rows')
        return (values - self.model.offset) @ self.encoder


def estimate_linear_prediction_observation(observations, grid, *, components=10,
        max_samples=400, epochs=250, learning_rate=.02, prediction_weight=1.,
        prediction_ridge=1e-6, whitening_components=None, rank_tolerance=1e-6,
        sampling='uniform', random_state=42, lag=1, times=None, segment_ids=None,
        noise_floor=1e-6):
    """Fit affine features balancing reconstruction and linear predictability.

    On centered calibration rows X, enforce population covariance E[z z.T]=I
    exactly through SVD whitening and a QR-orthonormal learned basis Q. Fit C
    and the calibration predictor B by least squares for each Q. Optimize

      reconstruction_RSS / sum(X**2)
      + prediction_weight * (prediction_RSS / pair_count
                             + prediction_ridge * ||B||_F**2) / latent_dim.

    This is a normalized version of reconstruction plus latent prediction loss,
    with an explicit small ridge on B for stability. It is not an extra ELBO
    term. Only Q is optimized by Adam; C/B are profiled out analytically using
    small covariance matrices. The objective is nonconvex; no global optimum
    is claimed. A PCA basis is recorded as a control, and the best objective
    encountered is returned. Weight zero returns that PCA subspace directly.

    Uniform/prefix temporal sampling is shared with Linear SFA, never forming
    pairs across segment boundaries. A single B requires constant lag duration;
    irregular durations are rejected instead of being silently treated as equal.
    Effective dimensions are capped by covariance rank and pair count.

    Returned C/offset are fixed. Sigma is diagonal sample reconstruction
    variance with an absolute floor and P0=I. Installing model with the standard
    fixed observation helper preserves exact affine Gaussian LDS inference.
    Its likelihood encoder generally differs from these raw calibration codes.
    """
    values = np.asarray(observations, dtype=np.float64)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    coordinates = np.asarray(grid, dtype=np.float64)
    if coordinates.ndim == 1:
        coordinates = coordinates[:, None]
    if (values.ndim != 2 or len(values) < 2 or values.shape[1] < 2
            or coordinates.shape != (values.shape[1], 1)
            or not np.isfinite(values).all() or not np.isfinite(coordinates).all()):
        raise ValueError('Linear prediction calibration requires finite ordered rows, one output and a matching grid')
    for name, value, minimum in (('components', components, 1), ('max_samples', max_samples, 2),
                                  ('epochs', epochs, 1), ('lag', lag, 1)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if components > values.shape[1] or lag >= len(values):
        raise ValueError('Components cannot exceed observation dimension; lag must be smaller than row count')
    if whitening_components is not None and (
            isinstance(whitening_components, bool) or not isinstance(whitening_components, Integral)
            or whitening_components < components):
        raise ValueError('whitening_components must be None or an integer >= components')
    if sampling not in ('uniform', 'prefix'):
        raise ValueError('Calibration sampling must be uniform or prefix')
    if not np.isfinite(prediction_weight) or prediction_weight < 0:
        raise ValueError('prediction_weight must be finite and nonnegative')
    if any(not np.isfinite(x) or x <= 0 for x in (learning_rate, prediction_ridge, noise_floor)):
        raise ValueError('Learning rate, predictor ridge and noise floor must be finite and positive')
    if not np.isfinite(rank_tolerance) or not 0 < rank_tolerance < 1:
        raise ValueError('rank_tolerance must lie strictly between zero and one')
    indices, pairs, durations = calibration_pairs(len(values), max_samples=max_samples,
        sampling=sampling, random_state=random_state, lag=lag, times=times, segment_ids=segment_ids)
    if not np.allclose(durations, durations[0], rtol=1e-6, atol=1e-9):
        raise ValueError('A single calibration predictor requires constant lag duration; resample or split irregular data')
    batch = values[indices]
    mean = batch.mean(axis=0)
    centered = batch - mean
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    rank = int(np.sum(singular > rank_tolerance * singular[0]))
    if not rank:
        raise ValueError('Linear prediction calibration has zero numerical covariance rank')
    white_dim = min(rank, len(pairs), rank if whitening_components is None else int(whitening_components))
    dimension = min(int(components), white_dim)
    scales = singular[:white_dim] / np.sqrt(len(batch))
    whitening = right[:white_dim].T / scales
    decoder_basis = right[:white_dim].T * scales
    source = (values[pairs[:, 0]] - mean) @ whitening
    target = (values[pairs[:, 1]] - mean) @ whitening
    source_cov = torch.as_tensor(source.T @ source / len(pairs), dtype=torch.float64)
    target_cov = torch.as_tensor(target.T @ target / len(pairs), dtype=torch.float64)
    cross_cov = torch.as_tensor(target.T @ source / len(pairs), dtype=torch.float64)
    variance = torch.as_tensor(scales ** 2, dtype=torch.float64)
    total_variance = float(np.sum(singular ** 2) / len(batch))
    eye = torch.eye(dimension, dtype=torch.float64)

    def objective(basis):
        s00 = basis.T @ source_cov @ basis
        s11 = basis.T @ target_cov @ basis
        s10 = basis.T @ cross_cov @ basis
        predictor = torch.linalg.solve(s00 + prediction_ridge * eye, s10.T).T
        reconstruction = 1. - torch.sum(variance[:, None] * basis.square()) / total_variance
        prediction = (torch.trace(s11) - 2 * torch.sum(predictor * s10)
                      + torch.sum((predictor @ s00) * predictor)) / dimension
        ridge = prediction_ridge * predictor.square().sum() / dimension
        total = reconstruction + prediction_weight * (prediction + ridge)
        return torch.stack((total, reconstruction, prediction, ridge))

    # A local RNG does not alter the model's global random state.
    baseline = torch.eye(white_dim, dimension, dtype=torch.float64)
    with torch.no_grad():
        initial = objective(baseline)
    if not bool(torch.isfinite(initial).all()):
        raise ValueError('Linear prediction calibration produced a nonfinite initial objective')
    history = [initial.numpy().copy()]
    best_basis, best_loss, best_iteration = baseline.clone(), float(initial[0]), 0
    if prediction_weight > 0 and dimension < white_dim:
        rng = np.random.default_rng(random_state)
        raw = torch.nn.Parameter(baseline + torch.as_tensor(
            .01 * rng.standard_normal((white_dim, dimension)), dtype=torch.float64))
        optimizer = torch.optim.Adam([raw], lr=learning_rate)
        for iteration in range(1, int(epochs) + 1):
            optimizer.zero_grad()
            basis = torch.linalg.qr(raw, mode='reduced').Q
            losses = objective(basis)
            if not bool(torch.isfinite(losses).all()):
                raise ValueError('Nonfinite calibration objective; reduce learning_rate or whitening rank')
            losses[0].backward()
            if raw.grad is None or not bool(torch.isfinite(raw.grad).all()):
                raise ValueError('Nonfinite calibration gradient; reduce learning_rate or whitening rank')
            optimizer.step()
            with torch.no_grad():
                basis = torch.linalg.qr(raw, mode='reduced').Q
                losses = objective(basis)
                if not bool(torch.isfinite(losses).all()):
                    raise ValueError('Nonfinite calibration step; reduce learning_rate or whitening rank')
                history.append(losses.numpy().copy())
                if float(losses[0]) < best_loss:
                    best_basis, best_loss, best_iteration = basis.clone(), float(losses[0]), iteration
    basis = best_basis.numpy()
    encoder = whitening @ basis
    signs = np.sign(encoder[np.abs(encoder).argmax(axis=0), np.arange(dimension)])
    encoder *= signs
    decoder = (decoder_basis @ basis) * signs
    source_codes, target_codes = source @ basis * signs, target @ basis * signs
    s00 = source_codes.T @ source_codes / len(pairs)
    s10 = target_codes.T @ source_codes / len(pairs)
    predictor = np.linalg.solve(s00 + prediction_ridge * np.eye(dimension), s10.T).T
    codes = centered @ encoder
    residual = centered - codes @ decoder.T
    sigma = np.diag(np.maximum(residual.var(axis=0, ddof=1), noise_floor))
    model = AffineGaussianObservation(C=decoder, Sigma=sigma, observation_grid=coordinates,
                                      offset=mean, latent_covariance=np.eye(dimension))
    return LinearPredictionObservationEstimate(model=model, encoder=encoder, predictor=predictor,
        indices=indices, pair_indices=pairs, pair_durations=durations,
        losses=np.asarray(history), best_iteration=best_iteration, numerical_rank=rank,
        whitening_dimension=white_dim, prediction_weight=float(prediction_weight),
        prediction_ridge=float(prediction_ridge))
