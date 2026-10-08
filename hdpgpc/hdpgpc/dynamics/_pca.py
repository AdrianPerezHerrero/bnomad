"""Independent minibatch PCA calibration for a fixed linear observation map."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PCAObservationEstimate:
    """Centered waveform decoder y = mean + components @ f + epsilon.

    ``components`` has shape (waveform points, latent components). The latent
    score prior variances and residual observation covariance are estimated on
    the same calibration minibatch, then held fixed as initialization/noise.
    """

    mean: np.ndarray
    components: np.ndarray
    latent_variances: np.ndarray
    residual_covariance: np.ndarray
    grid: np.ndarray
    beat_indices: tuple[int, ...]
    explained_variance_ratio: float


def estimate_pca_observation(observations, grid, *, components, max_beats=256,
                             random_state=0, sampling='uniform', noise_floor=1e-6):
    """Fit a centered PCA decoder from a representative one-output minibatch.

    ``sampling='prefix'`` uses only the first minibatch. For strictly causal
    evaluation, fit that prefix first and evaluate later beats, or supply an
    independent calibration record. Uniform sampling uses the whole record.
    The residual covariance is diagonal, with a scalar residual variance floor.
    No clustering labels or LDS posteriors are used in this independent pilot.
    """
    values = np.asarray(observations, dtype=np.float64)
    if values.ndim == 3 and values.shape[2] == 1:
        values = values[:, :, 0]
    coordinates = np.asarray(grid, dtype=np.float64)
    if coordinates.ndim == 1:
        coordinates = coordinates[:, None]
    if (values.ndim != 2 or coordinates.shape != (values.shape[1], 1)
            or not np.isfinite(values).all() or not np.isfinite(coordinates).all()):
        raise ValueError('PCA requires finite (beats, points, one output) data and a matching grid')
    if (isinstance(components, bool) or not isinstance(components, int)
            or components < 1 or components >= values.shape[1]):
        raise ValueError('components must be between 1 and points - 1')
    if isinstance(max_beats, bool) or not isinstance(max_beats, int) or max_beats < components + 1:
        raise ValueError('max_beats must contain at least components + 1 beats')
    if sampling not in ('uniform', 'prefix'):
        raise ValueError('sampling must be uniform or prefix')
    if not np.isfinite(noise_floor) or noise_floor <= 0:
        raise ValueError('noise_floor must be finite and positive')
    count = min(max_beats, len(values))
    if count <= components:
        raise ValueError('PCA calibration needs more beats than components')
    if sampling == 'prefix':
        indices = np.arange(count)
    else:
        indices = np.sort(np.random.default_rng(random_state).choice(
            len(values), size=count, replace=False))
    minibatch = values[indices]
    mean = minibatch.mean(axis=0)
    centered = minibatch - mean
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    decoder = right[:components].T.copy()
    # Fix arbitrary SVD signs so recorded configurations are reproducible.
    signs = np.sign(decoder[np.abs(decoder).argmax(axis=0), np.arange(components)])
    decoder *= signs
    scores = centered @ decoder
    variances = np.maximum(singular[:components] ** 2 / (count - 1), noise_floor)
    residual = centered - scores @ decoder.T
    residual_variance = np.maximum(residual.var(axis=0, ddof=1), noise_floor)
    total = float(np.sum(singular ** 2))
    ratio = float(np.sum(singular[:components] ** 2) / total) if total > 0 else 0.0
    return PCAObservationEstimate(
        mean=mean.copy(), components=decoder,
        latent_variances=variances,
        residual_covariance=np.diag(residual_variance),
        grid=coordinates.copy(), beat_indices=tuple(int(i) for i in indices),
        explained_variance_ratio=ratio,
    )
