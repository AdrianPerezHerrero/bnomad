"""Linear slow-feature calibration for the existing fixed affine observation."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ._fixed_observation import AffineGaussianObservation


@dataclass(frozen=True)
class SFAObservationEstimate:
    """A temporal encoder and its independently fitted Gaussian decoder.

    ``encoder`` has shape (observation dimension, latent dimension); raw SFA
    features are ``(y - model.offset) @ encoder``. They have zero mean and unit
    population covariance on ``indices``. ``pair_indices`` contains actual row
    pairs, never consecutive rows of a shuffled minibatch. Derivative variances
    are ordered from slowest to fastest in the supplied time units.

    ``model`` is an ordinary AffineGaussianObservation. Its Gaussian likelihood
    encoding generally differs from this raw SFA transform when reconstruction
    noise is diagonal. No raw encoder output is injected as a noiseless state.
    """

    model: AffineGaussianObservation
    encoder: np.ndarray
    indices: np.ndarray
    pair_indices: np.ndarray
    derivative_variances: np.ndarray
    numerical_rank: int
    whitening_dimension: int

    def transform(self, observations):
        """Return raw SFA features for finite observation rows (one output)."""
        values = np.asarray(observations, dtype=np.float64)
        if values.ndim == 3 and values.shape[-1] == 1:
            values = values[..., 0]
        if (values.ndim != 2 or values.shape[1] != self.model.observation_dim
                or not np.isfinite(values).all()):
            raise ValueError('SFA transform requires finite rows matching the observation dimension')
        return (values - self.model.offset) @ self.encoder


def estimate_sfa_observation(observations, grid, *, components=10, max_samples=400,
                             sampling='uniform', random_state=42, lag=1,
                             times=None, segment_ids=None, whitening_components=None,
                             rank_tolerance=1e-6, noise_floor=1e-6):
    """Calibrate a fixed affine emission using linear Slow Feature Analysis.

    The temporal objective is E[||W.T (y[t+lag]-y[t]) / dt||^2], subject to
    W.T S W = I for centered calibration covariance S. SVD whitening followed by
    a symmetric eigenproblem selects the slowest directions. Numerically null
    covariance directions are discarded. Optional PCA whitening truncation is
    only a pilot space; final features are selected by temporal slowness.

    Rows must already be chronological within each contiguous segment. Supply
    ``segment_ids`` to prevent pairs spanning sequence boundaries. ``times``
    must increase within each segment; otherwise row indices define time.
    For irregular data, differences are divided by actual elapsed time.

    Prefix sampling uses at most max_samples initial rows. Uniform sampling
    selects genuine lagged pairs and uses their unique endpoints, with at most
    max_samples rows (up to floor(max_samples/2) pairs unless all rows fit).
    This preserves adjacency even for a small minibatch of a long sequence.
    The whitening dimension is also capped by pair count to avoid directions
    whose apparent slowness is due solely to insufficient derivative samples.
    The returned latent dimension can be smaller than requested if rank or
    temporal evidence is insufficient. No labels or cluster posteriors are used.

    A least-squares affine decoder is fitted to the SFA features. Its diagonal
    Sigma is sample reconstruction-residual variance with an absolute floor;
    its initial latent covariance is the unit covariance imposed by SFA.
    Conditioning, likelihoods, noise policies and serialization are all provided
    by the existing affine class, without a second temporal likelihood factor.
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
        raise ValueError('SFA requires finite ordered observation rows, one output and a matching grid')
    for name, value, minimum in (('components', components, 1), ('max_samples', max_samples, 2),
                                  ('lag', lag, 1)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if components > values.shape[1] or lag >= len(values):
        raise ValueError('SFA components cannot exceed observation dimension; lag must be smaller than row count')
    if whitening_components is not None and (
            isinstance(whitening_components, bool) or not isinstance(whitening_components, Integral)
            or whitening_components < components):
        raise ValueError('whitening_components must be None or an integer >= components')
    if sampling not in ('uniform', 'prefix'):
        raise ValueError('SFA sampling must be uniform or prefix')
    if not np.isfinite(rank_tolerance) or not 0 < rank_tolerance < 1:
        raise ValueError('SFA rank_tolerance must lie strictly between zero and one')
    if not np.isfinite(noise_floor) or noise_floor <= 0:
        raise ValueError('SFA noise_floor must be finite and positive')

    segments = np.zeros(len(values), dtype=int) if segment_ids is None else np.asarray(segment_ids)
    if segments.shape != (len(values),):
        raise ValueError('segment_ids must have one entry per observation')
    # Run numbers also prevent pairing across a repeated segment ID after a gap.
    runs = np.r_[0, np.cumsum(segments[1:] != segments[:-1])]
    time = np.arange(len(values), dtype=float) if times is None else np.asarray(times, dtype=float)
    if (time.shape != (len(values),) or not np.isfinite(time).all()
            or np.any((np.diff(time) <= 0) & (runs[1:] == runs[:-1]))):
        raise ValueError('SFA times must be finite and strictly increasing within each segment')
    starts = np.arange(len(values) - lag)
    starts = starts[runs[starts] == runs[starts + lag]]
    count = min(int(max_samples), len(values))
    if sampling == 'prefix':
        starts = starts[starts + lag < count]
        indices = np.arange(count)
    elif count == len(values):
        indices = np.arange(count)
    else:
        pair_count = min(count // 2, len(starts))
        starts = np.sort(np.random.default_rng(random_state).choice(starts, pair_count, replace=False))
        indices = np.unique(np.r_[starts, starts + lag])
    if not len(starts):
        raise ValueError('SFA calibration has no valid within-segment lagged pairs; increase samples or reduce lag')
    pairs = np.column_stack((starts, starts + lag))
    batch = values[indices]
    mean = batch.mean(axis=0)
    centered = batch - mean
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    rank = int(np.sum(singular > rank_tolerance * singular[0]))
    if not rank:
        raise ValueError('SFA calibration has zero numerical covariance rank')
    white_dim = min(rank, len(pairs), rank if whitening_components is None else int(whitening_components))
    dimension = min(int(components), white_dim)
    whitening = right[:white_dim].T * (np.sqrt(len(batch)) / singular[:white_dim])
    deltas = ((values[pairs[:, 1]] - values[pairs[:, 0]])
              / (time[pairs[:, 1]] - time[pairs[:, 0]])[:, None]) @ whitening
    derivative_covariance = deltas.T @ deltas / len(pairs)
    eigenvalues, directions = np.linalg.eigh(.5 * (derivative_covariance + derivative_covariance.T))
    encoder = whitening @ directions[:, :dimension]
    # Fix arbitrary signs for reproducible calibration artifacts.
    encoder *= np.sign(encoder[np.abs(encoder).argmax(axis=0), np.arange(dimension)])
    codes = centered @ encoder
    code_covariance = codes.T @ codes / len(batch)
    cross = centered.T @ codes / len(batch)
    decoder = np.linalg.solve(code_covariance, cross.T).T
    residual = centered - codes @ decoder.T
    sigma = np.diag(np.maximum(residual.var(axis=0, ddof=1), noise_floor))
    model = AffineGaussianObservation(C=decoder, Sigma=sigma, observation_grid=coordinates,
                                      offset=mean, latent_covariance=np.eye(dimension))
    return SFAObservationEstimate(model=model, encoder=encoder, indices=indices,
        pair_indices=pairs, derivative_variances=np.maximum(eigenvalues[:dimension], 0.),
        numerical_rank=rank, whitening_dimension=white_dim)
