"""Real windowed Fourier decoders for the fixed affine observation path."""

from numbers import Integral

import numpy as np

from ._fixed_observation import AffineGaussianObservation


def _grid_and_bins(grid, n_harmonics, frequency_bins, window_size):
    if np.iscomplexobj(grid):
        raise ValueError('Fourier observation grid must be real')
    grid = np.asarray(grid, dtype=np.float64)
    if grid.ndim == 1:
        grid = grid[:, None]
    if grid.ndim != 2 or grid.shape[1] != 1 or len(grid) < 2 or not np.isfinite(grid).all():
        raise ValueError('Fourier observation requires a finite grid with at least two points')
    spacing = np.diff(grid[:, 0])
    if np.any(spacing <= 0) or not np.allclose(spacing, spacing[0], rtol=1e-7, atol=0):
        raise ValueError('Discrete Fourier observation requires a uniformly spaced increasing grid')
    if (isinstance(window_size, (bool, np.bool_)) or not isinstance(window_size, Integral)
            or window_size < 2):
        raise ValueError('window_size must be an integer of at least two points')
    highest = min(len(grid), int(window_size)) // 2
    if n_harmonics is not None and frequency_bins is not None:
        raise ValueError('Select n_harmonics or frequency_bins, not both')
    if n_harmonics is not None:
        if (isinstance(n_harmonics, (bool, np.bool_)) or not isinstance(n_harmonics, Integral)
                or not 0 <= n_harmonics <= highest):
            raise ValueError(f'n_harmonics must be an integer between 0 and {highest}')
        bins = list(range(int(n_harmonics) + 1))
    elif frequency_bins is not None:
        bins = list(frequency_bins)
        if (not bins or any(isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral)
                            or not 0 <= k <= highest for k in bins)
                or len(set(bins)) != len(bins)):
            raise ValueError(f'frequency_bins must be distinct integers between 0 and {highest}')
        bins = sorted(int(k) for k in bins)
    else:
        bins = list(range(highest + 1))
    return grid, bins


def real_fourier_basis(grid, *, window_size=16, n_harmonics=None, frequency_bins=None):
    """Return a real windowed synthesis matrix C, with shape (points, states).

    Fixed non-overlapping rectangular windows partition the grid. Columns
    are ordered by window, then bin: DC, cos(1), sin(1), ..., with one cosine
    at Nyquist for even window lengths. The final window can be shorter,
    including one point; no padding is performed. Omit both bin selections
    to retain all N coordinates and obtain a square orthonormal decoder.
    n_harmonics retains bins 0..h in each window; explicit frequency_bins
    selects local bins. Bins unavailable in a short final window are omitted.
    Frequency k in a window of length L is k/(L*grid_spacing).
    To reproduce the former global basis explicitly, set window_size=len(grid).
    """
    grid, bins = _grid_and_bins(grid, n_harmonics, frequency_bins, window_size)
    n = len(grid)
    blocks = []
    for start in range(0, n, int(window_size)):
        length = min(int(window_size), n - start)
        phase = 2 * np.pi * np.arange(length, dtype=np.float64) / length
        columns = []
        for k in bins:
            if k > length // 2:
                continue
            if k == 0:
                columns.append(np.ones(length) / np.sqrt(length))
            elif length % 2 == 0 and k == length // 2:
                columns.append((-1.0) ** np.arange(length) / np.sqrt(length))
            else:
                columns.extend((np.sqrt(2 / length) * np.cos(k * phase),
                                np.sqrt(2 / length) * np.sin(k * phase)))
        local = np.column_stack(columns) if columns else np.empty((length, 0))
        blocks.append((start, local))
    C = np.zeros((n, sum(local.shape[1] for _, local in blocks)))
    column = 0
    for start, local in blocks:
        C[start:start + len(local), column:column + local.shape[1]] = local
        column += local.shape[1]
    return C


def _covariance(value, dimension, name):
    if np.iscomplexobj(value):
        raise ValueError(f'{name} must be real')
    value = np.asarray(value, dtype=np.float64)
    if value.ndim == 0:
        value = float(value) * np.eye(dimension)
    elif value.ndim == 1 and value.shape == (dimension,):
        value = np.diag(value)
    if value.shape != (dimension, dimension):
        raise ValueError(f'{name} must be a scalar variance, a diagonal vector, or a {dimension}x{dimension} covariance')
    return value


def make_fourier_observation(grid, *, window_size=16, n_harmonics=None, frequency_bins=None,
                             Sigma=1.0, latent_covariance=1.0, offset=None):
    """Construct the existing affine observation with a windowed Fourier C.

    Input observations and likelihoods stay in waveform space. Sigma is
    waveform-space noise; latent_covariance initializes the Gaussian Fourier
    state. Scalar and diagonal arguments specify variances, not standard
    deviations. A truncated basis models omitted frequencies as residual
    observation noise. No calibration, FFT preprocessing, or new LDS engine
    is introduced. Install with config.with_fixed_observation_model.

    encode_measurement equals C.T @ (y-offset) for isotropic Sigma; encode
    instead returns the posterior after combining that likelihood with the
    initial latent prior, consistently with all fixed affine decoders.
    """
    grid, bins = _grid_and_bins(grid, n_harmonics, frequency_bins, window_size)
    C = real_fourier_basis(grid, window_size=window_size, frequency_bins=bins)
    if offset is not None and np.iscomplexobj(offset):
        raise ValueError('Fourier observation offset must be real')
    return AffineGaussianObservation(
        C=C, Sigma=_covariance(Sigma, len(grid), 'Sigma'),
        observation_grid=grid,
        offset=np.zeros(len(grid)) if offset is None else np.asarray(offset, dtype=np.float64),
        latent_covariance=_covariance(latent_covariance, C.shape[1], 'latent_covariance'))
