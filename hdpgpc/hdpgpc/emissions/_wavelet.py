"""Full orthonormal Haar decoder for the existing fixed affine observation path."""

import numpy as np

from ._fixed_observation import AffineGaussianObservation


def _wavelet_grid(grid):
    if np.iscomplexobj(grid):
        raise ValueError('Wavelet observation grid must be real')
    grid = np.asarray(grid, dtype=np.float64)
    if grid.ndim == 1:
        grid = grid[:, None]
    if (grid.ndim != 2 or grid.shape[1] != 1 or len(grid) < 2
            or not np.isfinite(grid).all()):
        raise ValueError('Wavelet observation requires a finite grid with at least two points')
    n = len(grid)
    if n & (n - 1):
        raise ValueError('Full orthonormal Haar observation requires a power-of-two point count; '
                         'resample explicitly rather than padding the decoder')
    spacing = np.diff(grid[:, 0])
    if np.any(spacing <= 0) or not np.allclose(spacing, spacing[0], rtol=1e-7, atol=0):
        raise ValueError('Haar observation requires a uniformly spaced increasing grid')
    return grid


def real_wavelet_basis(grid, *, wavelet='haar'):
    """Return the full N-by-N real orthonormal Haar synthesis matrix C.

    Columns are the constant scaling function followed by detail functions,
    ordered from coarse to fine scale and left to right within each scale.
    A detail on a block of length L is +1/sqrt(L) on its first half and
    -1/sqrt(L) on its second half, zero elsewhere. N must be a power of two.
    No padding, coefficient truncation, thresholding or calibration is used.
    ``wavelet`` currently supports only 'haar'. Encoding is C.T @ (y-offset).
    """
    if wavelet != 'haar':
        raise ValueError("Only the full orthonormal 'haar' wavelet is supported")
    grid = _wavelet_grid(grid)
    n = len(grid)
    C = np.zeros((n, n), dtype=np.float64)
    C[:, 0] = 1 / np.sqrt(n)
    column = 1
    length = n
    while length >= 2:
        for start in range(0, n, length):
            middle = start + length // 2
            C[start:middle, column] = 1 / np.sqrt(length)
            C[middle:start + length, column] = -1 / np.sqrt(length)
            column += 1
        length //= 2
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
        raise ValueError(f'{name} must be a scalar variance, a diagonal vector, '
                         f'or a {dimension}x{dimension} covariance')
    return value


def make_wavelet_observation(grid, *, wavelet='haar', Sigma=1.0,
                             latent_covariance=1.0, offset=None):
    """Construct an AffineGaussianObservation with a fixed full Haar C.

    Input data and scores remain in waveform space. Sigma is waveform-space
    noise; latent_covariance is the initial wavelet-state prior. Scalars and
    diagonal vectors specify variances, not standard deviations. Covariances
    must be positive definite. The unregularized encode_measurement computes
    C.T @ (y-offset); encode combines this likelihood with the latent prior.
    Install with config.with_fixed_observation_model, exactly as for Fourier
    or PCA. Existing affine serialization stores the basis itself.
    """
    grid = _wavelet_grid(grid)
    C = real_wavelet_basis(grid, wavelet=wavelet)
    if offset is not None and np.iscomplexobj(offset):
        raise ValueError('Wavelet observation offset must be real')
    return AffineGaussianObservation(
        C=C, Sigma=_covariance(Sigma, len(grid), 'Sigma'),
        observation_grid=grid,
        offset=np.zeros(len(grid)) if offset is None else np.asarray(offset, dtype=np.float64),
        latent_covariance=_covariance(latent_covariance, len(grid), 'latent_covariance'))
