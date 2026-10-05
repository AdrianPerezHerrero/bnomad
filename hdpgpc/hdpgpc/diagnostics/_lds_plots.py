"""Read-only diagnostics for the fitted linear state-space matrices."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


_NAMES = ('A', 'C', 'Gamma', 'Sigma')


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
    return np.array(value, dtype=np.float64, copy=True)


def model_lds_arrays(model, *, output=0):
    """Copy the final stored LDS matrices for each represented cluster/output.

    The returned mapping is ``cluster_id -> {A, C, Gamma, Sigma}``. These are
    the matrices actually used by the fitted emission, rather than posterior
    uncertainty summaries or a fresh estimate from the observations.
    """
    backend = getattr(model, '_backend', model)
    if backend is None or not hasattr(backend, 'gpmodels'):
        raise ValueError('model must contain fitted cluster emissions')
    if not isinstance(output, int) or not 0 <= output < len(backend.gpmodels):
        raise ValueError('output must be a valid zero-based channel index')
    if hasattr(model, 'cluster_ids_'):
        cluster_ids = [int(cluster) for cluster in model.cluster_ids_]
    else:
        cluster_ids = list(range(len(backend.gpmodels[output])))
    if not cluster_ids:
        raise ValueError('model has no represented clusters')
    result = {}
    for cluster in cluster_ids:
        gp = backend.gpmodels[output][cluster]
        result[cluster] = {name: _array(getattr(gp, name)[-1]) for name in _NAMES}
    return result


def _matrix_spectrum(name, matrix):
    singular = np.linalg.svd(matrix, compute_uv=False)
    largest = float(singular[0])
    smallest = float(singular[-1])
    condition = largest / smallest if smallest > 0 else np.inf
    tolerance = np.finfo(float).eps * max(matrix.shape) * largest
    rank = int(np.count_nonzero(singular > tolerance))
    if name in ('Gamma', 'Sigma'):
        symmetric = (matrix + matrix.T) / 2
        eigenvalues = np.linalg.eigvalsh(symmetric)[::-1]
        relative_asymmetry = (np.linalg.norm(matrix - matrix.T, ord='fro') /
                              max(np.linalg.norm(matrix, ord='fro'), np.finfo(float).tiny))
        extra = (f'min λ={eigenvalues[-1]:.2g} · '
                 f'asym={relative_asymmetry:.1e}')
        ylabel = 'Eigenvalue of symmetric part'
    elif matrix.shape[0] == matrix.shape[1]:
        eigenvalues = np.linalg.eigvals(matrix)
        eigenvalues = eigenvalues[np.argsort(np.abs(eigenvalues))[::-1]]
        radius = float(np.max(np.abs(eigenvalues)))
        extra = f'ρ={radius:.3g}' if name == 'A' else f'max |λ|={radius:.3g}'
        ylabel = 'Eigenvalue magnitude'
    else:
        eigenvalues = singular
        extra = 'rectangular: singular values'
        ylabel = 'Singular value'
    return eigenvalues, condition, rank, extra, ylabel


def plot_lds_diagnostics(matrices, *, title='Fitted LDS diagnostics',
                         clusters_per_page=4, max_eigenvalues=12):
    """Yield paged figures showing stored matrices and their leading spectra.

    ``matrices`` is the mapping returned by :func:`model_lds_arrays` (or an
    equivalent mapping of cluster IDs to four two-dimensional arrays). Each
    cluster occupies two rows: matrix heatmaps and spectral plots. The matrix
    color range is shared within each parameter column and page. Covariance
    eigenvalues are computed from the symmetric part; asymmetry is reported.
    No figure is shown or saved here. Consume and close each page in turn for
    large records, so later pages need not remain in memory.
    """
    if not isinstance(clusters_per_page, int) or clusters_per_page < 1:
        raise ValueError('clusters_per_page must be a positive integer')
    if not isinstance(max_eigenvalues, int) or max_eigenvalues < 1:
        raise ValueError('max_eigenvalues must be a positive integer')
    if not matrices:
        raise ValueError('matrices must contain at least one cluster')
    copied = {}
    for cluster, parameters in matrices.items():
        copied[cluster] = {}
        for name in _NAMES:
            matrix = _array(parameters[name])
            if matrix.ndim != 2 or not all(matrix.shape) or not np.isfinite(matrix).all():
                raise ValueError(f'cluster {cluster} {name} must be a finite nonempty matrix')
            if name in ('A', 'Gamma', 'Sigma') and matrix.shape[0] != matrix.shape[1]:
                raise ValueError(f'cluster {cluster} {name} must be square')
            copied[cluster][name] = matrix

    cluster_ids = list(copied)
    for first in range(0, len(cluster_ids), clusters_per_page):
        page_ids = cluster_ids[first:first + clusters_per_page]
        fig, axes = plt.subplots(
            2 * len(page_ids), 4,
            figsize=(19, 4.5 * len(page_ids) + 1.0),
            constrained_layout=True, squeeze=False,
            gridspec_kw={'height_ratios': [2.1, 1.4] * len(page_ids)},
        )
        limits = {
            name: max(float(np.max(np.abs(copied[cluster][name])))
                      for cluster in page_ids) or 1.0
            for name in _NAMES
        }
        for row, cluster in enumerate(page_ids):
            for column, name in enumerate(_NAMES):
                matrix = copied[cluster][name]
                matrix_axis = axes[2 * row, column]
                spectrum_axis = axes[2 * row + 1, column]
                try:
                    eigenvalues, condition, rank, extra, ylabel = _matrix_spectrum(name, matrix)
                except np.linalg.LinAlgError:
                    eigenvalues = np.array([])
                    condition, rank, extra, ylabel = np.nan, 0, 'decomposition failed', 'Eigenvalue'
                limit = limits[name]
                image = matrix_axis.imshow(
                    matrix, cmap='coolwarm', vmin=-limit, vmax=limit,
                    interpolation='nearest', aspect='auto', rasterized=True,
                )
                matrix_axis.set_title(
                    f'{name} ({matrix.shape[0]}×{matrix.shape[1]})\n'
                    f'κ₂={condition:.2g} · rank={rank} · {extra}', fontsize=9,
                )
                matrix_axis.set_xlabel('Column', fontsize=8)
                matrix_axis.set_ylabel(
                    f'Cluster {cluster}\nRow' if column == 0 else 'Row', fontsize=9,
                )
                matrix_axis.tick_params(labelsize=7)
                fig.colorbar(image, ax=matrix_axis, fraction=0.045, pad=0.02)

                leading = eigenvalues[:max_eigenvalues]
                if len(leading):
                    values = (leading if name in ('Gamma', 'Sigma')
                              else np.abs(leading))
                    ranks = np.arange(1, len(values) + 1)
                    spectrum_axis.plot(ranks, values, marker='o', markersize=3,
                                       linewidth=1.1, color='tab:blue')
                    spectrum_axis.set_xticks(ranks)
                    if name == 'A':
                        spectrum_axis.axhline(1, color='tab:red', linestyle='--',
                                              linewidth=0.8, label='Unit radius')
                        spectrum_axis.legend(loc='best', fontsize=7, frameon=False)
                    if name in ('Gamma', 'Sigma'):
                        spectrum_axis.axhline(0, color='tab:red', linestyle=':', linewidth=0.8)
                    spectrum_axis.set_yscale('symlog', linthresh=1e-8)
                    leading_text = ', '.join(f'{value:.2g}' for value in leading[:3])
                    symbol = 's' if name == 'C' and matrix.shape[0] != matrix.shape[1] else 'λ'
                    spectrum_axis.set_title(f'Leading {symbol}: {leading_text}', fontsize=8)
                else:
                    spectrum_axis.text(0.5, 0.5, 'Spectrum unavailable',
                                       ha='center', va='center', transform=spectrum_axis.transAxes)
                spectrum_axis.set(xlabel='Rank', ylabel=ylabel)
                spectrum_axis.tick_params(labelsize=7)
                spectrum_axis.grid(alpha=0.2)
        page = first // clusters_per_page + 1
        pages = (len(cluster_ids) + clusters_per_page - 1) // clusters_per_page
        fig.suptitle(f'{title} · page {page}/{pages} · clusters {page_ids[0]}–{page_ids[-1]}',
                     fontsize=14)
        yield fig
