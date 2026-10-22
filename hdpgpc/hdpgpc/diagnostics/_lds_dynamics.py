"""Read-only one-step LDS dynamics projected onto an observation grid."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.array(value, dtype=np.float64, copy=True)


def _symmetric(value):
    return 0.5 * (value + value.T)


def _grid(value):
    coordinates = _array(value)
    if coordinates.ndim == 1:
        coordinates = coordinates[:, None]
    if (coordinates.ndim != 2 or coordinates.shape[1] != 1
            or not len(coordinates) or not np.isfinite(coordinates).all()):
        raise ValueError('observation grid must have shape (points, 1)')
    return coordinates


def _gp_projection(emission, coordinates, C, Sigma):
    """Use the same kernel interpolation and residual rule as GP pred_dist."""
    basis = _grid(emission.x_basis)
    if np.array_equal(coordinates, basis):
        return C.copy(), np.zeros(len(basis)), Sigma.copy()

    kernel = emission.gp.kernel
    K_bb = np.asarray(kernel(basis, basis), dtype=np.float64)
    K_bo = np.asarray(kernel(basis, coordinates), dtype=np.float64)
    jitter = 1e-4 * max(float(np.mean(np.abs(np.diag(Sigma)))),
                         np.finfo(float).eps)
    interpolation = np.linalg.solve(K_bb + jitter * np.eye(len(basis)), K_bo).T

    # pred_dist supplies the GP mean correction and its interpolation residual.
    # Calling it with zero native mean isolates the affine offset and the
    # observation/interpolation covariance without modifying the fitted state.
    tensor_grid = torch.as_tensor(coordinates, dtype=emission.x_basis.dtype,
                                  device=emission.x_basis.device)
    native_zero = torch.zeros((len(basis), 1), dtype=emission.x_basis.dtype,
                              device=emission.x_basis.device)
    offset, residual = emission.gp.pred_dist(
        tensor_grid, emission.x_basis, native_zero, emission.Sigma[-1])
    return interpolation @ C, _array(offset).reshape(-1), _array(residual)


def model_lds_dynamics(model, *, grid=None, output=0):
    """Return one-step waveform-space LDS quantities for each active cluster.

    The native smoothed state is ``N(m, P)`` and its fitted transition is
    ``f_next = A f + w``, ``w ~ N(0, Gamma)``. On the chosen observation grid,
    ``y = b + H f + epsilon``. A fixed affine decoder uses ``H=C``; ordinary
    and inducing-point GP emissions use their native C followed by GP grid
    interpolation. The residual covariance follows the existing GP prediction
    rule, including interpolation uncertainty where that rule includes it.

    ``projected_transition = H A pinv(H)`` is an extension to observation
    coordinates for visualization. For full-column-rank H, it represents
    dynamics on the image of H. If H is rank deficient, it is only a
    least-squares extension. It must not be applied to a noisy waveform as
    a state update.
    Supply the waveform grid when the LDS uses inducing points. Without it,
    the diagnostic uses the native LDS basis.
    """
    backend = getattr(model, '_backend', model)
    if backend is None or not hasattr(backend, 'gpmodels'):
        raise ValueError('model must contain fitted cluster emissions')
    if not isinstance(output, int) or not 0 <= output < len(backend.gpmodels):
        raise ValueError('output must be a valid zero-based channel index')
    cluster_ids = ([int(cluster) for cluster in model.cluster_ids_]
                   if hasattr(model, 'cluster_ids_') else
                   list(range(len(backend.gpmodels[output]))))
    if not cluster_ids:
        raise ValueError('model has no represented clusters')

    result = {}
    for cluster in cluster_ids:
        emission = backend.gpmodels[output][cluster]
        direct_grid = getattr(emission, 'direct_observation_grid', None)
        coordinates = _grid(grid if grid is not None else
                            (direct_grid if direct_grid is not None else emission.x_basis))
        A, C = _array(emission.A[-1]), _array(emission.C[-1])
        Gamma, Sigma = _array(emission.Gamma[-1]), _array(emission.Sigma[-1])
        mean = _array(emission.f_star_sm[-1]).reshape(-1)
        covariance = _array(emission.cov_f_sm[-1])
        dimension = len(mean)
        if (A.shape != (dimension, dimension) or Gamma.shape != A.shape
                or covariance.shape != A.shape or C.shape[1] != dimension
                or Sigma.shape != (C.shape[0], C.shape[0])):
            raise ValueError(f'cluster {cluster} has inconsistent LDS dimensions')

        if direct_grid is not None:
            if not np.array_equal(coordinates, _grid(direct_grid)):
                raise ValueError('fixed observation requires its calibrated waveform grid')
            offset = _array(emission.direct_observation_offset).reshape(-1)
            H, residual = C.copy(), Sigma.copy()
        else:
            H, offset, residual = _gp_projection(emission, coordinates, C, Sigma)

        if not all(np.isfinite(x).all() for x in
                   (A, C, Gamma, Sigma, mean, covariance, H, offset, residual)):
            raise ValueError(f'cluster {cluster} contains non-finite LDS quantities')
        if H.shape != (len(coordinates), dimension) or offset.shape != (len(coordinates),):
            raise ValueError(f'cluster {cluster} observation projection has wrong dimensions')

        process = _symmetric(H @ Gamma @ H.T)
        propagated = _symmetric(H @ A @ covariance @ A.T @ H.T)
        current_state = _symmetric(H @ covariance @ H.T)
        residual = _symmetric(residual)
        forecast = _symmetric(propagated + process + residual)
        current = offset + H @ mean
        next_mean = offset + H @ A @ mean
        result[cluster] = {
            'grid': coordinates[:, 0].copy(),
            'observation_map': H, 'offset': offset,
            'current_mean': current, 'next_mean': next_mean,
            'expected_change': next_mean - current,
            'current_covariance': _symmetric(current_state + residual),
            'forecast_covariance': forecast,
            'propagated_state_covariance': propagated,
            'process_covariance': process,
            'observation_covariance': residual,
            'projected_transition': H @ A @ np.linalg.pinv(H),
            'latent_spectral_radius': float(np.max(np.abs(np.linalg.eigvals(A)))),
            'process_min_eigenvalue': float(np.linalg.eigvalsh(process)[0]),
            'forecast_min_eigenvalue': float(np.linalg.eigvalsh(forecast)[0]),
            'observable_rank': int(np.linalg.matrix_rank(H)),
            'latent_dimension': dimension,
        }
    return result


def _std(covariance):
    return np.sqrt(np.maximum(np.diag(covariance), 0.0))


def plot_lds_dynamics(dynamics, *, title='Projected LDS dynamics'):
    """Return one figure stacking a full six-panel row pair per cluster.

    Each cluster keeps the original 17 by 8.5 inch panel area; the figure
    grows vertically with the number of clusters. The caller owns the figure
    and may save or show it once.
    """
    if not dynamics:
        raise ValueError('dynamics must contain at least one cluster')
    fig = plt.figure(figsize=(17, 8.5 * len(dynamics) + 0.5),
                     constrained_layout=True)
    sections = fig.subfigures(len(dynamics), 1, squeeze=False)
    fig.suptitle(title, fontsize=15)
    for section, (cluster, values) in zip(sections[:, 0], dynamics.items()):
        x = values['grid']
        current, next_mean = values['current_mean'], values['next_mean']
        current_sd = _std(values['current_covariance'])
        forecast_sd = _std(values['forecast_covariance'])
        process_sd = _std(values['process_covariance'])
        observation_sd = _std(values['observation_covariance'])
        propagated_sd = _std(values['propagated_state_covariance'])
        axes = section.subplots(2, 3)

        axis = axes[0, 0]
        axis.plot(x, current, color='tab:blue', label='Current decoded state')
        axis.fill_between(x, current - 2 * current_sd, current + 2 * current_sd,
                          color='tab:blue', alpha=0.12)
        axis.plot(x, next_mean, color='tab:orange', label='One-step forecast')
        axis.fill_between(x, next_mean - 2 * forecast_sd, next_mean + 2 * forecast_sd,
                          color='tab:orange', alpha=0.17)
        axis.set_title('Waveform means and ±2 SD')
        axis.legend(fontsize=8)

        axis = axes[0, 1]
        change = values['expected_change']
        axis.axhline(0, color='0.4', linewidth=0.8)
        axis.plot(x, change, color='tab:purple', label='Expected dynamic change')
        axis.fill_between(x, change - 2 * process_sd, change + 2 * process_sd,
                          color='tab:purple', alpha=0.18, label='±2 process SD')
        axis.set_title('One-step change from current latent mean')
        axis.legend(fontsize=8)

        axis = axes[0, 2]
        axis.plot(x, propagated_sd, label='Propagated state', color='tab:blue')
        axis.plot(x, process_sd, label='Process', color='tab:purple')
        axis.plot(x, observation_sd, label='Observation / interpolation',
                  color='tab:green')
        axis.plot(x, forecast_sd, label='Total forecast', color='tab:orange', linewidth=1.8)
        axis.set_title('One-step SD components')
        axis.set_ylabel('Standard deviation')
        axis.legend(fontsize=8)

        transition = values['projected_transition']
        limit = float(np.max(np.abs(transition))) or 1.0
        image = axes[1, 0].imshow(transition, cmap='coolwarm', vmin=-limit,
                                  vmax=limit, aspect='auto', rasterized=True)
        axes[1, 0].set_title('Observable transition H A H⁺')
        section.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04)

        process = values['process_covariance']
        limit = float(np.max(np.abs(process))) or 1.0
        image = axes[1, 1].imshow(process, cmap='coolwarm', vmin=-limit,
                                  vmax=limit, aspect='auto', rasterized=True)
        axes[1, 1].set_title('Projected process covariance H Γ Hᵀ')
        section.colorbar(image, ax=axes[1, 1], fraction=0.046, pad=0.04)

        eigenvalues, eigenvectors = np.linalg.eigh(process)
        axis = axes[1, 2]
        for position in np.argsort(eigenvalues)[::-1][:3]:
            eigenvalue = max(float(eigenvalues[position]), 0.0)
            if eigenvalue > 0:
                mode = 2 * np.sqrt(eigenvalue) * eigenvectors[:, position]
                axis.plot(x, mode, label=f'λ={eigenvalue:.2g}')
        axis.axhline(0, color='0.4', linewidth=0.8)
        axis.set_title('Leading ±2 SD process-change modes')
        if axis.lines and len(axis.lines) > 1:
            axis.legend(fontsize=8)

        for axis in axes.flat:
            if axis.images:
                axis.set(xlabel='Observation index', ylabel='Observation index')
            else:
                axis.set_xlabel('Observation grid')
                axis.grid(alpha=0.2)
        section.suptitle(
            f'Cluster {cluster} · latent dim {values["latent_dimension"]}'
            f' · observable rank {values["observable_rank"]}'
            f' · ρ(A)={values["latent_spectral_radius"]:.3g}'
            f' · min λ(forecast cov)={values["forecast_min_eigenvalue"]:.2g}',
            fontsize=12)
    return fig
