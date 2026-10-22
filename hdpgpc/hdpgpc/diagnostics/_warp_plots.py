"""Matplotlib diagnostics for fitted input warps.

The plot accepts plain arrays so synthetic experiments and record analyses use
the same visual layout. ``model_warp_arrays`` adapts the native stored histories
without running another inference pass.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
    return np.asarray(value, dtype=np.float64)


def model_warp_arrays(model, grid, *, output=0):
    """Return ``(displacement, posterior_std)`` for fitted observations.

    Both arrays, when present, have shape ``(observations, points)``.
    ``posterior_std`` is ``None`` for the legacy point-warp policy. For offline
    fits, the last stored candidate offsets are selected by the final hard
    assignments. Flow posterior standard deviations are matched to those
    stored offsets, since cluster columns may be reordered during fitting.
    """
    backend = getattr(model, '_backend', model)
    labels = _array(model.labels_ if hasattr(model, 'labels_') else
                    backend.resp_assigned[-1]).astype(int).reshape(-1)
    x = _array(grid).reshape(-1)
    if output < 0 or output >= backend.n_outputs:
        raise ValueError('output must be a valid zero-based channel index')
    flow = bool(getattr(backend.alignment, 'uses_candidate_input_warp', False))
    uncertain = bool(getattr(backend.alignment, 'uses_posterior_flow', False))

    if flow and len(getattr(backend, 'flow_posteriors', ())) == len(labels):
        results = backend.flow_posteriors
        displacement = np.stack([_array(p.mean_displacement).reshape(-1) for p in results])
        std = (np.stack([_array(p.std_displacement).reshape(-1) for p in results])
               if uncertain else None)
    else:
        history = getattr(backend, 'x_w', None)
        if hasattr(history, 'shape') and len(history.shape) == 4:
            history = _array(history)
            if history.shape[:3] != (len(labels), len(x), backend.n_outputs):
                raise ValueError('Stored warp history does not match labels, grid and outputs')
            if np.any((labels < 0) | (labels >= history.shape[3])):
                raise ValueError('Final cluster labels exceed the stored warp candidate columns')
            displacement = history[np.arange(len(labels)), :, output, labels]
        elif flow:
            posteriors = getattr(backend, '_flow_offline_posteriors', {})
            if not posteriors or any((i, int(cluster)) not in posteriors
                                     for i, cluster in enumerate(labels)):
                raise ValueError('No complete fitted warp history is available for this schedule')
            displacement = np.stack([
                _array(posteriors[i, int(cluster)].mean_displacement).reshape(-1)
                for i, cluster in enumerate(labels)
            ])
        else:
            cache = getattr(backend, '_warp_cache_full', {})
            representatives = getattr(backend, 'f_ind_old', ())
            if not cache or any(int(cluster) >= len(representatives) or
                                (output, int(representatives[int(cluster)])) not in cache
                                for cluster in labels):
                raise ValueError('No complete fitted warp history is available for this schedule')
            displacement = np.stack([
                _array(cache[output, int(representatives[int(cluster)])][0])[i]
                for i, cluster in enumerate(labels)
            ])
        std = None
        if uncertain:
            posteriors = getattr(backend, '_flow_offline_posteriors', {})
            if posteriors:
                std_values = np.full_like(displacement, np.nan)
                by_sample = {}
                for (sample, _), result in posteriors.items():
                    by_sample.setdefault(sample, []).append(result)
                for i in range(len(labels)):
                    candidates = by_sample.get(i, ())
                    if not candidates:
                        continue
                    distances = [np.mean((_array(result.mean_displacement).reshape(-1)
                                          - displacement[i]) ** 2)
                                 for result in candidates]
                    closest = candidates[int(np.argmin(distances))]
                    if np.allclose(_array(closest.mean_displacement).reshape(-1),
                                   displacement[i], rtol=1e-5, atol=1e-8):
                        std_values[i] = _array(closest.std_displacement).reshape(-1)
                if np.isfinite(std_values).all():
                    std = std_values

    if displacement.shape != (len(labels), len(x)):
        raise ValueError('Fitted warp history does not match labels and grid')
    return displacement, std


def plot_warp_diagnostics(grid, displacements, *, labels=None,
                          posterior_std=None, grid_samples=None,
                          title='Warp diagnostics',
                          max_curves=24):
    """Plot fitted warp maps, all-observation offsets and cluster summaries.

    ``displacements`` is the offset ``g(x)-x`` with shape ``(N,T)``. Optional
    ``posterior_std`` has the same shape and represents within-observation
    warp uncertainty. The cluster envelope shows between-observation spread,
    which is a different quantity. The returned figure is never shown or
    saved by this function; callers control that policy.
    """
    x = _array(grid).reshape(-1)
    offset = _array(displacements)
    if offset.ndim == 3 and offset.shape[-1] == 1:
        offset = offset[..., 0]
    if offset.ndim != 2 or offset.shape[1] != len(x) or not len(offset):
        raise ValueError('displacements must have shape (observations, grid points)')
    if len(x) < 2 or not np.all(np.diff(x) > 0):
        raise ValueError('grid must contain at least two strictly increasing points')
    if not np.isfinite(offset).all():
        raise ValueError('displacements must be finite')
    if not isinstance(max_curves, int) or max_curves < 1:
        raise ValueError('max_curves must be a positive integer')
    if labels is None:
        assigned = np.zeros(len(offset), dtype=int)
    else:
        assigned = np.asarray(labels).reshape(-1)
        if len(assigned) != len(offset):
            raise ValueError('labels must contain one cluster per observation')
    uncertainty = None if posterior_std is None else _array(posterior_std)
    if uncertainty is not None:
        if uncertainty.shape != offset.shape or not np.isfinite(uncertainty).all():
            raise ValueError('posterior_std must be finite and match displacements')
        if np.any(uncertainty < 0):
            raise ValueError('posterior_std cannot be negative')
    draws = None if grid_samples is None else _array(grid_samples)
    if draws is not None:
        if draws.ndim == 3 and draws.shape[-1] == 1 and len(offset) == 1:
            draws = draws[None, :, :, 0]
        elif draws.ndim == 4 and draws.shape[-1] == 1:
            draws = draws[..., 0]
        if draws.ndim != 3 or draws.shape[0] != len(offset) or draws.shape[2] != len(x):
            raise ValueError('grid_samples must have shape (observations, draws, points)')
        if not np.isfinite(draws).all():
            raise ValueError('grid_samples must be finite')

    clusters = np.unique(assigned)
    palette = plt.get_cmap('tab20') if len(clusters) <= 20 else plt.get_cmap('hsv', len(clusters))
    colors = {cluster: palette(i % palette.N) for i, cluster in enumerate(clusters)}
    selected = set()
    for cluster in clusters:
        members = np.flatnonzero(assigned == cluster)
        selected.update(members[[0, len(members) // 2, -1]].tolist())
    selected.update(np.linspace(0, len(offset) - 1, min(len(offset), max_curves),
                                dtype=int).tolist())
    selected = np.array(sorted(selected), dtype=int)
    if len(selected) > max_curves:
        selected = selected[np.linspace(0, len(selected) - 1, max_curves, dtype=int)]

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), constrained_layout=True,
                             gridspec_kw={'height_ratios': [1.1, 1.35, 1.1]})
    map_axis, heat_axis, summary_axis = axes
    map_axis.plot(x, x, color='0.35', linestyle='--', linewidth=1.2, label='Identity')
    for i in selected:
        cluster = assigned[i]
        warped = x + offset[i]
        if draws is not None:
            for sample in draws[i, :min(12, draws.shape[1])]:
                map_axis.plot(x, sample, color=colors[cluster],
                              alpha=0.09, linewidth=0.65)
        map_axis.plot(x, warped, color=colors[cluster], alpha=0.45, linewidth=0.85)
        if uncertainty is not None and len(selected) <= 12:
            map_axis.fill_between(x, warped - 2 * uncertainty[i],
                                  warped + 2 * uncertainty[i],
                                  color=colors[cluster], alpha=0.06)
    map_axis.set(xlabel='Observed coordinate', ylabel='Warped coordinate g(x)',
                 title=f'Warp maps ({len(selected)} selected of {len(offset)} observations)')
    map_axis.legend(frameon=False, loc='best')

    limit = max(float(np.quantile(np.abs(offset), 0.99)), np.finfo(float).eps)
    image = heat_axis.imshow(offset, aspect='auto', origin='upper',
                             extent=(x[0], x[-1], len(offset) - 0.5, -0.5),
                             cmap='coolwarm', vmin=-limit, vmax=limit,
                             interpolation='nearest', rasterized=True)
    heat_axis.set(xlabel='Observed coordinate', ylabel='Observation index',
                  title='Displacement g(x) - x across observations')
    fig.colorbar(image, ax=heat_axis, label='Displacement')

    for cluster in clusters:
        member = offset[assigned == cluster]
        shade = colors[cluster]
        center = np.mean(member, axis=0)
        lower, upper = np.quantile(member, (0.1, 0.9), axis=0)
        summary_axis.fill_between(x, lower, upper, color=shade, alpha=0.16)
        summary_axis.plot(x, center, color=shade, linewidth=1.7,
                          label=f'Cluster {cluster} (n={len(member)})')
        if uncertainty is not None:
            mean_std = np.mean(uncertainty[assigned == cluster], axis=0)
            if len(offset) == 1:
                summary_axis.fill_between(x, center - 2 * mean_std,
                                          center + 2 * mean_std,
                                          color=shade, alpha=0.18)
            else:
                summary_axis.plot(x, center + 2 * mean_std, color=shade,
                                  linestyle=':', alpha=0.7, linewidth=0.85)
                summary_axis.plot(x, center - 2 * mean_std, color=shade,
                                  linestyle=':', alpha=0.7, linewidth=0.85)
    summary_axis.axhline(0, color='0.35', linestyle='--', linewidth=0.8)
    if len(offset) == 1:
        summary_title = ('Posterior mean displacement and ±2 SD' if uncertainty is not None
                         else 'Displacement of the observation')
    else:
        summary_title = ('Cluster mean; shaded 10–90% between-observation range'
                         + ('; dotted ±2 mean posterior SD' if uncertainty is not None else ''))
    summary_axis.set(xlabel='Observed coordinate', ylabel='Displacement',
                     title=summary_title)
    summary_axis.legend(frameon=False, ncol=min(len(clusters), 5), fontsize='small')
    fig.suptitle(title)
    return fig
