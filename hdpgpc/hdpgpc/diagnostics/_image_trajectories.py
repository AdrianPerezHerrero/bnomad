"""Read-only image reconstructions and retained cluster LDS trajectories."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ._plot_selection import cluster_plot_ids, empty_cluster_figure


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
    return np.array(value, dtype=np.float64, copy=True)


def model_latent_trajectories(model, *, output=0):
    """Copy retained state histories and one-step means under the final A.

    Indices identify included observations, not necessarily MAP members in soft
    mode. State zero is returned separately as the initial prior. A trajectory
    step is between consecutive observations included in that cluster: elapsed
    time outside the cluster is not automatically propagated. This diagnostic
    requires retained full histories, as in ordinary offline inference.
    """
    backend = getattr(model, '_backend', model)
    if backend is None or not hasattr(backend, 'gpmodels') or not 0 <= output < len(backend.gpmodels):
        raise ValueError('A fitted model and valid output are required')
    cluster_ids = list(model.cluster_ids_) if hasattr(model, 'cluster_ids_') else range(len(backend.gpmodels[output]))
    result = {}
    for cluster in cluster_ids:
        gp = backend.gpmodels[output][int(cluster)]
        indices = np.asarray(gp.indexes, dtype=np.int64)
        if (not len(indices) or np.any(indices < 0) or len(gp.f_star_sm) != len(indices) + 1
                or len(gp.cov_f_sm) != len(indices) + 1 or np.any(np.diff(indices) <= 0)):
            raise ValueError(f'Cluster {cluster} lacks a full ordered posterior history; use offline inference')
        means = np.stack([_array(mean).reshape(-1) for mean in gp.f_star_sm[1:]])
        covariance = np.stack([_array(cov) for cov in gp.cov_f_sm[1:]])
        A = _array(gp.A[-1])
        result[int(cluster)] = {
            'indices': indices.copy(), 'means': means, 'covariances': covariance,
            'forecast_means': means @ A.T,
            'prior_mean': _array(gp.f_star_sm[0]).reshape(-1),
            'prior_covariance': _array(gp.cov_f_sm[0]),
        }
    return result


def _colors(ids):
    palette = plt.get_cmap('tab20' if len(ids) <= 20 else 'turbo')
    positions = np.arange(len(ids)) if len(ids) <= 20 else np.linspace(0., 1., len(ids))
    return {int(cluster): palette(position) for cluster, position in zip(ids, positions)}


def _coordinates(means):
    return np.column_stack((means[:, 0], means[:, 1] if means.shape[1] > 1 else np.zeros(len(means))))


def _runs(indices):
    """Do not draw observed trajectory edges across MAP membership gaps."""
    return np.split(np.arange(len(indices)), np.flatnonzero(np.diff(indices) != 1) + 1)


def _times(times, count):
    times = np.arange(count, dtype=float) if times is None else np.asarray(times, dtype=float)
    if times.shape != (count,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('times must be finite and strictly increasing, one per observation')
    return times


def plot_latent_trajectories(model, *, times=None, output=0, title='', trajectories=None,
                             min_cluster_samples=1, time_label='Video time (s)', time_unit='s'):
    """Per cluster: state path, component uncertainty and final-A changes.

    Uses the first two common latent coordinates (one-dimensional states get a
    zero second coordinate). Components are not refit for this visualization.
    Orange arrows show A*m-m in that projection; they are model changes, not
    observed future frames. Two-dimensional arrows omit other state dimensions.
    Only clusters with at least min_cluster_samples final MAP assignments get
    detail panels; histories and labels are never modified.
    """
    labels = np.asarray(model.labels_, dtype=int)
    times = _times(times, len(labels))
    paths = model_latent_trajectories(model, output=output) if trajectories is None else trajectories
    visible = set(cluster_plot_ids(labels, min_cluster_samples).tolist())
    ids = [cluster for cluster in sorted(paths) if cluster in visible]
    colors = _colors(sorted(paths))
    if not ids:
        return empty_cluster_figure(title or 'Latent trajectories', min_cluster_samples)
    fig, axes = plt.subplots(len(ids), 3, figsize=(16, 3.6 * len(ids)), squeeze=False,
                             constrained_layout=True)
    heading = (title + ' | ' if title else '') + 'Smoothed latent trajectories and one-step LDS changes'
    if min_cluster_samples > 1:
        heading += f' | {len(ids)}/{len(paths)} clusters with >= {min_cluster_samples} assigned samples'
    fig.suptitle(heading)
    for row, cluster in enumerate(ids):
        state = paths[cluster]
        indices = state['indices']
        owned = labels[indices] == cluster
        indices = indices[owned]
        means, covariances = state['means'][owned], state['covariances'][owned]
        forecasts = state['forecast_means'][owned]
        color = colors[cluster]
        axes[row, 0].set(title=f'Cluster {cluster}: {len(indices)} MAP frames (color = time)',
                         xlabel='Latent component 1', ylabel='Latent component 2')
        axes[row, 1].set(title='Smoothed components +/- 2 posterior SD',
                         xlabel=time_label, ylabel='Latent coefficient')
        axes[row, 2].set(title='Orange: final A one-step change',
                         xlabel='Latent component 1', ylabel='Latent component 2')
        if not len(indices):
            continue
        xy, predicted = _coordinates(means), _coordinates(forecasts)
        for run in _runs(indices):
            axes[row, 0].plot(xy[run, 0], xy[run, 1], color=color, alpha=.7, linewidth=1)
            for component in range(min(3, means.shape[1])):
                component_color = plt.get_cmap('tab10')(component)
                t = times[indices[run]]
                m = means[run, component]
                sd = np.sqrt(np.maximum(covariances[run, component, component], 0.))
                axes[row, 1].plot(t, m, color=component_color, linewidth=1)
                axes[row, 1].fill_between(t, m - 2 * sd, m + 2 * sd, color=component_color, alpha=.15)
        axes[row, 0].scatter(xy[:, 0], xy[:, 1], c=times[indices], cmap='viridis', s=10)
        for position, marker in ((0, 'o'), (-1, 'X')):
            axes[row, 0].scatter(*xy[position], s=50, marker=marker, color=color, edgecolor='k')
            axes[row, 0].annotate(f'{times[indices[position]]:g}{time_unit}', xy[position], fontsize=8,
                                   xytext=(4, 4), textcoords='offset points')
        selected = np.unique(np.linspace(0, len(indices) - 1, min(20, len(indices)), dtype=int))
        delta = predicted[selected] - xy[selected]
        axes[row, 2].scatter(xy[:, 0], xy[:, 1], color=color, s=8, alpha=.5)
        axes[row, 2].quiver(xy[selected, 0], xy[selected, 1], delta[:, 0], delta[:, 1],
                            angles='xy', scale_units='xy', scale=1, color='tab:orange', width=.004)
        axes[row, 2].update_datalim(np.vstack((xy, predicted[selected])))
        axes[row, 2].autoscale_view()
        axes[row, 1].legend(handles=[Line2D([], [], color=plt.get_cmap('tab10')(j), label=f'f{j + 1}')
                                    for j in range(min(3, means.shape[1]))], fontsize=8, loc='best')
        for axis in axes[row]:
            axis.grid(alpha=.15)
    return fig


def plot_image_clustering_summary(model, values, annotations, *, image_shape, times=None,
                                  frames_per_cluster=10, trajectories=None, record='', metrics=None,
                                  grid=None, ordered=True, warped=False, show_latent=True,
                                  annotation_colors=None, evaluation_mask=None,
                                  representation_name='PCA', random_state=42,
                                  calibration_encoder=None, min_cluster_samples=1,
                                  time_label='Video time (s)', time_unit='s',
                                  reference_title='Reference anatomical regions (evaluation only)',
                                  image_limits=(0., 1.)):
    """Image-native replacement for the waveform gallery; returns [figure].

    Frame rows show observations, fixed affine encoder/decoder reconstructions, actual smoothed
    LDS reconstructions and one-step means from final A. All use common display
    limits (default [0,1]); reconstruction clipping is display-only. Image labels are evaluation
    metadata. Only fixed affine one-output decoders are supported here.
    Gallery frames are sampled without replacement and displayed in time order.
    The majority-label timeline uses only evaluation-included reference labels;
    clusters with no reference label are gray. Sampling is seeded independently
    of model inference and never changes the fitted model or its assignments.
    An optional (pixels, latent dimension) calibration_encoder supplies raw
    reconstruction features, e.g. SFA. This changes only the calibration row;
    smoothed states and forecasts always use the model's Gaussian LDS posterior.
    min_cluster_samples filters galleries and latent paths by final MAP counts;
    assignment timelines, transition counts and metrics retain all observations.
    """
    from ..emissions._fixed_observation import AffineGaussianObservation

    values = np.asarray(values, dtype=float)
    if len(image_limits) != 2 or not np.isfinite(image_limits).all() or image_limits[0] >= image_limits[1]:
        raise ValueError('image_limits must be two increasing finite intensity values')
    low, high = image_limits
    shape = tuple(image_shape)
    if (values.ndim != 3 or values.shape[2] != 1 or len(shape) != 2
            or int(np.prod(shape)) != values.shape[1] or frames_per_cluster < 1):
        raise ValueError('Images require (frames, height*width, 1) values and positive gallery size')
    labels = np.asarray(model.labels_, dtype=int)
    annotations = np.asarray(annotations).astype(str)
    if labels.shape != (len(values),) or annotations.shape != labels.shape:
        raise ValueError('Image labels/annotations must align with frames')
    evaluated = (np.ones(len(values), dtype=bool) if evaluation_mask is None
                 else np.asarray(evaluation_mask, dtype=bool))
    if evaluated.shape != labels.shape:
        raise ValueError('Evaluation mask must align with image frames')
    times = _times(times, len(labels))
    paths = model_latent_trajectories(model) if trajectories is None else trajectories
    ids, colors = sorted(paths), _colors(sorted(paths))
    eligible = set(cluster_plot_ids(labels, min_cluster_samples).tolist())
    visible_ids = [cluster for cluster in ids if cluster in eligible]
    members_by_cluster = {cluster: np.flatnonzero(labels == cluster) for cluster in ids}
    majorities = {}
    majority_correct = 0
    for cluster, members in members_by_cluster.items():
        reference = annotations[members[evaluated[members]]]
        classes_in_cluster, counts = np.unique(reference, return_counts=True)
        majorities[cluster] = str(classes_in_cluster[np.argmax(counts)]) if len(counts) else None
        majority_correct += int(counts.max()) if len(counts) else 0
    purity = majority_correct / evaluated.sum() if evaluated.any() else None
    rng = np.random.default_rng(random_state)
    fig = plt.figure(figsize=(18, 5.8 + 4.1 * len(visible_ids)), constrained_layout=True)
    layout = fig.add_gridspec(len(visible_ids) + 1, 1, height_ratios=[5.5] + [4.] * len(visible_ids))
    top = layout[0].subgridspec(1, 3)
    timeline = top[0].subgridspec(3, 1)
    truth_axis, assignment_axis = fig.add_subplot(timeline[0]), fig.add_subplot(timeline[1])
    majority_axis = fig.add_subplot(timeline[2], sharex=assignment_axis)
    path_axis, transition_axis = fig.add_subplot(top[1]), fig.add_subplot(top[2])
    classes = sorted(np.unique(annotations))
    class_colors = {name: plt.get_cmap('tab10')(i % 10) for i, name in enumerate(classes)}
    truth_axis.scatter(times, [classes.index(label) for label in annotations],
                       c=[class_colors[label] for label in annotations], s=8)
    truth_axis.set(title=reference_title, xlabel=time_label)
    truth_axis.set_yticks(np.arange(len(classes)), classes, fontsize=7)
    for tick, label in zip(truth_axis.get_yticklabels(), classes):
        tick.set_color(class_colors[label])
    assignment_axis.scatter(times, labels, c=[colors[int(label)] for label in labels], s=8)
    assignment_axis.set(title='Inferred regimes', xlabel=time_label, ylabel='Cluster')
    assignment_axis.set_yticks(ids)
    majority_axis.scatter(times, labels,
                          c=[class_colors.get(majorities[int(cluster)], '0.6') for cluster in labels], s=8)
    majority_axis.set(title='Inferred regimes colored by majority label',
                      xlabel=time_label, ylabel='Cluster')
    majority_axis.set_yticks(ids)
    majority_axis.set_ylim(assignment_axis.get_ylim())
    for cluster in visible_ids:
        state = paths[cluster]
        owned = labels[state['indices']] == cluster
        rows = state['indices'][owned]
        xy = _coordinates(state['means'][owned])
        for run in _runs(rows):
            path_axis.plot(xy[run, 0], xy[run, 1], color=colors[cluster], linewidth=.8)
        path_axis.scatter(xy[:, 0], xy[:, 1], color=colors[cluster], s=7)
    path_axis.set(title=f'Smoothed states: common {representation_name} coordinates',
                  xlabel='Latent component 1', ylabel='Latent component 2')
    if not visible_ids:
        path_axis.text(.5, .5, f'No clusters with >= {min_cluster_samples} samples',
                       ha='center', va='center', transform=path_axis.transAxes)
    transitions = np.zeros((len(ids), len(ids)))
    slots = {cluster: index for index, cluster in enumerate(ids)}
    for a, b in zip(labels[:-1], labels[1:]):
        transitions[slots[int(a)], slots[int(b)]] += 1
    transitions /= np.maximum(transitions.sum(1, keepdims=True), 1)
    transition_axis.imshow(transitions, vmin=0, vmax=1, cmap='Blues', aspect='auto')
    transition_axis.set(title='Empirical consecutive-frame transitions', xlabel='Next cluster', ylabel='Current cluster')
    transition_axis.set_xticks(range(len(ids)), ids)
    transition_axis.set_yticks(range(len(ids)), ids)
    for row, cluster in enumerate(visible_ids, 1):
        gp = model._backend.gpmodels[0][cluster]
        decoder = getattr(gp, 'fixed_observation_model', None)
        if decoder is None:
            # Earlier legacy proposal copies retained affine arrays but lost
            # the per-emission attachment. The declared model-level decoder
            # is usable only if it agrees with this cluster's retained arrays.
            decoder = getattr(model._backend, 'fixed_observation_model', None)
        if not isinstance(decoder, AffineGaussianObservation):
            raise ValueError(f'Image gallery requires a fixed affine observation model for cluster {cluster}')
        direct_grid = getattr(gp, 'direct_observation_grid', None)
        direct_offset = getattr(gp, 'direct_observation_offset', None)
        if getattr(gp, 'factor_analysis', False):
            direct_offset = decoder.offset
        if (direct_grid is None or direct_offset is None
                or not np.array_equal(_array(gp.C[-1]), decoder.C)
                or not np.array_equal(_array(direct_grid), decoder.grid)
                or not np.array_equal(_array(direct_offset).reshape(-1), decoder.offset)):
            raise ValueError(f'Cluster {cluster} affine parameters disagree with its fixed image decoder')
        state, members = paths[cluster], members_by_cluster[cluster]
        selected = np.sort(rng.choice(members, size=min(frames_per_cluster, len(members)), replace=False))
        all_means = np.vstack((state['prior_mean'][None], state['means']))
        states = all_means[np.searchsorted(state['indices'], selected, side='right')]
        pixels = values[selected, :, 0]
        C, offset, A = decoder.C, decoder.offset, _array(gp.A[-1])
        encoder = (np.linalg.pinv(C).T if calibration_encoder is None
                   else np.asarray(calibration_encoder, dtype=float))
        if encoder.shape != C.shape or not np.isfinite(encoder).all():
            raise ValueError('Calibration encoder must have finite (pixels, latent dimension) entries')
        raw_projection = offset + ((pixels - offset) @ encoder) @ C.T
        reconstructed = offset + states @ C.T
        forecast = offset + (states @ A.T) @ C.T
        majority = majorities[cluster] or 'unlabeled'
        gallery = layout[row].subgridspec(5, frames_per_cluster,
                                         height_ratios=[.3, 1, 1, 1, 1])
        heading = fig.add_subplot(gallery[0, :])
        heading.axis('off')
        heading.text(0., .5, f'Cluster {cluster} | n={len(members)} | majority {majority} | '
                     f'{len(selected)} random observations in time order',
                     transform=heading.transAxes, va='center', fontsize=10)
        rows = (('Observed', pixels), (f'{representation_name} only', raw_projection),
                ('Smoothed LDS', reconstructed), ('A one-step', forecast))
        for image_row, (name, frames) in enumerate(rows):
            for column in range(frames_per_cluster):
                axis = fig.add_subplot(gallery[image_row + 1, column])
                axis.set_xticks([]); axis.set_yticks([])
                if column == 0:
                    axis.set_ylabel(name, fontsize=9)
                if column >= len(selected):
                    axis.axis('off')
                    continue
                axis.imshow(np.clip(frames[column].reshape(shape), low, high), cmap='gray', vmin=low, vmax=high)
                if image_row == 0:
                    axis.set_title(f'{times[selected[column]]:g}{time_unit} | {annotations[selected[column]]}', fontsize=8)
                for spine in axis.spines.values():
                    spine.set_edgecolor(colors[cluster]); spine.set_linewidth(1.5)
    metric_text = f' | Purity {purity:.3f}' if purity is not None else ' | Purity N/A'
    if metrics:
        metric_text += f' | ARI {metrics["adjusted_rand_index"]:.3f} | NMI {metrics["normalized_mutual_information"]:.3f}'
    selection_text = (f'\nDetails: {len(visible_ids)}/{len(ids)} clusters with >= {min_cluster_samples} samples; '
                      'overview and scores include all observations' if min_cluster_samples > 1 else '')
    fig.suptitle(f'{record} | {len(values)} frames | {len(ids)} clusters{metric_text}\n'
                 f'Label metrics: {evaluated.sum()}/{len(values)} frames; '
                 'A one-step is a native cluster step, not an observed future frame' + selection_text)
    return [fig]
