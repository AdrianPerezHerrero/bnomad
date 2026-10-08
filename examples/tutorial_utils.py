"""Data, public-result plotting and reporting shared by the user tutorials.

No inference, parameter selection or access to private model state lives here.
"""
from pathlib import Path
import hashlib
import json
import platform
from importlib import metadata
from urllib.request import urlopen

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.lines import Line2D

from hdpgpc.diagnostics import model_lds_arrays

DATA_REVISION = '615109a9e474d37626abaff6c51f1c159b26124a'
DATA_HASHES = {
    '100.npy': 'cf3c01ac96a1ae9683122a4af57f71c44d3e89eec090ed9d53c502b8ff466b89',
    '100_labels.npy': 'ea7161896ee6a6ce762bda26ae7aa9e9a4debf9543ced3f4948e18f0c3df9917',
}

# MIT-BIH annotation colours used by the original diagnostic plots.
ANNOTATION_COLORS = {
    0: 'k', 1: 'b', 2: 'r', 3: 'g', 4: 'c', 5: 'purple',
    6: 'darkgreen', 7: 'maroon', 8: 'orange', 9: 'lightgreen',
    10: 'magenta', 11: 'lightblue', 12: 'darkblue', 13: 'red',
    14: 'red', 15: 'goldenrod', 16: 'red',
}
ANNOTATION_CODES = {
    'N': 1, 'V': 2, 'R': 3, '!': 4, 'F': 5, 'L': 6,
    'A': 7, '/': 8, 'Q': 9, 'f': 10, 'E': 11, 'J': 12,
    'j': 13, 'e': 14, 'a': 15, 'S': 16,
}


def _annotation_color(label):
    if isinstance(label, bytes):
        label = label.decode('utf-8')
    return ANNOTATION_COLORS[ANNOTATION_CODES.get(str(label), 0)]


def _darker_color(color):
    """Keep an annotation hue while giving its latent curve a darker tone."""
    rgb = np.asarray(to_rgb(color))
    return tuple(rgb * 0.62) if np.any(rgb) else (0.22, 0.22, 0.22)


def _plot_latent_process(axis, state, gamma, color, *, legend=False):
    """Draw f_star_sm and a process-noise band on the latent state's own grid.

    Gamma is a process covariance, so its diagonal contributes standard
    deviations through sqrt(diag(Gamma)); this band is not the smoothed-state
    posterior covariance or the observation predictive covariance.
    """
    x = np.asarray(state.grid).reshape(-1)
    mean = np.asarray(state.smoothed_mean).reshape(-1)
    gamma = np.asarray(gamma)
    if gamma.shape != (len(mean), len(mean)) or len(x) != len(mean):
        raise ValueError('latent mean, grid and Gamma must have matching LDS dimensions')
    sd = np.sqrt(np.maximum(np.diag(gamma), 0.0))
    latent_color = _darker_color(color)
    if np.any(sd > 0):
        axis.fill_between(x, mean - 2 * sd, mean + 2 * sd,
                          color=latent_color, alpha=0.16, zorder=2,
                          label='Latent ±2√diag Γ' if legend else None)
    axis.plot(x, mean, color=latent_color, linewidth=1.65,
              linestyle='--', zorder=4,
              label='Smoothed latent mean' if legend else None)


def _plot_cluster_latent(axis, state, gamma, color, *, pca=False, legend=False):
    if not pca:
        _plot_latent_process(axis, state, gamma, color, legend=legend)
        return
    # PCA coordinates index components rather than waveform time; a separate
    # inset avoids presenting their coefficients on the observation x-axis.
    inset = axis.inset_axes((0.62, 0.57, 0.35, 0.36))
    _plot_latent_process(inset, state, gamma, color)
    inset.set_title('PCA state ±2√diag Γ', fontsize=7)
    inset.tick_params(labelsize=6)


def load_record(record_name="100"):
    """Load the specified MIT-BIH record; fetch at most 3.3 MB if no local copy is present."""
    data_file = f"{record_name}.npy"
    labels_file = f"{record_name}_labels.npy"
    required_files = [data_file, labels_file]
    
    candidates = [p / 'hdpgpc/data/mitbih' for p in
                  (Path(__file__).resolve().parents[1], Path.cwd(), *Path.cwd().parents)]
    directory = next((p for p in candidates if (p / data_file).exists()),
                     Path.cwd() / '.tutorial-data')
    directory.mkdir(parents=True, exist_ok=True)
    
    for name in required_files:
        expected = DATA_HASHES.get(name)
        path = directory / name
        if not path.exists():
            url = f'https://raw.githubusercontent.com/AdrianPerezHerrero/HDP-GPC/{DATA_REVISION}/hdpgpc/data/mitbih/{name}'
            print(f'Downloading {name} from the pinned repository record')
            with urlopen(url, timeout=60) as response:
                payload = response.read(3_400_001)
            if len(payload) > 3_400_000:
                raise ValueError('Record download size exceeded')
            if expected and hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError('Record checksum mismatch')
            path.write_bytes(payload)
        if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Unexpected record bytes: {path}')
            
    data = np.load(directory / data_file, allow_pickle=False)
    annotations = np.load(directory / labels_file, allow_pickle=False)
    
    print('Waveforms:', data.shape, data.dtype, '; annotations:', annotations.shape)
    print('Annotation lengths differ: no ground-truth pairing or accuracy claim.')
    return data, annotations


def environment():
    result = {'python': platform.python_version(), 'platform': platform.platform(),
              'versions': {n: metadata.version(n) for n in
                           ('hdpgpc', 'numpy', 'scipy', 'torch', 'gpytorch', 'linear-operator', 'pyro-ppl', 'scikit-learn')}}
    print(json.dumps(result, indent=2))
    return result


def plot_input(raw, grid, values):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), constrained_layout=True)
    for i in range(min(3, len(raw))):
        axes[0].plot(np.arange(raw.shape[1]), raw[i, :, 0], label=f'Beat {1700+i}')
        axes[1].plot(grid.ravel(), values[i, :, 0], marker='.', label=f'Beat {1700+i}')
    axes[0].set(xlabel='Original point index', ylabel='Stored amplitude', title='Record channel 0')
    axes[1].set(xlabel='Rescaled coordinate (0–4)', ylabel='Scaled amplitude', title='15-point teaching input')
    axes[0].legend(frameon=False)
    return fig


def plot_results(model, values, grid=None, title='Clustering results'):
    """Plot hard assignments and cluster fits with their included observations.

    Each cluster gets one row of panels, one per output. Included observations
    are drawn with low opacity so dense clusters remain readable; the fitted
    observation mean and its two-standard-deviation band are drawn on top.
    The darker dashed curve shows the smoothed latent mean, with a separate
    band from the diagonal of the final process covariance Gamma.
    """
    values = np.asarray(values)
    if values.ndim != 3:
        raise ValueError('values must have shape (observations, points, outputs)')
    assignments = np.asarray(model.labels_).reshape(-1)
    if len(assignments) != len(values):
        raise ValueError('model labels and values must contain the same observations')

    cluster_ids = np.asarray(model.cluster_ids_).reshape(-1)
    count = values.shape[2]
    pca = model.config.observation.pca_grid is not None
    lds_matrices = [model_lds_arrays(model, output=output) for output in range(count)]
    colors = plt.get_cmap('tab20')
    fig = plt.figure(
        figsize=(5.0 * count, max(4.0, 2.35 * (len(cluster_ids) + 1))),
        constrained_layout=True,
    )
    layout = fig.add_gridspec(
        len(cluster_ids) + 1,
        count,
        height_ratios=[0.6] + [1.0] * len(cluster_ids),
    )

    assignment_axis = fig.add_subplot(layout[0, :])
    observation_index = np.arange(len(assignments))
    for position, cluster in enumerate(cluster_ids):
        color = colors(position % colors.N)
        member = assignments == cluster
        assignment_axis.scatter(
            observation_index[member], assignments[member],
            s=8, color=color, label=f'Cluster {cluster}', linewidths=0,
        )
    assignment_axis.set(
        xlabel='Observation index', ylabel='Cluster',
        title='Hard assignments', xlim=(-1, max(len(assignments), 1)),
    )
    assignment_axis.set_yticks(cluster_ids)
    assignment_axis.legend(frameon=False, ncol=min(len(cluster_ids), 8), loc='upper right')

    for position, cluster in enumerate(cluster_ids):
        color = colors(position % colors.N)
        member = assignments == cluster
        prediction = model.predictive(int(cluster), grid=grid)
        latent_states = model.posterior(int(cluster))
        x = prediction.grid.reshape(-1)
        if len(x) != values.shape[1]:
            raise ValueError('plot grid and observation point count must match')
        for output in range(count):
            axis = fig.add_subplot(layout[position + 1, output])
            for sample in values[member, :, output]:
                axis.plot(x, sample, color=color, alpha=0.09, linewidth=0.55)
            mean = prediction.mean[:, output]
            # Clip only for display; preserve the predictive covariance itself.
            sd = np.sqrt(np.maximum(np.diag(prediction.covariance[output]), 0))
            axis.fill_between(x, mean - 2 * sd, mean + 2 * sd, color=color, alpha=0.2)
            axis.plot(x, mean, color=color, linewidth=1.8, zorder=3,
                      label='Observation mean' if position == output == 0 else None)
            _plot_cluster_latent(
                axis, latent_states[output], lds_matrices[output][int(cluster)]['Gamma'],
                color, pca=pca, legend=position == output == 0,
            )
            if position == output == 0:
                axis.legend(loc='best', fontsize=7, frameon=False)
            axis.set(
                xlabel='Rescaled coordinate', ylabel='Scaled amplitude',
                title=f'Cluster {cluster}: output {output} ({int(member.sum())} samples)',
            )

    if title:
        fig.suptitle(title)
    return fig


def plot_record_results(model, values, annotations, *, grid=None, record='', metrics=None,
                        channels=None, warped=False):
    """Plot the full-record overview and annotated cluster gallery.

    A separate figure is returned for each output channel. Transition entries
    are row-normalized counts of consecutive hard assignments. The annotation
    colour of every sample is independent of its assigned cluster.
    """
    values = np.asarray(values)
    if values.ndim != 3 or not all(values.shape):
        raise ValueError('values must have shape (observations, points, outputs)')
    assignments = np.asarray(model.labels_).reshape(-1)
    if len(assignments) != len(values):
        raise ValueError('model labels and values must contain the same observations')
    cluster_ids = np.asarray(model.cluster_ids_).reshape(-1)
    if not len(cluster_ids):
        raise ValueError('model must contain at least one assigned cluster')
    pca = model.config.observation.pca_grid is not None

    if annotations is not None:
        annotations = np.asarray(annotations).reshape(-1)
        if len(annotations) != len(assignments):
            raise ValueError('annotations must align one-to-one with observations')
        annotations = np.array([
            label.decode('utf-8') if isinstance(label, bytes) else str(label)
            for label in annotations
        ])

    channels = tuple(range(values.shape[2])) if channels is None else tuple(channels)
    if len(channels) != values.shape[2]:
        raise ValueError('channels must name each output in values')

    predictions = {int(cluster): model.predictive(int(cluster), grid=grid)
                   for cluster in cluster_ids}
    latent_states = {int(cluster): model.posterior(int(cluster))
                     for cluster in cluster_ids}
    for prediction in predictions.values():
        if len(prediction.grid) != values.shape[1]:
            raise ValueError('plot grid and observation point count must match')

    majority = {}
    for cluster in cluster_ids:
        if annotations is None:
            majority[int(cluster)] = None
        else:
            labels, counts = np.unique(annotations[assignments == cluster], return_counts=True)
            majority[int(cluster)] = labels[np.argmax(counts)]

    transition_counts = np.zeros((len(cluster_ids), len(cluster_ids)), dtype=int)
    if len(assignments) > 1:
        previous = np.searchsorted(cluster_ids, assignments[:-1])
        following = np.searchsorted(cluster_ids, assignments[1:])
        np.add.at(transition_counts, (previous, following), 1)
    totals = transition_counts.sum(axis=1, keepdims=True)
    transitions = np.divide(
        transition_counts, totals,
        out=np.zeros_like(transition_counts, dtype=float), where=totals != 0,
    )

    figures = []
    gallery_rows = (len(cluster_ids) + 2) // 3
    palette = (plt.get_cmap('tab20') if len(cluster_ids) <= 20
               else plt.get_cmap('hsv', len(cluster_ids)))
    tab20_order = list(range(0, 20, 2)) + list(range(1, 20, 2))
    cluster_colors = {
        int(cluster): palette(tab20_order[position] if len(cluster_ids) <= 20 else position)
        for position, cluster in enumerate(cluster_ids)
    }
    # Keep panel sizes fixed; only the figure height grows with gallery rows.
    overview_height = 4.5
    gallery_header_height = 0.5
    cluster_row_height = 3.5
    cluster_row_spacing = 0.3
    for output, channel in enumerate(channels):
        lds_matrices = model_lds_arrays(model, output=output)
        title_margin = 1.0 if warped else 0.6
        figure_height = (title_margin + overview_height + gallery_header_height
                         + (cluster_row_height + cluster_row_spacing) * gallery_rows)
        fig = plt.figure(figsize=(18, figure_height), constrained_layout=True)
        layout = fig.add_gridspec(
            gallery_rows + 2, 3,
            height_ratios=[overview_height, gallery_header_height]
                          + [cluster_row_height] * gallery_rows,
        )
        assignment_axis = fig.add_subplot(layout[0, 0])
        models_axis = fig.add_subplot(layout[0, 1])
        assignment_axis.set_box_aspect(0.75)
        models_axis.set_box_aspect(0.75)
        transition_layout = layout[0, 2].subgridspec(2, 1, height_ratios=[3.2, 1])
        transition_axis = fig.add_subplot(transition_layout[0, 0])
        cluster_legend_axis = fig.add_subplot(transition_layout[1, 0])
        cluster_legend_axis.axis('off')
        gallery_header = fig.add_subplot(layout[1, :])
        gallery_header.axis('off')

        for position, cluster in enumerate(cluster_ids):
            cluster = int(cluster)
            member = assignments == cluster
            label = majority[cluster]
            overview_color = cluster_colors[cluster]
            annotation_color = (_annotation_color(label) if label is not None
                                else overview_color)
            assignment_axis.scatter(
                np.flatnonzero(member), assignments[member], s=7,
                color=overview_color, linewidths=0, rasterized=True,
            )
            prediction = predictions[cluster]
            x = prediction.grid.reshape(-1)
            mean = prediction.mean[:, output]
            models_axis.plot(x, mean, color=overview_color, linewidth=1.6)

            axis = fig.add_subplot(layout[position // 3 + 2, position % 3])
            axis.set_box_aspect(0.59)
            member_values = values[member, :, output]
            if len(member_values):
                segments = np.stack((np.broadcast_to(x, member_values.shape), member_values), axis=-1)
                sample_colors = (
                    [to_rgba(_annotation_color(sample_label),
                             alpha=0.10 if sample_label == label else 0.25)
                     for sample_label in annotations[member]]
                    if annotations is not None
                    else [to_rgba(annotation_color, alpha=0.12)] * len(member_values)
                )
                axis.add_collection(LineCollection(
                    segments, colors=sample_colors, linewidths=0.55,
                    rasterized=True,
                ))
            # The predictive covariance is unchanged; clipping only protects the display.
            sd = np.sqrt(np.maximum(np.diag(prediction.covariance[output]), 0))
            axis.fill_between(x, mean - 2 * sd, mean + 2 * sd,
                              color=annotation_color, alpha=0.22)
            axis.plot(x, mean, color=annotation_color, linewidth=1.8, zorder=3)
            _plot_cluster_latent(
                axis, latent_states[cluster][output], lds_matrices[cluster]['Gamma'],
                annotation_color, pca=pca,
            )
            axis.autoscale_view()
            axis.set(xlabel='Rescaled coordinate', ylabel='Scaled amplitude',
                     title=f'Cluster {cluster} · {member.sum()} samples' +
                           (f' · majority {label}' if label else ''))

        assignment_axis.set(xlabel='Beat index', ylabel='Cluster', title='Regimes through the record',
                            xlim=(-1, len(assignments)))
        assignment_axis.set_yticks(cluster_ids)
        assignment_axis.grid(axis='x', alpha=0.2)
        models_axis.set(xlabel='Rescaled coordinate', ylabel='Scaled amplitude',
                        title='Fitted model means')

        image = transition_axis.imshow(transitions, vmin=0, vmax=1, cmap='Blues', aspect='auto')
        transition_axis.set(title='Transitions between clusters', xlabel='Next cluster',
                            ylabel='Current cluster')
        transition_axis.set_xticks(np.arange(len(cluster_ids)), cluster_ids)
        transition_axis.set_yticks(np.arange(len(cluster_ids)), cluster_ids)
        for row, column in zip(*np.nonzero(transition_counts)):
            value = transitions[row, column]
            transition_axis.text(column, row, f'{value:.2f}', ha='center', va='center',
                                 fontsize=8, color='white' if value > 0.5 else 'black')
        fig.colorbar(image, ax=transition_axis, fraction=0.046, pad=0.04, label='Probability')
        cluster_handles = [
            Line2D([0], [0], color=cluster_colors[int(cluster)], linewidth=2,
                   label=f'Cluster {cluster}')
            for cluster in cluster_ids
        ]
        cluster_legend_axis.legend(handles=cluster_handles, loc='center', frameon=False,
                                   ncol=min(3, len(cluster_ids)), fontsize=8)

        gallery_header.text(
            0, 0.5,
            ('Clusters · solid: PCA reconstruction (±2 SD) · inset: latent PCA state / Γ · faint: beats'
             if pca else
             'Clusters · solid: C f_sm (±2 SD) · dashed dark: f_sm (±2√diag Γ) · faint: beats'),
            va='center', fontsize=10, fontweight='bold',
        )
        if annotations is not None:
            used_labels = sorted(set(annotations), key=lambda label: (ANNOTATION_CODES.get(label, 0), label))
            handles = [Line2D([0], [0], color=_annotation_color(label), linewidth=2,
                              label=label) for label in used_labels]
            gallery_header.legend(handles=handles, loc='center right', frameon=False,
                                  ncol=min(len(handles), 8), fontsize=9)

        for position in range(len(cluster_ids), gallery_rows * 3):
            fig.add_subplot(layout[position // 3 + 2, position % 3]).axis('off')

        cluster_word = 'cluster' if len(cluster_ids) == 1 else 'clusters'
        headline = (f'Record {record} · channel {channel} · {len(values):,} beats · '
                    f'{len(cluster_ids)} {cluster_word}')
        if metrics is not None:
            headline += (f"\nPurity {metrics['purity']:.3f} · ARI {metrics['adjusted_rand_index']:.3f}"
                         f" · NMI {metrics['normalized_mutual_information']:.3f}")
        else:
            headline += '\nAnnotation metrics unavailable'
        if warped:
            headline += '\nInput waveforms use the original grid; model bands use aligned coordinates'
        fig.suptitle(headline, fontsize=15, fontweight='bold')
        figures.append(fig)
    return tuple(figures)


def plot_latent(model):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), constrained_layout=True)
    for cluster in model.cluster_ids_:
        for state in model.posterior(int(cluster)):
            label = f'C{cluster}, channel {state.output}, t={state.observation_index}'
            axes[0].plot(state.grid.ravel(), state.filtered_mean, label=label)
            axes[1].plot(state.grid.ravel(), state.smoothed_mean, label=label)
    for axis, title in zip(axes, ('Latest stored filtered mean', 'Latest stored smoothed mean')):
        axis.set(xlabel='Rescaled coordinate', ylabel='Latent mean', title=title)
        axis.legend(frameon=False, fontsize=8)
    return fig


def snapshot(model):
    """JSON-ready public posterior quantities, excluding timing/environment."""
    components = []
    for c in model.cluster_ids_:
        p = model.predictive(int(c))
        states = [dict(output=s.output, observation_index=s.observation_index,
                       filtered_mean=s.filtered_mean.tolist(), filtered_covariance=s.filtered_covariance.tolist(),
                       smoothed_mean=s.smoothed_mean.tolist(), smoothed_covariance=s.smoothed_covariance.tolist())
                  for s in model.posterior(int(c))]
        components.append(dict(cluster=int(c), mean=p.mean.tolist(), covariance=p.covariance.tolist(), latent=states))
    return dict(labels=model.labels_.tolist(), n_clusters=model.n_clusters_, components=components)


def save_report(name, model, values, config, versions, seconds, extra=None, record_name="100"):
    report = dict(config=config.to_dict(), environment=versions, inference_seconds=seconds,
                  data_revision=DATA_REVISION, record_sha256=DATA_HASHES.get(f'{record_name}.npy', 'unknown_hash'),
                  input_sha256=hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest(),
                  scientific_result=snapshot(model), extra=extra or {})
    path = Path('tutorial-results') / (name + '.json')
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print('Saved', path, '; inference seconds:', round(seconds, 3))
    return report
