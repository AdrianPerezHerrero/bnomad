"""Dataset-independent annotated cluster summaries on the observation grid."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.lines import Line2D
from ._lds_plots import model_lds_arrays


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
    # Fixed decoder coordinates index components rather than waveform time; a separate
    # inset avoids presenting their coefficients on the observation x-axis.
    inset = axis.inset_axes((0.62, 0.57, 0.35, 0.36))
    _plot_latent_process(inset, state, gamma, color)
    inset.set_title('Latent state ±2√diag Γ', fontsize=7)
    inset.tick_params(labelsize=6)


def plot_clustering_summary(model, values, annotations, *, grid=None, record='', metrics=None,
                        channels=None, warped=False, ordered=True, show_latent=True,
                        annotation_colors=None):
    """Plot the full-record overview and annotated cluster gallery.

    A separate figure is returned for each output channel. Transition entries
    are row-normalized counts of consecutive hard assignments. The annotation
    colour of every sample is independent of its assigned cluster.
    Set ordered=False for independent datasets: labels use a generic palette
    and the overview shows class composition instead of temporal transitions.
    show_latent=False omits the latent process and Gamma overlays.
    annotation_colors optionally overrides label colours for non-ECG datasets.
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
    pca = (model.config.observation.pca_grid is not None or
           model.config.observation.fixed_grid is not None)

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
    latent_states = ({int(cluster): model.posterior(int(cluster))
                      for cluster in cluster_ids} if show_latent else {})
    for prediction in predictions.values():
        if len(prediction.grid) != values.shape[1]:
            raise ValueError('plot grid and observation point count must match')

    majority = {}
    if not ordered and annotations is not None:
        classes = np.unique(annotations)
        label_palette = plt.get_cmap('tab10' if len(classes) <= 10 else
                                    'tab20' if len(classes) <= 20 else 'turbo')
        color_positions = (list(range(len(classes))) if len(classes) <= 10 else
                           list(range(0, 20, 2)) + list(range(1, 20, 2))
                           if len(classes) <= 20 else
                           np.linspace(0, 1, len(classes)))
        label_colors = {label: label_palette(color_positions[i])
                        for i, label in enumerate(classes)}
    else:
        label_colors = {}

    if annotation_colors is not None:
        label_colors.update({str(label): color for label, color in annotation_colors.items()})

    def annotation_color_for(label):
        return label_colors[label] if label in label_colors else _annotation_color(label)

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
        lds_matrices = model_lds_arrays(model, output=output) if show_latent else {}
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
            annotation_color = (annotation_color_for(label) if label is not None
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
                    [to_rgba(annotation_color_for(sample_label),
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
            if show_latent:
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
        if not ordered:
            assignment_axis.set(xlabel='Sample index (fit order)',
                                title='Final assignments (arbitrary row order)')
        assignment_axis.set_yticks(cluster_ids)
        assignment_axis.grid(axis='x', alpha=0.2)
        models_axis.set(xlabel='Rescaled coordinate', ylabel='Scaled amplitude',
                        title='Fitted model means')

        if ordered:
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
        elif annotations is not None:
            composition = np.array([
                [np.sum((assignments == cluster) & (annotations == label))
                 for cluster in cluster_ids] for label in classes], dtype=float)
            composition /= np.maximum(composition.sum(axis=0, keepdims=True), 1)
            image = transition_axis.imshow(composition, vmin=0, vmax=1,
                                           cmap='Blues', aspect='auto')
            transition_axis.set(title='Class composition within clusters',
                                xlabel='Cluster', ylabel='True class')
            transition_axis.set_xticks(np.arange(len(cluster_ids)), cluster_ids)
            displayed = np.unique(np.linspace(0, len(classes) - 1,
                                             min(20, len(classes)), dtype=int))
            transition_axis.set_yticks(displayed, classes[displayed])
            fig.colorbar(image, ax=transition_axis, fraction=0.046, pad=0.04,
                         label='Cluster fraction')
        else:
            transition_axis.axis('off')
        cluster_handles = [
            Line2D([0], [0], color=cluster_colors[int(cluster)], linewidth=2,
                   label=f'Cluster {cluster}')
            for cluster in cluster_ids
        ]
        cluster_legend_axis.legend(handles=cluster_handles, loc='center', frameon=False,
                                   ncol=min(3, len(cluster_ids)), fontsize=8)

        gallery_header.text(
            0, 0.5,
            ('Clusters · solid: fixed decoder reconstruction (±2 SD) · inset: latent state / Γ · faint: beats'
             if pca else
             'Clusters · solid: C f_sm (±2 SD) · dashed dark: f_sm (±2√diag Γ) · faint: beats'),
            va='center', fontsize=10, fontweight='bold',
        )
        if not show_latent:
            gallery_header.texts[-1].set_text(
                'Clusters | solid: fitted observation mean (+/-2 SD) | faint: samples by true class')
        if annotations is not None:
            used_labels = sorted(set(annotations), key=lambda label: (ANNOTATION_CODES.get(label, 0), label))
            handles = [Line2D([0], [0], color=annotation_color_for(label), linewidth=2,
                              label=label) for label in used_labels]
            gallery_header.legend(handles=handles, loc='center right', frameon=False,
                                  ncol=min(len(handles), 8), fontsize=9)
            if not ordered and len(handles) > 20:
                gallery_header.get_legend().remove()

        for position in range(len(cluster_ids), gallery_rows * 3):
            fig.add_subplot(layout[position // 3 + 2, position % 3]).axis('off')

        cluster_word = 'cluster' if len(cluster_ids) == 1 else 'clusters'
        headline = (f'Record {record} · channel {channel} · {len(values):,} beats · '
                    f'{len(cluster_ids)} {cluster_word}')
        if not ordered:
            headline = (f'Dataset {record} | output {channel} | {len(values):,} series | '
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
