"""Order-independent summaries of encoded observations and fitted clusters."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def plot_embedding_summary(codes, assignments, labels, metrics, *, title=None,
                           projection_name='Autoencoder'):
    """Plot every sample in a PCA view, assignments, and class composition.

    The display coordinates are the first two principal components of the
    supplied latent codes. Labels are used for display only. Cluster legends
    have a separate wrapping row; extra entries increase the figure height
    rather than taking space from the three data panels.
    """
    codes = np.asarray(codes, dtype=np.float64)
    assignments = np.asarray(assignments).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if (codes.ndim != 2 or codes.shape[1] < 1 or len(codes) < 2
            or len(assignments) != len(codes) or len(labels) != len(codes)
            or not np.isfinite(codes).all()):
        raise ValueError('Codes must be finite rows with one cluster and label per row')

    if codes.shape[1] > 1 and np.var(codes, axis=0).sum() > 1e-15:
        projection = PCA(n_components=2).fit(codes)
        xy = projection.transform(codes)
        variance = np.nan_to_num(projection.explained_variance_ratio_)
    elif codes.shape[1] > 1:
        xy = np.zeros((len(codes), 2))
        variance = (0.0, 0.0)
    else:
        xy = np.column_stack((codes[:, 0], np.zeros(len(codes))))
        variance = (1.0 if np.var(codes[:, 0]) > 1e-15 else 0.0, 0.0)
    clusters, cluster_index = np.unique(assignments, return_inverse=True)
    classes, class_index = np.unique(labels, return_inverse=True)
    contingency = np.zeros((len(classes), len(clusters)), dtype=np.int64)
    np.add.at(contingency, (class_index, cluster_index), 1)
    proportions = contingency / np.maximum(contingency.sum(axis=0, keepdims=True), 1)

    legend_columns = min(6, len(clusters))
    legend_rows = (len(clusters) + legend_columns - 1) // legend_columns
    legend_height = 0.45 + 0.28 * legend_rows
    figure = plt.figure(figsize=(17, 5 + legend_height), constrained_layout=True)
    layout = figure.add_gridspec(2, 3, width_ratios=(1, 1, 1.2),
                                height_ratios=(4.2, legend_height))
    axes = [figure.add_subplot(layout[0, column]) for column in range(3)]
    legend_axis = figure.add_subplot(layout[1, :])
    legend_axis.set_axis_off()
    cluster_cmap = plt.get_cmap('tab20', max(len(clusters), 2))
    for index, cluster in enumerate(clusters):
        chosen = cluster_index == index
        axes[0].scatter(xy[chosen, 0], xy[chosen, 1], s=7, alpha=0.55,
                        color=cluster_cmap(index), rasterized=True,
                        label=f'{cluster} (n={chosen.sum()})')
    handles, legend_labels = axes[0].get_legend_handles_labels()
    legend_axis.legend(handles, legend_labels, title='Clusters (sample counts)',
                       ncol=legend_columns, fontsize=8, loc='center',
                       markerscale=2, frameon=False, columnspacing=2,
                       handletextpad=0.5, borderaxespad=0)
    axes[0].set_title(f'{projection_name} projection · fitted clusters')

    label_cmap = plt.get_cmap('tab20' if len(classes) <= 20 else 'turbo',
                             max(len(classes), 2))
    scatter = axes[1].scatter(xy[:, 0], xy[:, 1], c=class_index, cmap=label_cmap,
                              vmin=-0.5, vmax=len(classes) - 0.5,
                              s=7, alpha=0.55, rasterized=True)
    axes[1].set_title(f'{projection_name} projection · true labels ({len(classes)})')
    if len(classes) <= 20:
        colorbar = figure.colorbar(scatter, ax=axes[1], ticks=np.arange(len(classes)),
                                   fraction=0.046, pad=0.02)
        colorbar.ax.set_yticklabels([str(label) for label in classes])
    coordinate_name = 'Latent' if projection_name == 'Autoencoder' else projection_name
    for axis in axes[:2]:
        axis.set_xlabel(f'{coordinate_name} PC1 ({variance[0]:.1%} variance)')
        axis.set_ylabel(f'{coordinate_name} PC2 ({variance[1]:.1%} variance)')
        axis.grid(alpha=0.15)

    image = axes[2].imshow(proportions, aspect='auto', vmin=0, vmax=1,
                           cmap='Blues', interpolation='nearest')
    figure.colorbar(image, ax=axes[2], label='Fraction of fitted cluster')
    axes[2].set_title('Class composition within each cluster')
    axes[2].set_xlabel('Fitted cluster')
    axes[2].set_ylabel('True class')
    axes[2].set_xticks(np.arange(len(clusters)),
                       [f'{cluster}\n(n={contingency[:, index].sum()})'
                        for index, cluster in enumerate(clusters)])
    if len(clusters) > 10:
        axes[2].tick_params(axis='x', labelrotation=90, labelsize=7)
    if len(classes) <= 30:
        axes[2].set_yticks(np.arange(len(classes)), [str(label) for label in classes])
    else:
        shown = np.unique(np.linspace(0, len(classes) - 1, 20, dtype=int))
        axes[2].set_yticks(shown, [str(classes[index]) for index in shown])
        axes[2].set_ylabel(f'True class ({len(classes)} classes; selected ticks)')

    metric_text = (f"n={len(codes):,}   clusters={len(clusters)}   "
                   f"purity={metrics['purity']:.3f}   "
                   f"ARI={metrics['adjusted_rand_index']:.3f}   "
                   f"NMI={metrics['normalized_mutual_information']:.3f}")
    figure.suptitle(f'{title or "Encoded clustering"}\n{metric_text}', fontsize=12)
    return figure
