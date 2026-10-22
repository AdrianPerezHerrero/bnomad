"""Image diagnostics for HDPMFA, using its public factor-analysis API."""
from __future__ import annotations

import numpy as np

from ._reporting import cluster_label_statistics


def _label_results(model, annotations, evaluation_mask):
    labels = model.labels_
    annotations = np.asarray(annotations)
    mask = np.ones(len(labels), dtype=bool) if evaluation_mask is None else np.asarray(evaluation_mask, dtype=bool)
    if annotations.shape != labels.shape or mask.shape != labels.shape:
        raise ValueError('Annotations and evaluation mask must align with training assignments')
    if mask.any():
        scores, _, _, mapping = cluster_label_statistics(labels[mask], annotations[mask])
    else:
        scores, mapping = {}, {}
    return labels, annotations, mask, scores, mapping


def plot_mfa_image_summary(model, observations, annotations, *, times=None,
                           evaluation_mask=None, title='MFA image clustering', random_state=42):
    """Regimes, labels, ELBO, a common display projection, and Markov weights.

    The display PCA is fitted to raw image pixels, never to component-local
    factor coordinates, and never changes the fitted model or its objective.
    All assignments are shown even when detail galleries hide small clusters.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    labels, annotations, mask, scores, mapping = _label_results(model, annotations, evaluation_mask)
    values = np.asarray(observations)
    time_label = 'Frame order' if times is None else 'Acquisition time (seconds)'
    times = np.arange(len(labels)) if times is None else np.asarray(times)
    if (values.shape != (len(labels), model.n_features_in_) or len(values) < 2
            or times.shape != labels.shape or not np.isfinite(values).all() or not np.isfinite(times).all()):
        raise ValueError('Provide at least two aligned training image vectors and times')
    ids = model.cluster_ids_
    colors = plt.get_cmap('tab20', max(1, len(ids)))
    cluster_color = {int(cluster): colors(i) for i, cluster in enumerate(ids)}
    classes = np.unique(annotations[mask])
    class_color = {label: plt.get_cmap('tab20')(i % 20) for i, label in enumerate(classes)}
    display_labels = list(classes) + ['ambiguous/unlabeled']
    lookup = {label: index for index, label in enumerate(classes)}
    true_ids = [lookup.get(label, len(classes)) if valid else len(classes)
                for label, valid in zip(annotations, mask)]
    majority = [mapping.get(int(cluster), 'ambiguous/unlabeled') for cluster in labels]
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), layout='constrained')
    fig.suptitle(f'{title} | {len(labels)} frames | {model.n_clusters_} MAP clusters'
        + (f'\nPurity {scores["purity"]:.3f} | ARI {scores["adjusted_rand_index"]:.3f} | '
           f'NMI {scores["normalized_mutual_information"]:.3f}' if scores else '\nNo unambiguous evaluation labels'))

    axes[0, 0].scatter(times, labels, c=[cluster_color[int(k)] for k in labels], s=9)
    axes[0, 0].set(yticks=ids, ylabel='Cluster', title='Inferred regimes (stored MAP assignments)')
    axes[0, 1].scatter(times, true_ids, c=[class_color.get(label, '.6') if valid else '.6'
        for label, valid in zip(annotations, mask)], s=9)
    axes[0, 1].set(yticks=np.arange(len(display_labels)), yticklabels=display_labels,
                   title='Reference anatomy (evaluation only)')
    axes[1, 0].scatter(times, labels, c=[class_color.get(label, '.6') for label in majority], s=9)
    axes[1, 0].set(yticks=ids, ylabel='Cluster',
                   title='Regimes colored by in-sample majority anatomy')
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xlabel(time_label)
        axis.grid(alpha=.15)
    axes[1, 1].plot(model.elbo_history_)
    axes[1, 1].set(title='Variational objective (including parameter KLs)',
                   xlabel='Completed coordinate sweep (0 = initialization)', ylabel='ELBO')
    projection = PCA(n_components=min(2, values.shape[1], len(values)),
        svd_solver='randomized', random_state=random_state).fit_transform(values)
    if projection.shape[1] == 1:
        projection = np.column_stack((projection[:, 0], np.zeros(len(values))))
    axes[2, 0].scatter(projection[:, 0], projection[:, 1],
        c=[cluster_color[int(k)] for k in labels], s=10, alpha=.7)
    axes[2, 0].set(title='Common pixel PCA display (not MFA latent coordinates)',
                   xlabel='Display PC 1', ylabel='Display PC 2')
    if model.config.hdp.hmm_switch:
        image = axes[2, 1].imshow(model.transition_matrix_, vmin=0, vmax=1, cmap='Blues', aspect='auto')
        axes[2, 1].set(title='Posterior transitions (represented support)',
            xticks=model.component_ids_, yticks=model.component_ids_, xlabel='To component', ylabel='From component')
        fig.colorbar(image, ax=axes[2, 1], label='Transition probability', shrink=.8)
    else:
        axes[2, 1].bar(model.component_ids_, model.occupancies_)
        axes[2, 1].set(title='Independent mixture (no transition conditioning)',
                       xlabel='Component', ylabel='Sum of soft responsibilities')
    return fig


def plot_mfa_image_gallery(model, images, annotations, *, factor_means=None,
                           evaluation_mask=None, frames_per_cluster=4,
                           min_cluster_samples=1, random_state=42):
    """Per-cluster mean/SD and paired observed/reconstructed training frames.

    Reconstructions are mu_k + W_k E[z_i] using the stored training MAP cluster.
    SD images show new-frame marginal pixel variability, integrating q(B), the
    standard-normal factor prior and shared pixel noise; they are not confidence
    intervals for the component mean. Pixels in image panels share [0,1] limits;
    values are clipped only for display, never in model calculations or saves.
    """
    import matplotlib.pyplot as plt

    labels, annotations, _, _, mapping = _label_results(model, annotations, evaluation_mask)
    images = np.asarray(images)
    if (images.ndim != 3 or len(images) != len(labels)
            or int(np.prod(images.shape[1:])) != model.n_features_in_
            or not np.isfinite(images).all()):
        raise ValueError('images must have shape (training frames, height, width)')
    if (not isinstance(frames_per_cluster, int) or frames_per_cluster < 1
            or not isinstance(min_cluster_samples, int) or min_cluster_samples < 1):
        raise ValueError('Gallery sizes must be positive integers')
    if factor_means is None:
        factors = model.transform_components(images.reshape(len(images), -1))
        factor_means = factors.mean[np.arange(len(images)), labels]
    factor_means = np.asarray(factor_means)
    if factor_means.shape != (len(images), model.n_factors_) or not np.isfinite(factor_means).all():
        raise ValueError('factor_means must be stored-MAP component coordinates, shape (N,r)')
    ids = [int(k) for k in model.cluster_ids_ if np.sum(labels == k) >= min_cluster_samples]
    if not ids:
        fig, axis = plt.subplots(figsize=(10, 3))
        axis.text(.5, .5, 'No cluster meets the image-gallery sample threshold', ha='center', va='center')
        axis.set_axis_off()
        return fig
    means, loadings = model.means_, model.loadings_
    components = model.components_
    count = min(frames_per_cluster, max(int(np.sum(labels == k)) for k in ids))
    fig, axes = plt.subplots(2 * len(ids), count + 1, squeeze=False,
        figsize=(2.7 * (count + 1), 5 * len(ids)), layout='constrained')
    fig.suptitle('MFA images | top: observations; bottom: posterior-factor reconstructions\n'
                 'Left column: component mean / marginal pixel SD; images displayed in [0,1]')
    rng = np.random.default_rng(random_state)
    for group, cluster in enumerate(ids):
        members = np.flatnonzero(labels == cluster)
        chosen = rng.choice(members, size=min(count, len(members)), replace=False)
        row = 2 * group
        axes[row, 0].imshow(means[cluster].reshape(images.shape[1:]), cmap='gray', vmin=0, vmax=1)
        axes[row, 0].set_title(f'Cluster {cluster}: {len(members)} frames\nMajority: {mapping.get(cluster, "unlabeled")}\nMean image')
        sd = np.sqrt(components[cluster].predictive_moments().variance.cpu().numpy())
        image = axes[row + 1, 0].imshow(sd.reshape(images.shape[1:]), cmap='magma', vmin=0)
        axes[row + 1, 0].set_title('Marginal pixel SD')
        fig.colorbar(image, ax=axes[row + 1, 0], shrink=.7)
        for column, index in enumerate(chosen, 1):
            axes[row, column].imshow(images[index], cmap='gray', vmin=0, vmax=1)
            axes[row, column].set_title(f'Frame row {index}\n{annotations[index]}')
            reconstruction = means[cluster] + loadings[cluster] @ factor_means[index]
            axes[row + 1, column].imshow(reconstruction.reshape(images.shape[1:]), cmap='gray', vmin=0, vmax=1)
            axes[row + 1, column].set_title('MFA reconstruction')
        for axis in axes[row:row + 2].flat:
            axis.set_axis_off()
    return fig
