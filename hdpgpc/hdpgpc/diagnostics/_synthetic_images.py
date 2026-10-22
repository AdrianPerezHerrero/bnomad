"""Ground-truth plots for the synthetic switching-LDS image experiment."""
import numpy as np


def plot_switching_lds_images(data):
    """Return image, time-series and matrix figures; never display them.

    All image panels use one common intensity scale. Parameter labels and
    latent states are ground truth, not fitted model output.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    palette = ('#377eb8', '#e69f00', '#984ea3')
    vmin, vmax = min(0., data.images.min()), max(1., data.images.max())
    segments = np.unique(data.segment_ids)
    gallery, axes = plt.subplots(len(segments), 7, figsize=(14, 2.05 * len(segments)), squeeze=False)
    for row, segment in enumerate(segments):
        indices = np.flatnonzero(data.segment_ids == segment)
        for column, index in enumerate(np.linspace(indices[0], indices[-1], 7).astype(int)):
            ax = axes[row, column]
            ax.imshow(data.images[index], cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(f't={index}', fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(palette[data.labels[index]])
                spine.set_linewidth(2)
        axes[row, 0].set_ylabel(f'Segment {segment}\n{data.regime_names[data.labels[indices[0]]]}')
    gallery.suptitle(f'Known LDS images | {data.scenario} | {data.state_mode} state evolution\n'
                     + ('Shared wave/eddy structure; subtle gradient differences and overlapping variability'
                        if data.scenario == 'physical_fields' else
                        'Distinct tissue appearances and dynamics; shared affine decoder'
                        if data.scenario == 'appearance_dynamics' else
                        'Same decoder and stationary image distribution; different temporal dynamics'))
    gallery.tight_layout(rect=(0, 0, 1, .95))

    dimension = data.states.shape[1]
    pairs = (dimension+1)//2
    temporal, axes = plt.subplots(pairs+2, 1, figsize=(14, 4+2.3*pairs), sharex=True,
                                  gridspec_kw={'height_ratios': [.45]+[2]*pairs+[1.6]})
    axes[0].imshow(data.labels[None, :], cmap=ListedColormap(palette), vmin=0, vmax=2,
                   aspect='auto', extent=(-.5, len(data.labels)-.5, -.5, .5), interpolation='nearest')
    axes[0].set_yticks([])
    axes[0].set_ylabel('Regime')
    for block in range(pairs):
        for j in range(min(2, dimension-2*block)):
            axes[block+1].plot(data.states[:, 2*block+j], lw=1, label=f'z{2*block+j}')
        axes[block+1].legend(loc='upper right')
        axes[block+1].set_ylabel('True latent state')
    projected = (data.values[:, :, 0] - data.offset) @ np.linalg.pinv(data.C).T
    axes[-1].plot(np.linalg.norm(projected - data.states, axis=1), lw=.8)
    axes[-1].set_ylabel('Oracle encoding\nerror norm')
    axes[-1].set_xlabel('Frame (ordered)')
    boundaries = np.flatnonzero(np.diff(data.segment_ids)) + .5
    for ax in axes[1:]:
        for boundary in boundaries:
            ax.axvline(boundary, color='gray', alpha=.4, lw=.7)
        ax.grid(alpha=.15)
    temporal.suptitle('Ground truth timeline | encoding error reflects only observation noise')
    temporal.tight_layout()

    matrices, axes = plt.subplots(2, 4, figsize=(15, 7))
    for k, name in enumerate(data.regime_names):
        ax = axes[0, k]
        image = ax.imshow(data.A[k], cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_title(f'A[{k}]: {name}\nspectral radius={max(abs(np.linalg.eigvals(data.A[k]))):.3f}')
        matrices.colorbar(image, ax=ax, fraction=.046)
        image = axes[1, k].imshow(data.Gamma[k], cmap='viridis', vmin=0, vmax=data.Gamma.max())
        diagonal = np.diag(data.Gamma[k])
        axes[1, k].set_title(f'Gamma[{k}] | diag range {diagonal.min():.2g}..{diagonal.max():.2g}')
        matrices.colorbar(image, ax=axes[1, k], fraction=.046)
    axes[0, 3].axis('off')
    axes[0, 3].text(0, .95, f'Shared C: {data.C.shape}\n'
                   f'Shared Sigma: {data.Sigma_diagonal[0]:.5g} I\n'
                   + ('Two appearance random walks\nFour damped oscillator coordinates\n'
                      'Regime-specific initial means\n' if data.scenario != 'dynamics_only' else
                      'Initial covariance: I\nStationary covariance: I\n') +
                   'Observation: offset + C z + noise\nNo clipping in saved floating-point data',
                   transform=axes[0, 3].transAxes, va='top', fontsize=10)
    combined = np.concatenate([data.C[:, j].reshape(data.images.shape[1:]) for j in range(dimension)], axis=1)
    axes[1, 3].imshow(combined, cmap='coolwarm', vmin=-abs(data.C).max(), vmax=abs(data.C).max())
    axes[1, 3].set_title(f'{dimension} shared spatial basis patterns')
    axes[1, 3].axis('off')
    matrices.suptitle('Known generative parameters (not fitted)')
    matrices.tight_layout()
    figures = {'images': gallery, 'trajectories': temporal, 'parameters': matrices}
    if data.scenario != 'dynamics_only':
        prototypes, axes = plt.subplots(1, len(data.regime_names), figsize=(10, 3.7), constrained_layout=True)
        for k, (ax, name) in enumerate(zip(np.atleast_1d(axes), data.regime_names)):
            ax.imshow(data.regime_mean_images[k], cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(name)
            ax.set_xticks([])
            ax.set_yticks([])
        prototypes.suptitle('Nominal regime appearances | actual frames add drift and oscillations')
        figures['regime_patterns'] = prototypes
    return figures


def plot_switching_lds_recovery(data, source_indices, assignments, inferred_states):
    """Compare inferred regimes/states to truth, always in the fit row order.

    inferred_states must already be projected into the known latent coordinate
    system. Majority labels serve evaluation only; colors never guide inference.
    """
    import matplotlib.pyplot as plt

    indices, assignments = np.asarray(source_indices), np.asarray(assignments, dtype=int)
    truth = data.labels[indices]
    times = np.arange(len(indices))
    palette = plt.get_cmap('tab10')
    majority = np.empty(len(indices), dtype=int)
    for cluster in np.unique(assignments):
        selected = assignments == cluster
        majority[selected] = np.bincount(truth[selected], minlength=len(data.regime_names)).argmax()
    dimension = data.states.shape[1]
    fig, axes = plt.subplots(3 + dimension, 1, sharex=True,
        figsize=(14, 3.6 + 1.65 * dimension), constrained_layout=True,
        gridspec_kw={'height_ratios': [.65, 1., .65] + [1.5] * dimension})
    for ax, values, name in ((axes[0], truth, 'True regime'),
                              (axes[1], assignments, 'Inferred cluster'),
                              (axes[2], majority, 'Cluster majority')):
        ax.scatter(times, values, c=[palette(int(k) % 10) for k in values], s=7)
        ticks = np.unique(values)
        ax.set_yticks(ticks, [data.regime_names[int(k)] for k in ticks] if ax is not axes[1] else ticks)
        ax.set_ylabel(name)
        ax.grid(alpha=.15)
    for component, ax in enumerate(axes[3:]):
        ax.plot(times, data.states[indices, component], color='black', lw=1.1, label='True state')
        ax.plot(times, inferred_states[:, component], color='tab:orange', lw=.8, alpha=.8,
                label='Smoothed state in true coordinates')
        ax.set_ylabel(f'z{component}')
        ax.grid(alpha=.15)
    axes[3].legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel('Fit row (source frame index is saved in assignments.csv)')
    fig.suptitle('Synthetic LDS recovery | truth and smoothed in-sample posterior\n'
                 'Majority labels are evaluation only; predicted clusters need not equal the true regimes')
    return fig
