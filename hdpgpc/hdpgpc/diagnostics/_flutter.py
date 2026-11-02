"""Read-only motion reconstruction and latent-window stability diagnostics."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ._affine_dynamics import transition_radius_samples, affine_dynamics_diagnostics


def flutter_diagnostics(model, windows, values, codes, decoder, center, scale, paths, *, samples, seed):
    labels = np.asarray(model.labels_, dtype=int)
    encoded_reconstruction = decoder.offset+codes @ decoder.C.T
    smoothed = np.full_like(values, np.nan)
    for cluster, path in paths.items():
        indices = path['indices']
        owned = labels[indices] == cluster
        smoothed[indices[owned]] = decoder.offset+path['means'][owned] @ decoder.C.T
    clusters = affine_dynamics_diagnostics(model, paths, windows.times, samples=samples, seed=seed,
        interpretation='AE delay-window dynamics; overlapping observations and per-cluster retained clock; not a mechanical failure certificate')
    available = np.isfinite(smoothed).all(1)
    observed_variance = float(np.sum((values-values.mean(0))**2))
    residual = values-encoded_reconstruction
    end_ae = encoded_reconstruction.reshape(-1, windows.window_samples, 2)[:, -1]*scale+center
    end_smoothed = smoothed.reshape(-1, windows.window_samples, 2)[:, -1]*scale+center
    truth = windows.record.motion[windows.ends]
    result = dict(reference_scope='No sample-wise labels: no ARI, purity or inferred onset accuracy',
        reconstruction_scope='In-sample, common frozen decoder; LDS reconstruction uses smoothed states',
        autoencoder_explained_variance=(1-float(np.sum(residual**2))/observed_variance
                                      if observed_variance > 0 else None),
        autoencoder_scaled_mse=float(np.mean(residual**2)),
        autoencoder_endpoint_rmse=np.sqrt(np.mean((end_ae-truth)**2, axis=0)),
        smoothed_windows=int(available.sum()),
        smoothed_endpoint_rmse=(np.sqrt(np.mean((end_smoothed[available]-truth[available])**2, axis=0))
                                if available.any() else None),
        clusters=clusters)
    return result, dict(autoencoder=end_ae, smoothed=end_smoothed)


def _colors(labels):
    palette = plt.get_cmap('tab20')
    # Use all ten distinct hues before the lighter variants of those hues.
    order = list(range(0, 20, 2)) + list(range(1, 20, 2))
    return {int(cluster): palette(order[i % 20]) for i, cluster in enumerate(np.unique(labels))}


def plot_flutter_summary(windows, labels, reconstructed, diagnostics, *, title):
    figure, axes = plt.subplots(6, 1, figsize=(14, 10),
        gridspec_kw={'height_ratios': [2, 2, .55, .8, .35, 1.8]}, constrained_layout=True)
    end = int(windows.ends[-1])+1
    for channel in range(2):
        axis = axes[channel]
        axis.plot(windows.record.times[:end], windows.record.motion[:end, channel],
                  color='.65', linewidth=.6, label='Observed motion')
        axis.plot(windows.times, reconstructed['autoencoder'][:, channel], color='tab:blue',
                  linewidth=.8, label='AE reconstruction at window end')
        axis.plot(windows.times, reconstructed['smoothed'][:, channel], color='tab:orange',
                  linewidth=.8, linestyle='--', label='Smoothed LDS reconstruction')
        axis.set(xlabel='Recorded time (s)', ylabel=f'Motion channel {channel+1}\n(source units)')
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8, ncol=3)
    axes[2].set_facecolor('#eeeeee')
    axes[2].text(.5, .5, windows.record.reference+' — record-level reference only',
                 ha='center', va='center', transform=axes[2].transAxes)
    axes[2].set(xticks=[], yticks=[], title='Published response description; exact transition times unavailable')
    colors = _colors(labels)
    for cluster, color in colors.items():
        selected = labels == cluster
        axes[3].scatter(windows.times[selected], np.zeros(selected.sum()), marker='|',
                        s=450, color=color, label=f'{cluster} (n={selected.sum()})')
    axes[3].set(xlabel='Window end time (s)', yticks=[], title='Inferred states (local cluster IDs)')
    axes[4].set_axis_off()
    handles, legend = axes[3].get_legend_handles_labels()
    axes[4].legend(handles, legend, ncol=min(len(colors), 6), fontsize=8, loc='center')
    axis = axes[5]
    for row in diagnostics['clusters']:
        cluster = row['cluster']
        axis.scatter(cluster, row['latent_spectral_radius'], marker='x', s=55, color=colors[cluster])
        if 'rho_quantiles' in row:
            lo, median, hi = row['rho_quantiles']
            axis.errorbar(cluster+.08, median, yerr=[[median-lo], [hi-median]],
                          fmt='o', color=colors[cluster], capsize=4)
    axis.axhline(1., linestyle='--', color='.4')
    axis.set(xticks=list(colors), xlabel='Local cluster ID', ylabel='Latent-window spectral radius',
        title='× posterior-mean A; ● variational median and 95% interval; dashed line ρ=1')
    axis.grid(alpha=.2)
    figure.suptitle(f'Flutter {title}, {windows.record.airflow_speed:g} m/s\n'
                   'Raw causal windows → common linear autoencoder → independent recording fit')
    return figure


def plot_flutter_embedding(codes, labels, times, *, title):
    xy = np.column_stack((codes[:, 0], codes[:, 1] if codes.shape[1] > 1 else np.zeros(len(codes))))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for cluster, color in _colors(labels).items():
        chosen = labels == cluster
        axes[0].scatter(*xy[chosen].T, s=6, color=color, alpha=.6, label=str(cluster))
    axes[0].legend(title='Local clusters', fontsize=8)
    axes[0].set_title('Inferred clusters')
    points = axes[1].scatter(*xy.T, s=6, c=times, cmap='viridis')
    figure.colorbar(points, ax=axes[1], label='Recorded window end time (s)')
    axes[1].set_title('Temporal order')
    for axis in axes:
        axis.set(xlabel='AE coordinate 1', ylabel='AE coordinate 2 (0 if r=1)')
        axis.grid(alpha=.2)
    figure.suptitle(title+'\nFirst two common AE coordinates; no extra PCA; other dimensions omitted')
    return figure


def plot_flutter_overview(reports):
    figure, axes = plt.subplots(len(reports), 2, figsize=(14, 2.3*len(reports)),
                               squeeze=False, constrained_layout=True)
    for pair, report in zip(axes, reports):
        path = Path(report['output_directory'])
        with np.load(path/'prepared_data.npz') as saved:
            times, motion, window_times, ends = (saved[key] for key in
                ('source_times', 'source_motion', 'window_times', 'window_ends'))
        with (path/'assignments.csv').open(newline='') as stream:
            labels = np.array([int(row['cluster']) for row in csv.DictReader(stream)])
        pair[0].plot(times, motion[:, 0], linewidth=.6, color='.55')
        for cluster, color in _colors(labels).items():
            chosen = labels == cluster
            pair[0].scatter(window_times[chosen], motion[ends[chosen], 0], color=color, s=4)
            pair[1].scatter(window_times[chosen], np.repeat(cluster, chosen.sum()), color=color, s=5)
        info = report['metadata']
        pair[0].set(title=f'{report["dataset"]}, {info["configuration"]}: {info["reference"]}',
                    ylabel='Motion channel 1\n(source units)', xlabel='Recorded time (s)')
        pair[1].set(title=f'Inferred states: K={report["clusters"]}',
                    xlabel='Window end time (s)', ylabel='Local state ID')
    figure.suptitle('Separate recordings, common unlabeled AE calibration\n'
                   'Cluster IDs are local; record descriptions are not pointwise segmentation truth')
    return figure


def plot_flutter_structure(reports):
    """Compare shared AE coordinates without inventing cross-record state IDs."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    seen, common = set(), None
    for report in reports:
        if report['dataset'] in seen:
            continue
        seen.add(report['dataset'])
        path = Path(report['output_directory'])
        with np.load(path/'autoencoder_calibration.npz') as saved:
            C, offset = saved['C'], saved['offset']
        if common is not None and not np.array_equal(C, common):
            raise ValueError('Cross-record structure plot requires a common decoder')
        common = C
        with np.load(path/'prepared_data.npz') as saved:
            codes = (saved['scaled_windows']-offset) @ np.linalg.pinv(C).T
            times = saved['window_times']
        label = report['dataset'].removeprefix('flutter_')
        line, = axes[1].plot(times, np.linalg.norm(codes, axis=1), linewidth=.8, label=label)
        axes[0].scatter(codes[:, 0], codes[:, 1] if codes.shape[1] > 1 else np.zeros(len(codes)),
                        s=5, alpha=.4, color=line.get_color())
    axes[0].set(title='First two common AE coordinates (other dimensions omitted)',
                xlabel='AE coordinate 1', ylabel='AE coordinate 2')
    axes[1].set(title='Norm of all AE coordinates; amplitudes preserved',
                xlabel='Recorded window end time (s)', ylabel='Latent coordinate norm')
    axes[1].legend(fontsize=8, title='Separate source recordings')
    for axis in axes:
        axis.grid(alpha=.2)
    figure.suptitle('Unsupervised common representation; record identity is display-only\n'
                   'No PCA projection, physical feature engineering or pooled temporal transitions')
    return figure
