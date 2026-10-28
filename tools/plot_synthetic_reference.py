"""Reproduce the README figure using the checked, packaged offline experiment.

From an installed package plus this source checkout:
    python tools/plot_synthetic_reference.py --output docs/assets/synthetic_reference.svg
Use a new --report path to retain configuration, data hash and environment.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('docs/assets/synthetic_reference.svg'))
    parser.add_argument('--report', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    for path in (args.output, args.report):
        if path is not None and path.exists() and not args.overwrite:
            parser.error(f'{path} exists; use a new path or explicitly pass --overwrite')

    from hdpgpc.examples import configuration, waveforms
    from hdpgpc.examples.reference import check_reference, run_experiment
    report = run_experiment('offline')
    check_reference(report)
    report['reference_check'] = 'passed'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    matplotlib.rcParams.update({'svg.hashsalt': 'hdpgpc-synthetic-reference-v2',
                                'font.family': 'DejaVu Sans', 'font.size': 10})
    result = report['scientific_result']
    X = waveforms(1)[..., 0]
    grid = np.asarray(configuration('offline').x_basis).ravel()
    labels = np.asarray(result['labels'])
    colors = ('#0072B2', '#D55E00')
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), layout='constrained')
    for i, wave in enumerate(X):
        axes[0].plot(grid, wave, 'o-', color=colors[labels[i]], alpha=.65, ms=3)
    for cluster in result['clusters']:
        k = cluster['id']
        mean = np.asarray(cluster['mean']).reshape(len(grid), -1)[:, 0]
        covariance = np.asarray(cluster['covariance'])
        if covariance.ndim == 3:
            covariance = covariance[0]
        sd = np.sqrt(np.clip(np.diag(covariance), 0, None))
        axes[1].plot(grid, mean, 'o-', color=colors[k], label=f'Cluster {k}', ms=4)
        axes[1].fill_between(grid, mean-2*sd, mean+2*sd, color=colors[k], alpha=.18)
    axes[1].legend(frameon=False)
    for k in range(result['n_clusters']):
        indices = np.flatnonzero(labels == k)
        axes[2].scatter(indices+1, labels[indices], color=colors[k], s=65)
    axes[0].set(title='Six input waveforms', xlabel='Within-waveform coordinate', ylabel='Amplitude')
    axes[1].set(title='Fitted observation means ±2 SD', xlabel='Within-waveform coordinate', ylabel='Amplitude')
    axes[2].set(title='Inferred groups in sequence order', xlabel='Observation number', ylabel='Cluster',
                xticks=np.arange(1, 7), yticks=[0, 1], ylim=(-.35, 1.35))
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.18)
    fig.suptitle('HDPGPC · reproducible synthetic example', fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # SVG timestamps are omitted and IDs use a fixed salt for repeatable output.
    metadata = {'Date': None} if args.output.suffix.lower() == '.svg' else {}
    fig.savefig(args.output, metadata=metadata, dpi=180)
    plt.close(fig)
    report['figure'] = {'matplotlib': matplotlib.__version__,
                       'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       'image_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f'Saved {args.output}; labels={labels.tolist()}; reference check passed')


if __name__ == '__main__':
    main()
