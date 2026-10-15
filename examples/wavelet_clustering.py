"""Cluster UCR ACSF1 with a real fixed Haar wavelet observation decoder.

python examples/wavelet_clustering.py --schedule offline --points 64
python examples/wavelet_clustering.py --schedule online_fast --sigma-mode cluster_diagonal

All ACSF1 TRAIN and TEST series are fitted by default. TRAIN data alone define
waveform normalization. Labels are used only for descriptive clustering scores.
UCR rows have no temporal order: HMM transitions are disabled and A is fixed
to zero, while Gamma learns per cluster in Haar wavelet coordinates. No wavelet
preprocessing is applied to the waveforms supplied to HDPGPC.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from hdpgpc import HDPGPC
from hdpgpc.config import MatrixPolicyConfig, NoisePolicyConfig, with_fixed_observation_model
from hdpgpc.emissions import make_wavelet_observation
from hdpgpc.examples import configuration
from hdpgpc.diagnostics import cluster_label_statistics
from tutorial_utils import plot_record_results
from ucr_dtw_gp_archive import DEFAULT_ARCHIVE, archive_datasets, download_archive, prepare_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument('--download', action='store_true', help='Download the UCR archive if absent.')
    parser.add_argument('--dataset', default='ACSF1', help='UCR dataset name (default ACSF1).')
    parser.add_argument('--schedule', choices=('offline', 'online', 'online_fast'), default='offline')
    parser.add_argument('--points', type=int, default=64,
                        help='Power-of-two waveform resolution (default 64); all Haar coefficients are retained.')
    parser.add_argument('--max-series', type=int, help='Optional random subset; default uses every series.')
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--max-clusters', type=int, default=30)
    parser.add_argument('--hdp-preset', choices=('less', 'balanced', 'more', 'max'), default='more')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', choices=('silent', 'progress', 'debug'), default='debug')
    parser.add_argument('--sigma-mode', default='cluster_diagonal',
                        choices=('fixed', 'cluster', 'cluster_diagonal', 'shared', 'shared_diagonal'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'agents' / 'wavelet_example')
    args = parser.parse_args()
    if (args.points < 8 or args.points & (args.points - 1)
            or args.iterations < 1 or args.max_clusters < 1
            or (args.max_series is not None and args.max_series < 2)):
        parser.error('Points must be a power of two >= 8; sizes and iterations must be positive')
    if args.download:
        download_archive(args.archive)
    if not args.archive.is_file():
        parser.error(f'Archive not found: {args.archive}; use --download')
    with ZipFile(args.archive) as archive:
        pairs = archive_datasets(archive)
        if args.dataset not in pairs:
            parser.error(f'Unknown UCR dataset: {args.dataset}')
        data = prepare_dataset(archive, pairs[args.dataset], args.points)
    waveforms = np.concatenate((data.train, data.test))
    labels = np.concatenate((data.train_labels, data.test_labels))
    rng = np.random.default_rng(args.seed)
    selected = (rng.choice(len(waveforms), args.max_series, replace=False)
                if args.max_series is not None and args.max_series < len(waveforms)
                else np.arange(len(waveforms)))
    selected = rng.permutation(selected)
    values, labels = waveforms[selected, :, None], labels[selected]
    points = values.shape[1]
    if points & (points - 1):
        parser.error(f'Loader returned {points} points; use a power-of-two --points no larger than the series length')
    grid = np.linspace(0.0, 1.0, points)[:, None]
    print(f'UCR {args.dataset}: TRAIN={len(data.train)}, TEST={len(data.test)}, '
          f'fitted={len(values)}, classes={len(np.unique(labels))}, points={points}')
    decoder = make_wavelet_observation(
        grid, Sigma=0.5, latent_covariance=0.5)
    base = configuration('offline')
    base = replace(
        base, verbose=args.verbose,
        hdp=replace(base.hdp, max_models=args.max_clusters,
                    hdp_hyp=args.hdp_preset, hmm_switch=False),
        inference=replace(base.inference, schedule=args.schedule, warp=False,
                          it_limit=args.iterations if args.schedule == 'offline' else None,
                          n_explore_steps=10, hard_assignments=False),
        dynamics=replace(base.dynamics, lds_update='conjugate_batch'
                         if args.schedule == 'offline' else 'conjugate_sequential'),
        numerics=replace(base.numerics, seed=args.seed, num_threads=1),
    )
    dimension = len(grid)
    identity = np.eye(dimension)
    config = with_fixed_observation_model(
        base, decoder, sigma_policy=NoisePolicyConfig(mode=args.sigma_mode, value=0.01 * identity))
    # Independent UCR series: A=0 removes sequential prediction. Gamma learns
    # Haar wavelet-state covariance; Sigma stays in waveform coordinates.
    config = replace(config, dynamics=replace(
        config.dynamics, A_policy=MatrixPolicyConfig(
            mode='fixed', value=np.zeros((decoder.latent_dim, decoder.latent_dim)))))
    model = HDPGPC(config).fit(values, grid=grid)
    metrics, _, _, _ = cluster_label_statistics(model.labels_, labels)
    print(f'Haar wavelet: {points} waveform points, {decoder.latent_dim} real coordinates; '
          f'clusters={model.n_clusters_}, purity={metrics["purity"]:.3f}, '
          f'ARI={metrics["adjusted_rand_index"]:.3f}, '
          f'NMI={metrics["normalized_mutual_information"]:.3f}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'config.yaml').write_text(config.to_yaml(), encoding='utf-8')
    report = {
        'dataset': args.dataset, 'archive': str(args.archive.resolve()),
        'train_series': len(data.train), 'test_series': len(data.test),
        'fitted_series': len(values), 'raw_points': data.raw_points,
        'waveform_points': points, 'latent_dimension': decoder.latent_dim, 'wavelet': 'haar', 'full_basis': True,
        'seed': args.seed, 'combined_archive_indices': selected.tolist(),
        'true_labels': labels.tolist(), 'assignments': model.labels_.tolist(),
        'metrics': {key: float(value) for key, value in metrics.items()},
        'evaluation': 'transductive: TRAIN and TEST both participate in clustering',
    }
    (args.output_dir / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    figures = plot_record_results(model, values, labels, grid=grid,
                                  record=f'UCR {args.dataset} Haar wavelet', metrics=metrics,
                                  ordered=False, show_latent=False)
    for channel, figure in enumerate(figures):
        path = args.output_dir / f'{args.dataset}_wavelet_clusters_output_{channel}.png'
        try:
            figure.savefig(path, dpi=160)
        finally:
            plt.close(figure)
        print(f'Saved {path}')


if __name__ == '__main__':
    main()
