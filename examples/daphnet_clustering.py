"""Naive chronological Daphnet clustering with the ordinary HDP-GPC model.

python examples/daphnet_clustering.py --download --records S02R01
python examples/daphnet_clustering.py --records S02R01 --schedule online_fast
python examples/daphnet_clustering.py --records S02R01 --warp
python examples/daphnet_clustering.py --records S02R01 --normalize-samples
python examples/daphnet_clustering.py --records all --segments all --no-fit-gp

No encoder, fixed-decoder adapter or nonlinear features are used. Each window
is a waveform observation; A/Gamma learn its latent representer's evolution.
Change model policies in make_config or use --lds-preset. Label zero excludes
non-experimental intervals. Labels 1/2 only enter descriptive evaluation.
"""

import argparse
from contextlib import redirect_stdout, redirect_stderr
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import confusion_matrix

from hdpgpc import HDPGPC
from hdpgpc.config import (AlignmentConfig, DynamicsConfig, GPEmissionConfig,
                          HDPConfig, HDPGPCConfig, InferenceConfig, LDS_PRESETS,
                          MultiOutputConfig, NumericalConfig, with_lds_preset)
from hdpgpc.datasets import (DAPHNET_CHANNELS, daphnet_records, download_daphnet,
                             load_daphnet_record, window_daphnet_record)
from hdpgpc.diagnostics import (Tee, cluster_label_statistics, format_confusion_matrix,
                                format_lds_priors, model_lds_arrays, model_lds_dynamics,
                                plot_lds_diagnostics, plot_lds_dynamics)
from hdpgpc.preprocessing import normalize_samples, prepare_sequences
from tutorial_utils import plot_record_results


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / 'agents' / 'data' / 'daphnet' / 'daphnet.zip'
DEFAULT_OUTPUT = ROOT / 'agents' / 'daphnet_results'


def make_config(grid, args):
    """The experiment's model API/configuration: replace here for later AE runs."""
    update = args.lds_update or ('conjugate_batch' if args.schedule == 'offline'
                                else 'conjugate_sequential')
    if update == 'conjugate_batch' and args.schedule != 'offline':
        raise ValueError('conjugate_batch requires --schedule offline')
    kernel = (ConstantKernel(1., (1e-3, 1e3))
              * RBF(args.lengthscale, (0.02, 2.0))
              + WhiteKernel(args.gp_noise, (1e-5, 1.0)))
    base = HDPGPCConfig(
        x_basis=grid, verbose=args.verbose,
        hdp=HDPConfig(hdp_hyp=args.hdp_preset, hmm_switch=True,
                      max_models=args.max_clusters),
        emission=GPEmissionConfig(
            kernels=[kernel], fit_gp=args.fit_gp,
            ini_lengthscale=args.lengthscale, bound_lengthscale=(0.02, 2.0),
            ini_sigma=args.sigma_prior, bound_sigma=(1e-6, 1e3)),
        dynamics=DynamicsConfig(
            model_type='dynamic', ini_gamma=args.gamma_prior,
            bound_gamma=(1e-6, 1e3), annealing=False,
            bayesian_params=True, free_deg_MNIV=args.lds_degrees,
            lds_update=update, lds_scoring='final', lds_step_size=args.lds_step_size),
        alignment=AlignmentConfig(
            policy='dtw_gp' if args.warp else 'none',
            x_basis_warp=grid if args.warp else None),
        inference=InferenceConfig(
            schedule=args.schedule, warp=args.warp, hard_assignments=args.hard_assignments,
            birth_lds_kl_per_dimension=True, n_explore_steps=args.exploration_steps,
            it_limit=args.iterations if args.schedule == 'offline' else None),
        multi_output=MultiOutputConfig(n_outputs=1, share_gp=True, use_snr=False),
        numerics=NumericalConfig(seed=args.seed, num_threads=1,
                                 dtype='float64', deterministic=True),
    )
    return with_lds_preset(base, args.lds_preset)


def run_segment(record, windows, args):
    destination = args.output_dir / record.name / f'segment_{windows.segment:02d}'
    destination.mkdir(parents=True, exist_ok=True)
    stop = min(len(windows.values), args.max_windows or len(windows.values))
    raw = windows.values[:stop]
    if len(raw) < 2:
        raise ValueError('A clustering segment must contain at least two complete windows')
    if args.normalize_samples:
        values, centers, scales = normalize_samples(raw)
        np.savez_compressed(destination / 'sample_normalization.npz',
                            centers_g=centers, scales_g=scales)
        center = scale = None
        scaling = 'per_window'
    else:
        values, _ = prepare_sequences(raw, channels=(0,),
                                       scale_from_first=args.scaling == 'first_window')
        # This fixed transform uses only the first observation, also online.
        center = float(raw[0].mean()) if args.scaling == 'first_window' else 0.
        scale = float(raw[0].std()) if args.scaling == 'first_window' else 1.
        scaling = args.scaling
    annotations = windows.annotations[:stop]
    fractions = windows.freeze_fraction[:stop]
    grid = windows.grid
    config = make_config(grid, args)
    (destination / 'config.yaml').write_text(config.to_yaml(), encoding='utf-8')
    print(f'{record.name}, segment {windows.segment}: {len(values)} chronological windows, '
          f'{args.channel}, {args.window_seconds:g} s, {len(grid)} points, '
          f'schedule={args.schedule}, warp={args.warp}, '
          f'alignment={config.alignment.policy}, LDS={args.lds_preset}, '
          f'normalization={scaling}')
    print(f'Window annotations: non-freezing={int((annotations == 1).sum())}, '
          f'freezing={int((annotations == 2).sum())}, '
          f'mixed={int(((fractions > 0) & (fractions < 1)).sum())}')
    if args.verbose != 'silent':
        print(format_lds_priors(
            dimension=len(grid), model_type=config.dynamics.model_type,
            observation_noise=args.sigma_prior, process_noise=args.gamma_prior,
            degrees_of_freedom=args.lds_degrees, source='explicit experiment scales',
            policies={'A': config.dynamics.A_policy, 'Gamma': config.dynamics.Gamma_policy,
                      'C': config.observation.C_policy, 'Sigma': config.observation.Sigma_policy}))
    # Standard model call. No class labels are passed to fit.
    model = HDPGPC(config).fit(values, grid=grid)
    scores, _, _, cluster_labels = cluster_label_statistics(model.labels_, annotations)
    assigned = np.array([cluster_labels[int(cluster)] for cluster in model.labels_])
    confusion = confusion_matrix(annotations, assigned, labels=[1, 2])
    tp, fp, fn = int(confusion[1, 1]), int(confusion[0, 1]), int(confusion[1, 0])
    scores['freezing_precision_in_sample'] = tp / (tp + fp) if tp + fp else 0.
    scores['freezing_recall_in_sample'] = tp / (tp + fn) if tp + fn else None
    scores['freezing_f1_in_sample'] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    print(f'Clusters={model.n_clusters_}, purity={scores["purity"]:.3f}, '
          f'ARI={scores["adjusted_rand_index"]:.3f}, '
          f'NMI={scores["normalized_mutual_information"]:.3f}')
    if model.n_clusters_ >= args.max_clusters:
        print('Configured cluster cap reached; this is a limit exit, not a convergence claim.')
    print(format_confusion_matrix([1, 2], confusion))
    print('1 = experimental non-freezing; 2 = freezing. '
          'Cluster-majority mapping and scores are in-sample descriptions.')

    report = {
        'record': record.name, 'segment': windows.segment, 'channel': args.channel,
        'window_seconds': args.window_seconds,
        'warp': args.warp, 'alignment': config.alignment.policy,
        'stride_seconds': args.stride_seconds or args.window_seconds,
        'source_sampling_hz': 64, 'waveform_points': len(grid),
        'scaling': scaling, 'normalize_samples': args.normalize_samples,
        'requested_scaling': args.scaling, 'center_g': center, 'scale_g': scale,
        'fitted_windows': len(values), 'mixed_windows': int(((fractions > 0) & (fractions < 1)).sum()),
        'non_freezing_windows': int((annotations == 1).sum()),
        'freezing_windows': int((annotations == 2).sum()),
        'cluster_cap_reached': model.n_clusters_ >= args.max_clusters,
        'cluster_count': model.n_clusters_, 'metrics': scores,
        'confusion_classes': [1, 2], 'confusion': confusion.tolist(),
        'cluster_majority_labels': cluster_labels,
        'evaluation': 'in-sample window clustering; majority labels evaluated on fitted windows',
        'input': str(args.source.resolve()),
        'archive_sha256': args.archive_sha256,
    }
    (destination / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (destination / 'assignments.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['window', 'source_start_row', 'start_seconds', 'end_seconds',
                         'annotation', 'freeze_fraction', 'cluster', 'cluster_majority_annotation'])
        for index in range(len(values)):
            writer.writerow([index, int(windows.start_rows[index]), windows.start_seconds[index],
                             windows.end_seconds[index], int(annotations[index]), fractions[index],
                             int(model.labels_[index]), int(assigned[index])])
    # Save final matrices for quantitative comparison, independent of plotting.
    matrices = model_lds_arrays(model)
    np.savez_compressed(destination / 'lds_parameters.npz', **{
        f'cluster_{cluster}_{name}': matrix
        for cluster, parameters in matrices.items() for name, matrix in parameters.items()
        if name in ('A', 'C', 'Gamma', 'Sigma')})
    if args.plots:
        figures = plot_record_results(
            model, values, annotations, grid=grid,
            record=f'Daphnet {record.name} segment {windows.segment} | {args.channel}',
            metrics=scores, ordered=True, warped=args.warp,
            annotation_colors={1: 'tab:blue', 2: 'tab:red'})
        for output, figure in enumerate(figures):
            try:
                figure.savefig(destination / f'clusters_output_{output}.png', dpi=150)
            finally:
                plt.close(figure)
        if args.lds_plots:
            for page, figure in enumerate(plot_lds_diagnostics(matrices), start=1):
                try:
                    figure.savefig(destination / f'lds_parameters_{page:02d}.png', dpi=140)
                finally:
                    plt.close(figure)
            figure = plot_lds_dynamics(model_lds_dynamics(model, grid=grid),
                                       title=f'Daphnet {record.name}: projected LDS dynamics')
            try:
                figure.savefig(destination / 'lds_dynamics.png', dpi=140)
            finally:
                plt.close(figure)
    print(f'Saved {destination}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_ARCHIVE,
                        help='Official ZIP or extracted Daphnet directory.')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--list-records', action='store_true')
    parser.add_argument('--records', nargs='+', default=['S02R01'], help='IDs, or all.')
    parser.add_argument('--segments', choices=('longest', 'all'), default='longest')
    parser.add_argument('--channel', choices=DAPHNET_CHANNELS, default='ankle_vertical')
    parser.add_argument('--window-seconds', type=float, default=2.)
    parser.add_argument('--stride-seconds', type=float,
                        help='Default equals window length (non-overlapping).')
    parser.add_argument('--points', type=int, default=64,
                        help='Points per waveform; downsampling uses an anti-alias filter.')
    parser.add_argument('--max-windows', type=int, help='Fit only a chronological prefix, not a random subset.')
    parser.add_argument('--scaling', choices=('first_window', 'none'), default='first_window')
    parser.add_argument('--normalize-samples', action=argparse.BooleanOptionalAction, default=False,
                        help='Z-score each window independently; overrides --scaling (default off).')
    parser.add_argument('--schedule', choices=('offline', 'online', 'online_fast'), default='offline')
    parser.add_argument('--warp', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable existing DTW-GP alignment (default off); --no-warp disables it.')
    parser.add_argument('--iterations', type=int, default=40)
    parser.add_argument('--max-clusters', type=int, default=40)
    parser.add_argument('--hdp-preset', choices=('less', 'balanced', 'more', 'max'), default='more')
    parser.add_argument('--lds-preset', choices=LDS_PRESETS, default='only_plastic',
                        help='Default fixes C=I and Sigma; A/Gamma learn. all_free selects unrestricted LDS.')
    parser.add_argument('--lds-update', choices=('conjugate_batch', 'conjugate_sequential', 'natural_gradient'))
    parser.add_argument('--lds-degrees', type=int, default=20)
    parser.add_argument('--lds-step-size', type=float, default=0.2)
    parser.add_argument('--sigma-prior', type=float, default=0.1)
    parser.add_argument('--gamma-prior', type=float, default=0.1)
    parser.add_argument('--lengthscale', type=float, default=0.15, help='GP lengthscale in seconds.')
    parser.add_argument('--gp-noise', type=float, default=0.01, help='GP white noise, separate from Sigma.')
    parser.add_argument('--fit-gp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--hard-assignments', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--exploration-steps', type=int, default=10)
    parser.add_argument('--verbose', choices=('silent', 'progress', 'debug'), default='debug')
    parser.add_argument('--plots', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--lds-plots', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if (args.iterations < 1 or args.max_clusters < 1 or args.lds_degrees < 3
            or args.exploration_steps < 1 or (args.max_windows is not None and args.max_windows < 2)
            or not 0 < args.lds_step_size <= 1 or not 0.02 <= args.lengthscale <= 2
            or not 1e-5 <= args.gp_noise <= 1 or not 1e-6 <= args.sigma_prior <= 1e3
            or not 1e-6 <= args.gamma_prior <= 1e3):
        parser.error('Invalid size, prior, GP or update controls')
    if args.download:
        if args.source.is_dir():
            parser.error('--download needs a ZIP destination, not an extracted directory')
        download_daphnet(args.source)
    if not args.source.exists():
        parser.error(f'Daphnet source not found: {args.source}; use --download')
    available = daphnet_records(args.source)
    if args.list_records:
        print(' '.join(available))
        return
    records = available if args.records == ['all'] else [name.upper().removesuffix('.TXT') for name in args.records]
    if any(name not in available for name in records):
        parser.error(f'Unknown record; available: {available}')
    args.archive_sha256 = (hashlib.sha256(args.source.read_bytes()).hexdigest()
                           if args.source.is_file() else None)
    summaries = []
    for name in dict.fromkeys(records):
        record = load_daphnet_record(args.source, name)
        segments = window_daphnet_record(
            record, channel=args.channel, window_seconds=args.window_seconds,
            stride_seconds=args.stride_seconds, points=args.points)
        segments = [segment for segment in segments if len(segment.values) >= 2]
        if not segments:
            raise ValueError(f'{name}: no experimental segment contains two complete windows')
        if args.segments == 'longest':
            segments = [max(segments, key=lambda segment: len(segment.values))]
        for segment in segments:
            destination = args.output_dir / name / f'segment_{segment.segment:02d}'
            destination.mkdir(parents=True, exist_ok=True)
            with (destination / 'clustering.log').open('w', encoding='utf-8') as stream:
                with redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
                    summaries.append(run_segment(record, segment, args))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
