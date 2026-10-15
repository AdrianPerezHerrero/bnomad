"""Cluster UCR datasets with a TRAIN-calibrated fixed affine PCA emission.

python examples/ucr_pca_archive.py --datasets Coffee
python examples/ucr_pca_archive.py --datasets ACSF1 --components 12
python examples/ucr_pca_archive.py --resume

Omitting --datasets runs the whole local UCR archive. Normalization and PCA use
TRAIN only; both TRAIN and TEST then participate in unsupervised clustering.
Reported scores are descriptive transductive clustering measures. No class
labels enter calibration or fitting. C/offset are fixed, A=0, HMM conditioning
is disabled, and Gamma/Sigma follow the existing user-selected noise policies.
"""

import argparse
from contextlib import redirect_stdout, redirect_stderr
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import adjusted_rand_score

from hdpgpc import HDPGPC
from hdpgpc.config import (AlignmentConfig, DynamicsConfig, GPEmissionConfig,
                          HDPConfig, HDPGPCConfig, InferenceConfig, MatrixPolicyConfig,
                          MultiOutputConfig, NoisePolicyConfig, NumericalConfig,
                          with_pca_emission)
from hdpgpc.emissions import estimate_pca_observation
from hdpgpc.diagnostics import (Tee, cluster_label_statistics, format_confusion_matrix,
                                plot_embedding_summary)
from tutorial_utils import plot_record_results
from ucr_dtw_gp_archive import DEFAULT_ARCHIVE, archive_datasets, download_archive, prepare_dataset


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / 'agents' / 'ucr_pca_results'
MODEL_REVISION = 'fixed_affine_pca_zero_A_v1'
NOISE_MODES = ('fixed', 'cluster', 'cluster_diagonal', 'shared', 'shared_diagonal')
FIELDS = ('run_id', 'dataset', 'status', 'train', 'test', 'fitted', 'raw_points',
          'model_points', 'classes', 'clusters', 'latent_dim', 'pca_calibration_train',
          'explained_variance_ratio', 'purity_in_sample', 'ari_all', 'nmi_all',
          'ari_test_transductive', 'seconds', 'output_directory', 'error')


def make_config(estimate, args):
    """Install PCA using the package's existing fixed affine/LDS machinery."""
    dimension = estimate.components.shape[1]
    base = HDPGPCConfig(
        verbose=args.verbose,
        hdp=HDPConfig(hdp_hyp=args.hdp_preset, hmm_switch=False,
                      max_models=args.max_clusters),
        emission=GPEmissionConfig(
            kernels=[ConstantKernel(1.) * RBF(max(1., dimension / 4.))
                     + WhiteKernel(0.01)],
            fit_gp=False, ini_sigma=float(np.diag(estimate.residual_covariance).mean()),
            bound_sigma=(1e-8, 1e8)),
        dynamics=DynamicsConfig(
            ini_gamma=float(estimate.latent_variances.mean()),
            bound_gamma=(1e-8, 1e8), annealing=False, bayesian_params=True,
            free_deg_MNIV=args.lds_degrees,
            lds_update='conjugate_batch' if args.schedule == 'offline' else 'conjugate_sequential',
            lds_scoring='final'),
        alignment=AlignmentConfig(policy='none'),
        inference=InferenceConfig(
            schedule=args.schedule, warp=False, hard_assignments=args.hard_assignments,
            birth_lds_kl_per_dimension=True, n_explore_steps=args.exploration_steps,
            it_limit=args.iterations if args.schedule == 'offline' else None),
        multi_output=MultiOutputConfig(n_outputs=1, share_gp=True, use_snr=False),
        numerics=NumericalConfig(seed=args.seed, num_threads=1,
                                 dtype='float64', deterministic=True),
    )
    configured = with_pca_emission(
        base, estimate, sigma_policy=NoisePolicyConfig(mode=args.sigma_mode))
    # UCR rows are independent examples. A=0 removes latent temporal prediction;
    # Gamma becomes the cluster's latent spread, rather than a trajectory noise.
    return replace(configured, dynamics=replace(
        configured.dynamics,
        A_policy=MatrixPolicyConfig(mode='fixed', value=np.zeros((dimension, dimension))),
        Gamma_policy=NoisePolicyConfig(mode=args.gamma_mode,
                                       value=np.diag(estimate.latent_variances))))


def run_dataset(archive, name, pair, args, run_id, destination):
    started = time.perf_counter()
    data = prepare_dataset(archive, pair, args.max_points)
    grid = np.linspace(0., 1., data.train.shape[1])[:, None]
    calibration_count = min(args.pca_samples, len(data.train))
    dimension = min(args.components, len(grid) - 1, calibration_count - 1)
    if dimension < 1:
        raise ValueError('PCA requires at least two TRAIN series and two waveform points')
    # The existing pilot supplies orthonormal C, offset, code variances and
    # diagonal reconstruction-noise covariance, independently of clustering.
    estimate = estimate_pca_observation(
        data.train, grid, components=dimension, max_beats=args.pca_samples,
        random_state=args.seed, sampling='uniform', noise_floor=args.noise_floor)
    estimate = replace(estimate, latent_variances=estimate.latent_variances * args.process_noise,
                       residual_covariance=estimate.residual_covariance * args.observation_noise_scale)
    config = make_config(estimate, args)
    (destination / 'config.yaml').write_text(config.to_yaml(), encoding='utf-8')
    np.savez_compressed(destination / 'pca_calibration.npz',
                        C=estimate.components, offset=estimate.mean,
                        latent_prior_variances=estimate.latent_variances,
                        Sigma_prior=estimate.residual_covariance,
                        grid=estimate.grid, train_indices=np.asarray(estimate.beat_indices),
                        explained_variance_ratio=estimate.explained_variance_ratio)
    waveforms = np.concatenate((data.train, data.test))
    labels = np.concatenate((data.train_labels, data.test_labels))
    rng = np.random.default_rng(args.seed)
    selected = (rng.choice(len(waveforms), args.max_series, replace=False)
                if args.max_series is not None and args.max_series < len(waveforms)
                else np.arange(len(waveforms)))
    selected = rng.permutation(selected)
    values, labels = waveforms[selected, :, None], labels[selected]
    if args.verbose != 'silent':
        print(f'UCR {name}: TRAIN={len(data.train)}, TEST={len(data.test)}, '
              f'fitted={len(values)}, waveform points={len(grid)}, PCA dimension={dimension}', flush=True)
        print(f'PCA calibration={calibration_count} TRAIN series; '
              f'explained variance={estimate.explained_variance_ratio:.3%}; '
              f'Gamma={args.gamma_mode}, Sigma={args.sigma_mode}, '
              f'A=0, HMM=False, warp=False, schedule={args.schedule}', flush=True)
    model = HDPGPC(config).fit(values, grid=grid)
    scores, classes, confusion, cluster_labels = cluster_label_statistics(model.labels_, labels)
    print(f'{name}: clusters={model.n_clusters_}, purity={scores["purity"]:.3f}, '
          f'ARI={scores["adjusted_rand_index"]:.3f}, '
          f'NMI={scores["normalized_mutual_information"]:.3f}')
    print(format_confusion_matrix(classes, confusion))
    print('Majority-label mapping is fitted and evaluated on the clustered data.')
    if model.n_clusters_ >= args.max_clusters:
        print('Configured cluster cap reached; this is not a convergence claim.')
    with (destination / 'assignments.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['fit_row', 'archive_row', 'split', 'true_label', 'cluster', 'cluster_majority_label'])
        for row, source in enumerate(selected):
            cluster = int(model.labels_[row])
            writer.writerow([row, int(source), 'TRAIN' if source < len(data.train) else 'TEST',
                             labels[row], cluster, cluster_labels[cluster]])
    test_mask = selected >= len(data.train)
    result = {
        'status': 'ok', 'train': len(data.train), 'test': len(data.test),
        'fitted': len(values), 'raw_points': data.raw_points, 'model_points': len(grid),
        'classes': len(np.unique(labels)), 'clusters': model.n_clusters_, 'latent_dim': dimension,
        'pca_calibration_train': calibration_count,
        'explained_variance_ratio': estimate.explained_variance_ratio,
        'purity_in_sample': scores['purity'], 'ari_all': scores['adjusted_rand_index'],
        'nmi_all': scores['normalized_mutual_information'],
        'ari_test_transductive': (float(adjusted_rand_score(labels[test_mask], model.labels_[test_mask]))
                                  if test_mask.any() else ''),
        'seconds': round(time.perf_counter() - started, 3),
        'output_directory': str(destination.resolve()), 'error': '',
    }
    if args.plots:
        codes = (values[:, :, 0] - estimate.mean) @ estimate.components
        figure = plot_embedding_summary(codes, model.labels_, labels, scores,
                                         title=f'UCR {name} | fixed PCA', projection_name='PCA')
        try:
            figure.savefig(destination / 'projection.png', dpi=160)
        finally:
            plt.close(figure)
        figures = plot_record_results(
            model, values, labels, grid=grid, record=f'UCR {name} PCA', metrics=scores,
            ordered=False, show_latent=False)
        for output, figure in enumerate(figures):
            try:
                figure.savefig(destination / f'clusters_output_{output}.png', dpi=160)
            finally:
                plt.close(figure)
    report = {'run_id': run_id, 'dataset': name, **result,
              'missing_rows': data.missing_rows, 'confusion_classes': classes.tolist(),
              'confusion': confusion.tolist(), 'cluster_majority_labels': cluster_labels,
              'cluster_cap_reached': model.n_clusters_ >= args.max_clusters,
              'archive_sha256': args.archive_sha256,
              'calibration_indices_train': list(estimate.beat_indices),
              'evaluation': 'transductive: TRAIN and TEST both participate in clustering'}
    (destination / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--datasets', nargs='+', help='Names; omit to process the whole archive.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plots', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--max-series', type=int, help='Optional seeded random subset of TRAIN+TEST.')
    parser.add_argument('--max-points', type=int, default=64)
    parser.add_argument('--components', '--latent-dim', type=int, default=24,
                        help='PCA dimension, capped by waveform and TRAIN calibration sizes.')
    parser.add_argument('--pca-samples', type=int, default=400)
    parser.add_argument('--noise-floor', type=float, default=1e-6)
    parser.add_argument('--process-noise', '--latent-covariance-scale', type=float, default=1.0,
                        help='Multiply calibrated latent prior variances (default 1).')
    parser.add_argument('--observation-noise-scale', type=float, default=1.0,
                        help='Multiply calibrated residual Sigma prior (default 1).')
    parser.add_argument('--gamma-mode', choices=NOISE_MODES, default='cluster')
    parser.add_argument('--sigma-mode', choices=NOISE_MODES, default='cluster')
    parser.add_argument('--schedule', choices=('offline', 'online', 'online_fast'), default='offline')
    parser.add_argument('--iterations', type=int, default=60)
    parser.add_argument('--max-clusters', type=int, default=50)
    parser.add_argument('--hdp-preset', choices=('less', 'balanced', 'more', 'max'), default='max')
    parser.add_argument('--lds-degrees', type=int, default=10)
    parser.add_argument('--exploration-steps', type=int, default=10)
    parser.add_argument('--hard-assignments', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--verbose', choices=('silent', 'progress', 'debug'), default='progress')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if (args.max_points < 3 or args.components < 1 or args.pca_samples < 2
            or args.iterations < 1 or args.max_clusters < 1 or args.lds_degrees < 3
            or args.exploration_steps < 1 or not 0 <= args.seed < 2**32
            or (args.max_series is not None and args.max_series < 2)
            or any(not np.isfinite(value) or value <= 0 for value in
                   (args.noise_floor, args.process_noise, args.observation_noise_scale))):
        parser.error('Size, iteration, covariance and prior controls must be positive')
    if args.download:
        download_archive(args.archive)
    if not args.archive.is_file():
        parser.error(f'Archive not found: {args.archive}; use --download')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.archive.open('rb') as stream:
        args.archive_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    settings = {name: getattr(args, name) for name in (
        'max_series', 'max_points', 'components', 'pca_samples', 'noise_floor',
        'process_noise', 'observation_noise_scale', 'gamma_mode', 'sigma_mode',
        'schedule', 'iterations', 'max_clusters', 'hdp_preset', 'lds_degrees',
        'exploration_steps', 'hard_assignments', 'verbose', 'seed', 'plots')}
    run_id = hashlib.sha256((args.archive_sha256 + MODEL_REVISION
                            + json.dumps(settings, sort_keys=True)).encode()).hexdigest()[:16]
    (args.output_dir / f'settings_{run_id}.json').write_text(
        json.dumps(settings, indent=2), encoding='utf-8')
    results_path = args.output_dir / 'results.csv'
    if results_path.is_file():
        with results_path.open(newline='', encoding='utf-8') as stream:
            if csv.DictReader(stream).fieldnames != list(FIELDS):
                results_path = args.output_dir / f'results_{run_id}.csv'
    with ZipFile(args.archive) as archive:
        pairs = archive_datasets(archive)
        chosen = list(dict.fromkeys(args.datasets or sorted(pairs)))
        unknown = sorted(set(chosen) - set(pairs))
        if unknown:
            parser.error(f'Unknown datasets: {unknown}')
        if args.verbose != 'silent':
            print(f'UCR PCA: {len(chosen)} datasets selected; run_id={run_id}', flush=True)
        append = results_path.is_file()
        with results_path.open('a' if append else 'w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            if not append:
                writer.writeheader()
            for index, name in enumerate(chosen, start=1):
                destination = args.output_dir / name / run_id
                complete = destination / 'results.json'
                if args.resume and complete.is_file():
                    saved = json.loads(complete.read_text(encoding='utf-8'))
                    if (saved.get('status') == 'ok' and (not args.plots or
                            all((destination / path).is_file() for path in
                                ('projection.png', 'clusters_output_0.png')))):
                        print(f'[{index}/{len(chosen)}] {name}: already complete', flush=True)
                        continue
                destination.mkdir(parents=True, exist_ok=True)
                if args.verbose != 'silent':
                    print(f'[{index}/{len(chosen)}] {name}: fitting', flush=True)
                # Replace any older success marker before a requested rerun.
                # A failed rerun must not be silently skipped by --resume.
                complete.write_text(json.dumps({'status': 'running', 'run_id': run_id,
                                                'dataset': name}), encoding='utf-8')
                with (destination / 'clustering.log').open('w', encoding='utf-8') as log:
                    with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                        try:
                            result = run_dataset(archive, name, pairs[name], args, run_id, destination)
                        except Exception as error:
                            traceback.print_exc()
                            result = {'status': 'error', 'output_directory': str(destination.resolve()),
                                      'error': f'{type(error).__name__}: {error}'}
                            complete.write_text(json.dumps({'run_id': run_id, 'dataset': name,
                                                            **result}, indent=2), encoding='utf-8')
                writer.writerow({'run_id': run_id, 'dataset': name, **result})
                stream.flush()
                if result['status'] == 'ok':
                    print(f'  Saved {destination}', flush=True)
                else:
                    print(f'  ERROR: {result["error"]}', flush=True)
    print(f'Results: {results_path}', flush=True)


if __name__ == '__main__':
    main()
