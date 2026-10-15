"""Cluster the 2018 UCR archive directly, optionally with DTW-GP alignment.

From the repository root::

    python examples/ucr_dtw_gp_archive.py --datasets Coffee --download
    python examples/ucr_dtw_gp_archive.py --download --resume

All TRAIN and TEST series in all 128 standard datasets are fitted by default.
Labels are used only for descriptive scores. Archive row order has no temporal
meaning: HDP transitions are disabled and A is fixed at zero. Gamma and C are
fixed, while each cluster learns Sigma and its GP representer. GP kernel
hyperparameters are fitted unless --fixed-gp-hyperparameters is selected.
Alignment is temporarily disabled by default for a comparison run; use
--alignment dtw_gp to restore it.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
import traceback
from urllib.request import urlopen
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import adjusted_rand_score

from hdpgpc import HDPGPC
from hdpgpc.config import (
    AlignmentConfig, DynamicsConfig, GPEmissionConfig, HDPConfig,
    HDPGPCConfig, InferenceConfig, MatrixPolicyConfig, MultiOutputConfig,
    NoisePolicyConfig, NumericalConfig, ObservationConfig,
)
from hdpgpc.diagnostics import (cluster_label_statistics, plot_embedding_summary,
                                plot_birth_warp_diagnostic)
from tutorial_utils import plot_record_results


ARCHIVE_URL = 'https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip'
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / 'agents' / 'data' / 'ucr' / 'UCRArchive_2018.zip'
DEFAULT_OUTPUT = ROOT / 'agents' / 'ucr_dtw_gp_results'
ARCHIVE_PASSWORD = b'someone'
MODEL_REVISION = 'dtw_gp_observation_seeds_LDS_KL_per_dimension_clusters_v8'


@dataclass(frozen=True)
class PreparedDataset:
    train: np.ndarray
    test: np.ndarray
    train_labels: np.ndarray
    test_labels: np.ndarray
    raw_points: int
    missing_rows: int


def download_archive(path: Path) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.zip.part')
    with urlopen(ARCHIVE_URL, timeout=90) as response, temporary.open('wb') as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
    temporary.replace(path)


def archive_datasets(archive: ZipFile) -> dict[str, tuple[str, str]]:
    names = set(archive.namelist())
    pairs = {}
    for train in sorted(names):
        parts = train.split('/')
        if (len(parts) != 3 or parts[0] != 'UCRArchive_2018'
                or parts[2] != f'{parts[1]}_TRAIN.tsv'):
            continue
        test = f'UCRArchive_2018/{parts[1]}/{parts[1]}_TEST.tsv'
        if test not in names:
            raise ValueError(f'Missing TEST split for {parts[1]}')
        pairs[parts[1]] = (train, test)
    if not pairs:
        raise ValueError('Archive has no canonical UCR 2018 TRAIN/TEST pairs')
    return pairs


def _read_tsv(archive: ZipFile, member: str) -> tuple[np.ndarray, np.ndarray]:
    with archive.open(member, pwd=ARCHIVE_PASSWORD) as stream:
        table = np.genfromtxt(stream, delimiter='\t', dtype=np.float64,
                              missing_values=('?', 'NaN', 'nan'),
                              filling_values=np.nan)
    table = np.atleast_2d(table)
    if table.shape[1] < 3 or not len(table) or not np.isfinite(table[:, 0]).all():
        raise ValueError(f'{member} needs finite labels and at least two data points')
    return table[:, 1:], table[:, 0]


def _resample(rows: np.ndarray, points: int) -> tuple[np.ndarray, int]:
    result = np.empty((len(rows), points), dtype=np.float64)
    target = np.linspace(0.0, 1.0, points)
    source = np.linspace(0.0, 1.0, rows.shape[1])
    missing = 0
    for index, row in enumerate(rows):
        finite = np.isfinite(row)
        if finite.sum() < 2:
            raise ValueError(f'Series {index} has fewer than two finite points')
        missing += int(not finite.all())
        result[index] = np.interp(target, source[finite], row[finite])
    return result, missing


def prepare_dataset(archive: ZipFile, pair: tuple[str, str],
                    max_points: int) -> PreparedDataset:
    train_raw, train_labels = _read_tsv(archive, pair[0])
    test_raw, test_labels = _read_tsv(archive, pair[1])
    points = min(max_points, max(train_raw.shape[1], test_raw.shape[1]))
    if points < 3:
        raise ValueError('Need at least three waveform points')
    train, missing_train = _resample(train_raw, points)
    test, missing_test = _resample(test_raw, points)
    center = float(train.mean())
    scale = float(train.std())
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError('TRAIN series have zero or non-finite global variation')
    return PreparedDataset((train - center) / scale, (test - center) / scale,
                           train_labels, test_labels,
                           max(train_raw.shape[1], test_raw.shape[1]),
                           missing_train + missing_test)


def make_config(grid: np.ndarray, args) -> HDPGPCConfig:
    dimension = len(grid)
    identity = np.eye(dimension)
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * RBF(length_scale=0.2, length_scale_bounds=(0.01, 2.0))
              + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1.0)))
    update = ('conjugate_batch' if args.schedule == 'offline'
              else 'conjugate_sequential')
    return HDPGPCConfig(
        x_basis=grid, verbose='debug',
        hdp=HDPConfig(hdp_hyp=args.hdp_preset, hmm_switch=False,
                      max_models=args.max_clusters),
        emission=GPEmissionConfig(
            kernels=[kernel], fit_gp=args.fit_gp,
            ini_sigma=args.sigma_prior, bound_sigma=(1e-8, 1e8)),
        observation=ObservationConfig(
            C_policy=MatrixPolicyConfig(mode='fixed', value=identity),
            Sigma_policy=NoisePolicyConfig(
                mode=args.sigma_mode, value=args.sigma_prior * identity)),
        dynamics=DynamicsConfig(
            model_type='dynamic', ini_gamma=args.gamma,
            A_policy=MatrixPolicyConfig(
                mode='fixed', value=np.zeros((dimension, dimension))),
            Gamma_policy=NoisePolicyConfig(
                mode='fixed', value=args.gamma * identity),
            bound_gamma=(1e-8, 1e8), annealing=False,
            bayesian_params=True, free_deg_MNIV=args.lds_degrees,
            lds_update=update, lds_scoring='final'),
        alignment=AlignmentConfig(
            policy=args.alignment, x_basis_warp=grid,
            dtw_points=args.dtw_points, dtw_inducing=args.dtw_inducing,
            dtw_lengthscale=args.dtw_lengthscale,
            dtw_amplitude=args.dtw_amplitude,
            dtw_anchor_noise=args.dtw_anchor_noise),
        inference=InferenceConfig(
            schedule=args.schedule, warp=args.alignment != 'none', hard_assignments=True,
            birth_lds_kl_per_dimension=args.birth_lds_kl_per_dimension,
            n_explore_steps=args.exploration_steps,
            it_limit=args.iterations if args.schedule == 'offline' else None),
        multi_output=MultiOutputConfig(
            n_outputs=1, share_gp=True, use_snr=False),
        numerics=NumericalConfig(
            seed=args.seed, num_threads=1, dtype='float64',
            deterministic=True),
    )


def run_dataset(archive: ZipFile, name: str, pair: tuple[str, str],
                args, run_id: str) -> dict:
    started = time.perf_counter()
    data = prepare_dataset(archive, pair, args.max_points)
    grid = np.linspace(0.0, 1.0, data.train.shape[1])[:, None]
    values = np.concatenate((data.train, data.test))
    labels = np.concatenate((data.train_labels, data.test_labels))
    rng = np.random.default_rng(args.seed)
    if args.max_series is not None and len(values) > args.max_series:
        selected = rng.choice(len(values), size=args.max_series, replace=False)
    else:
        selected = np.arange(len(values))
    # GP initialization still starts from one series. A seeded permutation
    # avoids giving the archive's arbitrary row order a special role.
    selected = rng.permutation(selected)
    values, labels = values[selected], labels[selected]

    model = HDPGPC(make_config(grid, args))
    capture_births = (args.birth_plots if args.plots and args.schedule == 'offline'
                      and args.alignment == 'dtw_gp' else 0)
    model.fit(values[:, :, None], grid=grid,
              birth_diagnostic_limit=capture_births)
    assignments = model.labels_
    scores, _, _, _ = cluster_label_statistics(assignments, labels)
    plot_path = ''
    cluster_plot_path = ''
    if args.plots:
        figure = plot_embedding_summary(
            values, assignments, labels, scores,
            title=f'UCR {name} | {args.alignment}', projection_name='Waveform')
        destination = args.output_dir / 'plots'
        destination.mkdir(parents=True, exist_ok=True)
        plot_file = destination / f'{name}_{run_id}_summary.png'
        try:
            figure.savefig(plot_file, dpi=160)
        finally:
            plt.close(figure)
        plot_path = str(plot_file.resolve())
        cluster_figures = plot_record_results(
            model, values[:, :, None], labels, grid=grid, record=f'UCR {name}',
            metrics=scores, ordered=False, show_latent=False,
            warped=args.alignment != 'none')
        for output, cluster_figure in enumerate(cluster_figures):
            cluster_file = destination / f'{name}_{run_id}_clusters_output_{output}.png'
            try:
                cluster_figure.savefig(cluster_file, dpi=160)
            finally:
                plt.close(cluster_figure)
            if output == 0:
                cluster_plot_path = str(cluster_file.resolve())
            print(f'  final cluster plot: {cluster_file}', flush=True)
        for attempt, record in enumerate(model.birth_diagnostics_, start=1):
            if not record.get('rows'):
                print(f'  birth warp plot {attempt}: {record.get("diagnostic_error", "no curves")}',
                      flush=True)
                continue
            figure = plot_birth_warp_diagnostic(
                record, title=f'UCR {name} | true seed label {labels[record["seed"]]:g}')
            birth_file = destination / f'{name}_{run_id}_birth_{attempt:02d}.png'
            try:
                figure.savefig(birth_file, dpi=160)
            finally:
                plt.close(figure)
            print(f'  birth warp plot: {birth_file}', flush=True)
    test_mask = selected >= len(data.train)
    return {
        'status': 'ok', 'train': len(data.train), 'test': len(data.test),
        'fitted': len(values), 'raw_points': data.raw_points,
        'model_points': len(grid), 'missing_rows': data.missing_rows,
        'classes': len(np.unique(labels)), 'clusters': model.n_clusters_,
        'purity_in_sample': scores['purity'],
        'ari_all': scores['adjusted_rand_index'],
        'nmi_all': scores['normalized_mutual_information'],
        'ari_test_transductive': (float(adjusted_rand_score(
            labels[test_mask], assignments[test_mask])) if test_mask.any() else ''),
        'seconds': round(time.perf_counter() - started, 3),
        'plot': plot_path, 'cluster_plot': cluster_plot_path, 'error': '',
    }


FIELDS = ('run_id', 'dataset', 'status', 'train', 'test', 'fitted', 'raw_points',
          'model_points', 'missing_rows', 'classes', 'clusters',
          'purity_in_sample', 'ari_all', 'nmi_all', 'ari_test_transductive',
          'seconds', 'plot', 'cluster_plot', 'error')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument('--download', action='store_true',
                        help='Download the official UCR ZIP when absent.')
    parser.add_argument('--datasets', nargs='+',
                        help='Dataset names; omit to process all 128.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-plots', action='store_false', dest='plots')
    parser.add_argument('--max-series', type=int,
                        help='Diagnostic subset size; omit to fit every series.')
    parser.add_argument('--max-points', type=int, default=64)
    parser.add_argument('--schedule', choices=('offline', 'online_fast', 'online'),
                        default='offline')
    parser.add_argument('--iterations', type=int, default=30,
                        help='Maximum offline iterations.')
    parser.add_argument('--max-clusters', type=int, default=30)
    parser.add_argument('--hdp-preset', choices=('less', 'balanced', 'more', 'max'),
                        default='more')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Fixed Gamma = gamma times identity.')
    parser.add_argument('--sigma-prior', type=float, default=0.01,
                        help='Initial Sigma covariance scale.')
    parser.add_argument('--sigma-mode', choices=('cluster', 'cluster_diagonal'),
                        default='cluster')
    parser.add_argument('--lds-degrees', type=int, default=10)
    parser.add_argument('--fixed-gp-hyperparameters', action='store_false',
                        dest='fit_gp', help='Skip first-series GP kernel fitting.')
    parser.add_argument('--alignment', choices=('none', 'dtw_gp'), default='dtw_gp',
                        help='Alignment policy; none is the temporary comparison default.')
    parser.add_argument('--birth-plots', type=int, default=0,
                        help='Maximum offline DTW-GP birth attempts to plot per dataset; 0 disables.')
    parser.add_argument('--birth-lds-kl-per-dimension',
                        action=argparse.BooleanOptionalAction, default=True,
                        help='Divide proposal LDS KL by the latent LDS dimension (default on).')
    parser.add_argument('--dtw-points', type=int, default=24)
    parser.add_argument('--dtw-inducing', type=int, default=10)
    parser.add_argument('--dtw-lengthscale', type=float, default=0.2)
    parser.add_argument('--dtw-amplitude', type=float, default=0.1)
    parser.add_argument('--dtw-anchor-noise', type=float, default=0.05)
    parser.add_argument('--exploration-steps', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if (args.max_points < 3 or args.iterations < 1 or args.max_clusters < 1
            or args.lds_degrees < 1 or args.exploration_steps < 1
            or args.birth_plots < 0
            or args.dtw_points < 2 or args.dtw_inducing < 2
            or any(value <= 0 or not np.isfinite(value) for value in (
                args.gamma, args.sigma_prior, args.dtw_lengthscale,
                args.dtw_amplitude, args.dtw_anchor_noise))
            or (args.max_series is not None and args.max_series < 2)):
        parser.error('Size, iteration, covariance and alignment controls must be positive')
    if args.download:
        download_archive(args.archive)
    if not args.archive.is_file():
        parser.error(f'Archive not found: {args.archive}; use --download')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.archive.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    settings = {name: getattr(args, name) for name in (
        'max_series', 'max_points', 'schedule', 'iterations', 'max_clusters',
        'hdp_preset', 'gamma', 'sigma_prior', 'sigma_mode', 'lds_degrees',
        'fit_gp', 'alignment', 'birth_plots', 'dtw_points', 'dtw_inducing', 'dtw_lengthscale',
        'birth_lds_kl_per_dimension',
        'dtw_amplitude', 'dtw_anchor_noise', 'exploration_steps',
        'seed', 'plots')}
    settings['model_revision'] = MODEL_REVISION
    run_id = hashlib.sha256((digest + json.dumps(
        settings, sort_keys=True)).encode('utf-8')).hexdigest()[:16]
    print(f'UCR archive: {args.archive} (SHA-256 {digest})', flush=True)
    print(f'Run ID: {run_id}; settings: {settings}', flush=True)
    results_path = args.output_dir / 'results.csv'
    if results_path.is_file():
        with results_path.open(newline='', encoding='utf-8') as stream:
            if csv.DictReader(stream).fieldnames != list(FIELDS):
                results_path = args.output_dir / f'results_{run_id}.csv'
    completed = set()
    if args.resume and results_path.is_file():
        with results_path.open(newline='', encoding='utf-8') as stream:
            completed = {row['dataset'] for row in csv.DictReader(stream)
                         if row['status'] == 'ok' and row.get('run_id') == run_id
                         and (not args.plots or (Path(row.get('plot', '')).is_file()
                              and Path(row.get('cluster_plot', '')).is_file()))}
    with ZipFile(args.archive) as archive:
        pairs = archive_datasets(archive)
        chosen = args.datasets or sorted(pairs)
        unknown = sorted(set(chosen) - set(pairs))
        if unknown:
            parser.error(f'Unknown UCR datasets: {", ".join(unknown)}')
        print(f'Available standard datasets: {len(pairs)}; selected: {len(chosen)}',
              flush=True)
        append = results_path.is_file()
        with results_path.open('a' if append else 'w', newline='',
                               encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            if not append:
                writer.writeheader()
            for index, name in enumerate(chosen, start=1):
                if name in completed:
                    print(f'[{index}/{len(chosen)}] {name}: already complete',
                          flush=True)
                    continue
                print(f'[{index}/{len(chosen)}] {name}: fitting', flush=True)
                try:
                    record = run_dataset(archive, name, pairs[name], args, run_id)
                except Exception as error:
                    record = {'status': 'error',
                              'error': f'{type(error).__name__}: {error}'}
                    (args.output_dir / f'{name}_error.log').write_text(
                        traceback.format_exc(), encoding='utf-8')
                writer.writerow({'run_id': run_id, 'dataset': name, **record})
                stream.flush()
                if record['status'] == 'ok':
                    print(f"  clusters={record['clusters']} "
                          f"ARI={record['ari_all']:.4f} "
                          f"NMI={record['nmi_all']:.4f} "
                          f"elapsed={record['seconds']:.1f}s", flush=True)
                else:
                    print(f"  ERROR: {record['error']}", flush=True)
    print(f'Results: {results_path}', flush=True)


if __name__ == '__main__':
    main()
