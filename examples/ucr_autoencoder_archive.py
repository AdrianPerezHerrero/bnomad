"""Run the fixed linear-autoencoder HDP-GPC over the 2018 UCR archive.

From the repository root::

    python examples/ucr_autoencoder_archive.py --datasets Coffee
    python examples/ucr_autoencoder_archive.py

The default reads every TRAIN and TEST series in all 128 standard datasets.
TRAIN series alone calibrate normalization and the tied linear autoencoder;
both splits then enter *unsupervised* clustering. UCR rows have no meaningful
temporal order, so HMM transitions are disabled and the latent transition is
fixed to A=0 while each cluster learns Gamma. The scores are exploratory transductive
clustering measures, not held-out classification accuracy. No labels enter
model fitting.

The official ZIP can be downloaded with --download. It is kept in the
repository's ignored agents/data/ucr directory and read without extraction.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import time
import traceback
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import adjusted_rand_score

from autoencoder_record_clustering import TiedLinearAutoencoder
from hdpgpc import HDPGPC
from hdpgpc.config import (
    AlignmentConfig, DynamicsConfig, GPEmissionConfig, HDPConfig,
    HDPGPCConfig, InferenceConfig, MatrixPolicyConfig, MultiOutputConfig, NoisePolicyConfig,
    NumericalConfig, with_fixed_observation_model,
)
from hdpgpc.dynamics import AffineGaussianObservation
from hdpgpc.diagnostics import cluster_label_statistics, plot_embedding_summary


ARCHIVE_URL = 'https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip'
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / 'agents' / 'data' / 'ucr' / 'UCRArchive_2018.zip'
DEFAULT_OUTPUT = ROOT / 'agents' / 'ucr_autoencoder_results'
ARCHIVE_PASSWORD = b'someone'  # Provided by the archive briefing document.
MODEL_REVISION = 'zero_A_cluster_Gamma_v1'


@dataclass(frozen=True)
class PreparedDataset:
    train: np.ndarray
    test: np.ndarray
    train_labels: np.ndarray
    test_labels: np.ndarray
    raw_points: int
    missing_rows: int


def download_archive(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.zip.part')
    if path.is_file():
        return
    with urlopen(ARCHIVE_URL, timeout=90) as response, temporary.open('wb') as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
    temporary.replace(path)


def archive_datasets(archive: ZipFile) -> dict[str, tuple[str, str]]:
    """Select canonical UCRArchive_2018/<name>/<name>_{TRAIN,TEST}.tsv pairs."""
    names = set(archive.namelist())
    pairs = {}
    for name in sorted(names):
        parts = name.split('/')
        if (len(parts) != 3 or parts[0] != 'UCRArchive_2018'
                or parts[2] != f'{parts[1]}_TRAIN.tsv'):
            continue
        test = f'UCRArchive_2018/{parts[1]}/{parts[1]}_TEST.tsv'
        if test not in names:
            raise ValueError(f'Missing TEST split for {parts[1]}')
        pairs[parts[1]] = (name, test)
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
        raise ValueError('Need at least three points for a reduced latent model')
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


def calibrate_autoencoder(train: np.ndarray, *, grid: np.ndarray,
                          requested_dim: int, beats: int, epochs: int,
                          seed: int, covariance_scale: float = 1.0
                          ) -> tuple[AffineGaussianObservation, float, int]:
    """Use the same tied orthonormal linear decoder as the ECG example."""
    rng = np.random.default_rng(seed)
    count = min(beats, len(train))
    latent_dim = min(requested_dim, train.shape[1] - 1, count - 1)
    if latent_dim < 1:
        raise ValueError('Need at least two TRAIN series for autoencoder calibration')
    chosen = rng.choice(len(train), size=count, replace=False)
    batch = train[chosen]
    mean = batch.mean(axis=0)
    scale = float(np.sqrt(np.mean((batch - mean) ** 2)))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError('Autoencoder calibration has zero variation')

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    network = TiedLinearAutoencoder(train.shape[1], latent_dim, mean, scale)
    training = torch.as_tensor(batch, dtype=torch.float64)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.02)
    for _ in range(epochs):
        optimizer.zero_grad()
        reconstructed = network(training)
        loss = (((reconstructed - training) / scale) ** 2).mean()
        loss.backward()
        optimizer.step()
    network.eval()
    with torch.no_grad():
        basis = network.basis().numpy()
        codes = network.encode(training).numpy()
        residual = batch - network(training).numpy()
    latent_cov = np.atleast_2d(np.cov(codes, rowvar=False))
    ridge = max(1e-8, 1e-5 * float(np.trace(latent_cov)) / latent_dim)
    latent_cov += ridge * np.eye(latent_dim)
    latent_cov *= covariance_scale
    noise = max(float(np.mean(residual ** 2)),
                0.05 * float(np.var(batch)), 1e-8)
    decoder = AffineGaussianObservation(
        C=scale * basis, Sigma=noise * np.eye(train.shape[1]),
        observation_grid=grid, offset=mean,
        latent_covariance=latent_cov)
    return decoder, float(loss.detach()), count


def make_config(decoder: AffineGaussianObservation, args) -> HDPGPCConfig:
    dim = decoder.latent_dim
    kernel = (ConstantKernel(1.0) * RBF(length_scale=max(1.0, dim / 4.0))
              + WhiteKernel(noise_level=0.01))
    update = ('conjugate_batch' if args.schedule == 'offline'
              else 'conjugate_sequential')
    base = HDPGPCConfig(
        verbose='silent',
        hdp=HDPConfig(hdp_hyp=args.hdp_preset, hmm_switch=False,
                      max_models=args.max_clusters),
        emission=GPEmissionConfig(
            kernels=[kernel], fit_gp=False,
            ini_sigma=float(np.mean(np.diag(decoder.Sigma))),
            bound_sigma=(1e-8, 1e8)),
        dynamics=DynamicsConfig(
            ini_gamma=float(np.mean(np.diag(decoder.latent_covariance))),
            bound_gamma=(1e-8, 1e8),
            annealing=False, bayesian_params=True, free_deg_MNIV=10,
            lds_update=update, lds_scoring='final'),
        alignment=AlignmentConfig(policy='none'),
        inference=InferenceConfig(
            schedule=args.schedule, warp=False, hard_assignments=True,
            birth_lds_kl_per_dimension=True, n_explore_steps=4,
            it_limit=args.iterations if args.schedule == 'offline' else None),
        multi_output=MultiOutputConfig(n_outputs=1, share_gp=True, use_snr=False),
        numerics=NumericalConfig(seed=args.seed, num_threads=1,
                                 dtype='float64', deterministic=True),
    )
    configured = with_fixed_observation_model(
        base, decoder, sigma_policy=NoisePolicyConfig(mode='fixed'))
    # A=0 makes successive archive rows independent under the latent prior.
    # Gamma starts from the calibrated encoder-code covariance, then each
    # cluster updates its own posterior without depending on the previous row.
    return replace(configured, dynamics=replace(
        configured.dynamics,
        A_policy=MatrixPolicyConfig(mode='fixed', value=np.zeros((dim, dim))),
        Gamma_policy=NoisePolicyConfig(
            mode='cluster', value=decoder.latent_covariance.copy())))


def run_dataset(archive: ZipFile, name: str, pair: tuple[str, str],
                args, run_id: str) -> dict:
    started = time.perf_counter()
    data = prepare_dataset(archive, pair, args.max_points)
    grid = np.linspace(0.0, 1.0, data.train.shape[1])[:, None]
    decoder, ae_loss, calibration_count = calibrate_autoencoder(
        data.train, grid=grid, requested_dim=args.latent_dim,
        beats=args.ae_beats, epochs=args.ae_epochs, seed=args.seed,
        covariance_scale=args.process_noise)
    all_series = np.concatenate((data.train, data.test))
    all_labels = np.concatenate((data.train_labels, data.test_labels))
    if args.max_series is not None and len(all_series) > args.max_series:
        selected = np.sort(np.random.default_rng(args.seed).choice(
            len(all_series), size=args.max_series, replace=False))
        all_series, all_labels = all_series[selected], all_labels[selected]
    else:
        selected = np.arange(len(all_series))
    model = HDPGPC(make_config(decoder, args))
    model.fit(all_series[:, :, None], grid=grid)
    assignments = model.labels_
    scores, _, _, _ = cluster_label_statistics(assignments, all_labels)
    plot_path = ''
    if args.plots:
        codes = ((all_series - decoder.offset) @ np.linalg.pinv(decoder.C).T)
        figure = plot_embedding_summary(
            codes, assignments, all_labels, scores, title=f'UCR {name}')
        destination = args.output_dir / 'plots'
        destination.mkdir(parents=True, exist_ok=True)
        plot_file = destination / f'{name}_{run_id}_summary.png'
        try:
            figure.savefig(plot_file, dpi=160)
        finally:
            plt.close(figure)
        plot_path = str(plot_file.resolve())
    test_mask = selected >= len(data.train)
    return {
        'status': 'ok', 'train': len(data.train), 'test': len(data.test),
        'fitted': len(all_series), 'raw_points': data.raw_points,
        'model_points': len(grid), 'missing_rows': data.missing_rows,
        'classes': len(np.unique(all_labels)), 'clusters': model.n_clusters_,
        'latent_dim': decoder.latent_dim, 'ae_calibration_train': calibration_count,
        'ae_scaled_mse': ae_loss, 'purity_in_sample': scores['purity'],
        'ari_all': scores['adjusted_rand_index'],
        'nmi_all': scores['normalized_mutual_information'],
        'ari_test_transductive': (float(adjusted_rand_score(
            all_labels[test_mask], assignments[test_mask])) if test_mask.any() else ''),
        'seconds': round(time.perf_counter() - started, 3),
        'plot': plot_path, 'error': '',
    }


FIELDS = ('run_id', 'dataset', 'status', 'train', 'test', 'fitted', 'raw_points',
          'model_points', 'missing_rows', 'classes', 'clusters', 'latent_dim',
          'ae_calibration_train', 'ae_scaled_mse', 'purity_in_sample',
          'ari_all', 'nmi_all', 'ari_test_transductive', 'seconds', 'plot', 'error')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument('--download', action='store_true',
                        help='Download the official 2018 ZIP when absent.')
    parser.add_argument('--datasets', nargs='+',
                        help='Dataset folder names; omit to run all 128.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--no-plots', action='store_false', dest='plots',
                        help='Skip the per-dataset latent projection and cluster summary PNG.')
    parser.add_argument('--resume', action='store_true',
                        help='Skip datasets already recorded as successful.')
    parser.add_argument('--max-series', type=int,
                        help='Diagnostic subset size; omit for every TRAIN and TEST series.')
    parser.add_argument('--max-points', type=int, default=64,
                        help='Maximum waveform resolution after interpolation (default: 64).')
    parser.add_argument('--ae-beats', type=int, default=400)
    parser.add_argument('--ae-epochs', type=int, default=60)
    parser.add_argument('--latent-dim', type=int, default=15)
    parser.add_argument('--schedule', choices=('offline', 'online_fast', 'online'),
                        default='offline')
    parser.add_argument('--iterations', type=int, default=30,
                        help='Maximum offline iterations (default: 30).')
    parser.add_argument('--max-clusters', type=int, default=30)
    parser.add_argument('--hdp-preset', choices=('less', 'balanced', 'more'),
                        default='more')
    parser.add_argument('--process-noise', '--latent-covariance-scale',
                        dest='process_noise', type=float, default=1.0,
                        help='Scale the calibrated, fixed latent covariance (default: 1).')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if (args.max_points < 3 or args.ae_beats < 2 or args.ae_epochs < 1
            or args.latent_dim < 1 or args.iterations < 1 or args.max_clusters < 1
            or args.process_noise <= 0 or
            (args.max_series is not None and args.max_series < 2)):
        parser.error('All size, iteration and noise controls must be positive')
    if args.download:
        download_archive(args.archive)
    if not args.archive.is_file():
        parser.error(f'Archive not found: {args.archive}; use --download')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.archive.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    print(f'UCR archive: {args.archive} (SHA-256 {digest})', flush=True)
    settings = {name: getattr(args, name) for name in (
        'max_series', 'max_points', 'ae_beats', 'ae_epochs', 'latent_dim',
        'schedule', 'iterations', 'max_clusters', 'hdp_preset',
        'process_noise', 'seed', 'plots')}
    settings['model_revision'] = MODEL_REVISION
    run_id = hashlib.sha256((digest + json.dumps(
        settings, sort_keys=True)).encode('utf-8')).hexdigest()[:16]
    print(f'Run ID: {run_id}; settings: {settings}', flush=True)
    results_path = args.output_dir / 'results.csv'
    if results_path.is_file():
        with results_path.open(newline='', encoding='utf-8') as stream:
            old_fields = csv.DictReader(stream).fieldnames
        if old_fields != list(FIELDS):
            results_path = args.output_dir / f'results_{run_id}.csv'
    completed = set()
    if args.resume and results_path.is_file():
        with results_path.open(newline='', encoding='utf-8') as stream:
            completed = {row['dataset'] for row in csv.DictReader(stream)
                         if row['status'] == 'ok' and row.get('run_id') == run_id
                         and (not args.plots or Path(row.get('plot', '')).is_file())}
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
                    print(f'[{index}/{len(chosen)}] {name}: already complete', flush=True)
                    continue
                print(f'[{index}/{len(chosen)}] {name}: fitting', flush=True)
                try:
                    record = run_dataset(archive, name, pairs[name], args, run_id)
                except Exception as error:
                    record = {'status': 'error', 'error': f'{type(error).__name__}: {error}'}
                    (args.output_dir / f'{name}_error.log').write_text(
                        traceback.format_exc(), encoding='utf-8')
                writer.writerow({'run_id': run_id, 'dataset': name, **record})
                stream.flush()
                if record['status'] == 'ok':
                    print(f"  clusters={record['clusters']} ARI={record['ari_all']:.4f} "
                          f"NMI={record['nmi_all']:.4f} "
                          f"elapsed={record['seconds']:.1f}s", flush=True)
                else:
                    print(f"  ERROR: {record['error']}", flush=True)
    print(f'Results: {results_path}', flush=True)


if __name__ == '__main__':
    main()
