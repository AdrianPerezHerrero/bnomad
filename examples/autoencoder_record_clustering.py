"""Train a 400-beat linear autoencoder, then cluster a complete MIT-BIH record.

Run from the repository root::

    python examples/autoencoder_record_clustering.py --records 100
    python examples/autoencoder_record_clustering.py --records 100 102 --save-plots

Edit the controls below for the record, channel, latent dimension, LDS/HDP
priors, and inference schedule. This deliberately uses a *linear*, tied-weight
autoencoder: its frozen decoder is an exact affine Gaussian observation model.
The encoder defines the latent coordinates and their initial covariance. During
clustering, each beat is encoded into an exact latent Gaussian likelihood
factor and combined with the cluster's dynamic prior. With the default
isotropic decoder noise, its latent measurement mean equals the trained
encoder output. Waveform likelihoods still use the frozen decoder.

For online experiments, choose AE_SAMPLING='prefix'. Those calibration beats are
replayed during fitting, so metrics on the same record are descriptive, not a
strictly held-out evaluation. A nonlinear decoder is not supported by the
current conjugate LDS fitter.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import sys
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from tutorial_utils import plot_record_results
from hdpgpc import HDPGPC
from hdpgpc.config import (
    AlignmentConfig, DynamicsConfig, GPEmissionConfig, HDPConfig,
    HDPGPCConfig, InferenceConfig, MultiOutputConfig, NoisePolicyConfig,
    NumericalConfig, with_fixed_observation_model,
)
from hdpgpc.dynamics import AffineGaussianObservation
from hdpgpc.diagnostics import (
    Tee, cluster_label_statistics, format_confusion_matrix,
    model_lds_arrays, model_lds_dynamics,
    plot_lds_diagnostics, plot_lds_dynamics,
)
from hdpgpc.preprocessing import prepare_sequences
from hdpgpc.datasets import load_mitbih_record


# Experiment controls. The decoder and encoder always use one waveform channel.
# RECORDS = ('100',)
RECORDS = (
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119",
    "121", "122", "123", "124",
    "200", "201", "202", "203", "205", "207", "208", "209", "210",
    "212", "213", "214", "215", "217", "219", "220", "221", "222", "223",
    "228", "230", "231", "232", "233", "234",
)
CHANNEL = 0
SCALE_WAVEFORMS = False
UNIT_GRID = False
INFERENCE_MODE = 'offline'       # 'offline', 'online', 'online_fast'
VERBOSE = 'debug'             # 'silent', 'progress', 'debug'
HARD_ASSIGNMENTS = True
OFFLINE_ITERATIONS = 50
HDP_PRESET = 'more'           # 'less', 'balanced', 'more'
MAX_CLUSTERS = 35
LDS_UPDATE = 'conjugate_batch'    # offline; online uses conjugate_sequential
LDS_DEGREES_OF_FREEDOM = 10
INITIAL_PROCESS_NOISE = 0.1      # Latent-coordinate variance, not waveform variance.
SIGMA_POLICY = 'fixed'           # 'fixed', 'cluster_diagonal', 'cluster', etc.
BIRTH_LDS_KL_PER_POINT = True

AE_BEATS = 1000
AE_SAMPLING = 'uniform'           # 'uniform' or 'prefix'; online requires prefix.
LATENT_DIM = 15
AE_EPOCHS = 500
AE_LEARNING_RATE = 0.02
AE_NOISE_STRUCTURE = 'isotropic'  # 'isotropic' keeps encoder coordinates exact; or 'diagonal'.
AE_NOISE_FLOOR_FRACTION = 0.05   # Fraction of calibration waveform variance.
AE_RANDOM_SEED = 42

SAVE_PLOTS = True                # When True, save and close instead of showing.
PLOT_LDS = True
PLOT_LDS_DYNAMICS = True
PLOT_DPI = 160
OUTPUT_DIRECTORY = Path(__file__).resolve().parents[1] / 'agents' / 'plots_autoencoder_more'
DATA_DIRECTORY = Path(__file__).resolve().parents[1] / 'hdpgpc' / 'data' / 'mitbih'


class TiedLinearAutoencoder(torch.nn.Module):
    """Learn an orthonormal latent subspace with an affine Gaussian decoder.

    Q has orthonormal columns. Encoder: z = (y - offset) Q / scale.
    Decoder: y_hat = offset + scale Q z. This ties the two maps and lets the
    Gaussian LDS use exactly the same latent coordinates as the encoder.
    """

    def __init__(self, points: int, components: int, mean, scale: float):
        super().__init__()
        self.raw_basis = torch.nn.Parameter(torch.randn(points, components, dtype=torch.float64))
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float64).reshape(1, -1))
        self.scale = float(scale)

    def basis(self):
        return torch.linalg.qr(self.raw_basis, mode='reduced').Q

    def encode(self, waveform):
        return ((waveform - self.mean) / self.scale) @ self.basis()

    def decode(self, latent):
        return self.mean + self.scale * (latent @ self.basis().T)

    def forward(self, waveform):
        basis = self.basis()
        latent = ((waveform - self.mean) / self.scale) @ basis
        return self.mean + self.scale * (latent @ basis.T)


def train_autoencoder(values: np.ndarray, grid: np.ndarray, *, sampling: str):
    """Fit only the selected 400-beat minibatch and freeze its Gaussian decoder."""
    waveforms = np.asarray(values[:, :, 0], dtype=np.float64)
    if len(waveforms) < AE_BEATS or not 1 <= LATENT_DIM < min(AE_BEATS, waveforms.shape[1]):
        raise ValueError('Need at least AE_BEATS observations and 1 <= LATENT_DIM < waveform points')
    if sampling not in ('uniform', 'prefix'):
        raise ValueError('AE sampling must be uniform or prefix')
    rng = np.random.default_rng(AE_RANDOM_SEED)
    indices = (np.arange(AE_BEATS) if sampling == 'prefix' else
               np.sort(rng.choice(len(waveforms), size=AE_BEATS, replace=False)))
    minibatch = waveforms[indices]
    mean = minibatch.mean(axis=0)
    scale = float(np.sqrt(np.mean((minibatch - mean) ** 2)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Autoencoder calibration minibatch has zero variance')

    torch.manual_seed(AE_RANDOM_SEED)
    torch.set_num_threads(1)
    network = TiedLinearAutoencoder(waveforms.shape[1], LATENT_DIM, mean, scale)
    training = torch.from_numpy(minibatch)
    optimizer = torch.optim.Adam(network.parameters(), lr=AE_LEARNING_RATE)
    losses = []
    network.train()
    for _ in range(AE_EPOCHS):
        optimizer.zero_grad()
        reconstruction = network(training)
        loss = torch.mean(((reconstruction - training) / scale) ** 2)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    network.eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        basis = network.basis().numpy()
        codes = network.encode(training).numpy()
        reconstructed = network(training).numpy()

    latent_covariance = np.cov(codes, rowvar=False)
    latent_covariance = np.atleast_2d(latent_covariance)
    ridge = max(1e-8, 1e-5 * float(np.trace(latent_covariance)) / LATENT_DIM)
    latent_covariance = latent_covariance + ridge * np.eye(LATENT_DIM)
    residual = minibatch - reconstructed
    floor = max(1e-8, AE_NOISE_FLOOR_FRACTION * float(np.var(minibatch)))
    if AE_NOISE_STRUCTURE == 'isotropic':
        variance = max(float(np.mean(residual ** 2)), floor)
        sigma = variance * np.eye(waveforms.shape[1])
    elif AE_NOISE_STRUCTURE == 'diagonal':
        sigma = np.diag(np.maximum(residual.var(axis=0, ddof=1), floor))
    else:
        raise ValueError('AE_NOISE_STRUCTURE must be isotropic or diagonal')
    observation = AffineGaussianObservation(
        C=scale * basis, Sigma=sigma, observation_grid=grid,
        offset=mean, latent_covariance=latent_covariance,
    )
    return network, observation, indices, np.asarray(losses), residual


def make_config(decoder: AffineGaussianObservation, schedule: str) -> HDPGPCConfig:
    if schedule not in ('offline', 'online', 'online_fast'):
        raise ValueError('Unknown inference schedule')
    lds_update = LDS_UPDATE if schedule == 'offline' else 'conjugate_sequential'
    latent_kernel = (ConstantKernel(1.0) * RBF(length_scale=max(1.0, LATENT_DIM / 4))
                     + WhiteKernel(noise_level=0.01))
    base = HDPGPCConfig(
        verbose=VERBOSE,
        hdp=HDPConfig(hdp_hyp=HDP_PRESET, max_models=MAX_CLUSTERS),
        emission=GPEmissionConfig(
            kernels=[latent_kernel], fit_gp=False,
            ini_sigma=float(np.mean(np.diag(decoder.Sigma))),
            bound_sigma=(1e-8, 1e8),
        ),
        dynamics=DynamicsConfig(
            ini_gamma=INITIAL_PROCESS_NOISE,
            bound_gamma=(1e-8, 1e8),
            annealing=False, bayesian_params=True,
            free_deg_MNIV=LDS_DEGREES_OF_FREEDOM,
            lds_update=lds_update, lds_scoring='final',
        ),
        alignment=AlignmentConfig(policy='none'),
        inference=InferenceConfig(
            schedule=schedule, warp=False, hard_assignments=HARD_ASSIGNMENTS,
            birth_lds_kl_per_sqrt_point=BIRTH_LDS_KL_PER_POINT,
            n_explore_steps=10, offline_representer_selection='fixed',
            it_limit=OFFLINE_ITERATIONS if schedule == 'offline' else None,
        ),
        multi_output=MultiOutputConfig(n_outputs=1, share_gp=True, use_snr=False),
        numerics=NumericalConfig(seed=AE_RANDOM_SEED, num_threads=1,
                                 dtype='float64', deterministic=True),
    )
    return with_fixed_observation_model(
        base, decoder, sigma_policy=NoisePolicyConfig(mode=SIGMA_POLICY))


def plot_autoencoder(values, network, indices, losses, assignments, *, record):
    """Show training, decoded waveforms, and encoded full-record coordinates."""
    figure, axes = plt.subplots(1, 3, figsize=(16, 3.8), constrained_layout=True)
    axes[0].plot(np.arange(1, len(losses) + 1), losses, color='tab:blue')
    axes[0].set(xlabel='Training epoch', ylabel='Scaled MSE', title='Autoencoder calibration')
    chosen = indices[np.linspace(0, len(indices) - 1, 4, dtype=int)]
    samples = torch.as_tensor(values[chosen, :, 0], dtype=torch.float64)
    with torch.no_grad():
        reconstructions = network(samples).numpy()
    for sample_index, original, reconstructed in zip(chosen, samples.numpy(), reconstructions):
        axes[1].plot(original, alpha=0.55, linewidth=1,
                     label=f'beat {sample_index + 1}')
        axes[1].plot(reconstructed, alpha=0.9, linewidth=1.2, linestyle='--')
    axes[1].set(xlabel='Waveform point', ylabel='Amplitude', title='Solid: observed; dashed: decoded')
    axes[1].legend(fontsize=7, ncol=2)
    with torch.no_grad():
        codes = network.encode(torch.as_tensor(values[:, :, 0], dtype=torch.float64)).numpy()
    for cluster in np.unique(assignments):
        selected = assignments == cluster
        x = codes[selected, 0]
        y = (codes[selected, 1] if codes.shape[1] > 1 else
             np.flatnonzero(selected))
        axes[2].scatter(x, y, s=5, alpha=0.45, label=f'C{cluster}')
    axes[2].set(xlabel='Encoder coordinate 1',
                ylabel='Encoder coordinate 2' if codes.shape[1] > 1 else 'Beat index',
                title='Encoded beats coloured by fitted cluster')
    axes[2].legend(fontsize=7, ncol=2, markerscale=2)
    figure.suptitle(f'Record {record}: fixed autoencoder calibration')
    return figure


def run_record(record: str, *, schedule: str, sampling: str,
               save_plots: bool, output_directory: Path) -> None:
    loaded = load_mitbih_record(DATA_DIRECTORY, record)
    signal, annotations = loaded.waveforms, loaded.annotations
    values, grid = prepare_sequences(
        signal, channels=(CHANNEL,), scale_from_first=SCALE_WAVEFORMS,
        unit_grid=UNIT_GRID)
    if schedule != 'offline' and sampling != 'prefix':
        raise ValueError('Online AE calibration must use a record prefix')

    print(f'Record {record}: {len(values)} beats; AE calibration={AE_BEATS} {sampling} beats')
    if loaded.omitted_trailing_annotations:
        print('Legacy extraction omitted the last waveform window; '
              'its trailing annotation was excluded before fitting.')
    start = time.perf_counter()
    network, decoder, indices, losses, residual = train_autoencoder(values, grid, sampling=sampling)
    calibration_seconds = time.perf_counter() - start
    print(f'Autoencoder: latent dimension={LATENT_DIM}, epochs={AE_EPOCHS}, '
          f'final scaled MSE={losses[-1]:.6g}, '
          f'mean residual variance={np.mean(residual ** 2):.6g}, '
          f'calibration time={calibration_seconds:.1f} s')
    print(f'Decoder likelihood: waveform-space Gaussian, fixed C and '
          f'Sigma policy={SIGMA_POLICY}; LDS A/Gamma learned per cluster')
    probes = indices[np.linspace(0, len(indices) - 1, 8, dtype=int)]
    with torch.no_grad():
        trained_codes = network.encode(
            torch.as_tensor(values[probes, :, 0], dtype=torch.float64)).numpy()
        lds_codes = np.stack([
            decoder.encode_measurement(values[index, :, 0])[0].reshape(-1).numpy()
            for index in probes])
    difference = float(np.max(np.abs(trained_codes - lds_codes)))
    print(f'Trained encoder vs LDS latent measurement, max absolute difference '
          f'on {len(probes)} calibration beats={difference:.3g}')
    if AE_NOISE_STRUCTURE == 'isotropic' and difference > 1e-7:
        raise RuntimeError('The fitted encoder and the LDS latent measurement disagree')

    config = make_config(decoder, schedule)
    model = HDPGPC(config)
    start = time.perf_counter()
    model.fit(values, grid=grid)
    fitting_seconds = time.perf_counter() - start
    assignments = model.labels_
    print(f'HDP-GPC fitting: {fitting_seconds:.1f} s; clusters={model.n_clusters_}')
    print('Cluster sizes:', dict(sorted(Counter(assignments.tolist()).items())))
    scores = None
    if len(annotations) == len(assignments):
        scores, classes, confusion, mapping = cluster_label_statistics(assignments, annotations)
        print('In-sample label statistics: '
              f"purity={scores['purity']:.4f}, ARI={scores['adjusted_rand_index']:.4f}, "
              f"NMI={scores['normalized_mutual_information']:.4f}")
        print('Cluster majority annotations:', mapping)
        print(format_confusion_matrix(
            classes, confusion, true_name='real annotation',
            assigned_name='cluster-assigned annotation'))
    else:
        print(f'Classification statistics skipped: {len(annotations)} labels for '
              f'{len(assignments)} observations.')

    calibration_figure = plot_autoencoder(
        values, network, indices, losses, assignments, record=record)
    cluster_figures = plot_record_results(
        model, values, annotations if scores is not None else None,
        grid=grid, record=record, metrics=scores, channels=(CHANNEL,), warped=False)
    lds_figures = (plot_lds_diagnostics(
        model_lds_arrays(model), title=f'Record {record}: latent LDS and fixed decoder')
        if PLOT_LDS else [])
    dynamics_figure = (plot_lds_dynamics(
        model_lds_dynamics(model, grid=grid),
        title=f'Record {record}: decoded LDS dynamics')
        if PLOT_LDS_DYNAMICS else None)
    if save_plots:
        output_directory.mkdir(parents=True, exist_ok=True)
        calibration_path = output_directory / f'record_{record}_autoencoder.png'
        calibration_figure.savefig(calibration_path, dpi=PLOT_DPI)
        plt.close(calibration_figure)
        print(f'Saved autoencoder plot: {calibration_path}')
        for figure in cluster_figures:
            path = output_directory / f'record_{record}_clustering.png'
            figure.savefig(path, dpi=PLOT_DPI)
            plt.close(figure)
            print(f'Saved clustering plot: {path}')
        for page, figure in enumerate(lds_figures, start=1):
            path = output_directory / f'record_{record}_lds_{page:02d}.png'
            figure.savefig(path, dpi=PLOT_DPI)
            plt.close(figure)
            print(f'Saved LDS plot: {path}')
        if dynamics_figure is not None:
            path = output_directory / f'record_{record}_lds_dynamics.png'
            dynamics_figure.savefig(path, dpi=PLOT_DPI)
            plt.close(dynamics_figure)
            print(f'Saved LDS dynamics plot: {path}')
        parameter_path = output_directory / f'record_{record}_autoencoder.npz'
        with torch.no_grad():
            encoder_matrix = network.basis().numpy() / network.scale
        np.savez(parameter_path, **decoder.state_dict(), encoder_matrix=encoder_matrix,
                 calibration_indices=indices, training_loss=losses)
        print(f'Saved fixed decoder and encoder coordinates: {parameter_path}')
        configuration_path = output_directory / f'record_{record}_config.yaml'
        configuration_path.write_text(config.to_yaml(), encoding='utf-8')
        print(f'Saved HDP-GPC configuration: {configuration_path}')
    else:
        list(lds_figures)
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', nargs='+', default=RECORDS)
    parser.add_argument('--schedule', choices=('offline', 'online', 'online_fast'),
                        default=INFERENCE_MODE)
    parser.add_argument('--ae-sampling', choices=('uniform', 'prefix'), default=None,
                        help='Defaults to uniform offline and prefix online.')
    parser.add_argument('--save-plots', action=argparse.BooleanOptionalAction,
                        default=SAVE_PLOTS)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIRECTORY)
    args = parser.parse_args()
    sampling = args.ae_sampling or ('prefix' if args.schedule != 'offline' else AE_SAMPLING)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for record in args.records:
        log_path = args.output_dir / f'record_{record}_autoencoder_clustering.log'
        with log_path.open('w', encoding='utf-8', buffering=1) as log_file:
            with redirect_stdout(Tee(sys.stdout, log_file)), redirect_stderr(Tee(sys.stderr, log_file)):
                try:
                    run_record(str(record), schedule=args.schedule,
                               sampling=sampling, save_plots=args.save_plots,
                               output_directory=args.output_dir)
                    print(f'Log file: {log_path}')
                except BaseException:
                    traceback.print_exc(file=log_file)
                    raise


if __name__ == '__main__':
    main()
