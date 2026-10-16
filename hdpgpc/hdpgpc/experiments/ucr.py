"""One UCR execution path with explicit waveform/PCA/AE/Fourier/wavelet choices."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from .. import HDPGPC
from ..config import (HDPGPCConfig, HDPConfig, GPEmissionConfig, DynamicsConfig,
    ObservationConfig, MatrixPolicyConfig, NoisePolicyConfig, InferenceConfig,
    AlignmentConfig, MultiOutputConfig, NumericalConfig, with_fixed_observation_model)
from ..datasets.ucr import archive_datasets, prepare_dataset, download_archive
from ..emissions import (AffineGaussianObservation, estimate_pca_observation,
    estimate_autoencoder_observation, make_fourier_observation, make_wavelet_observation)
from ..preprocessing import normalize_samples
from ._arguments import add_common_arguments, alignment_policy, lds_update, validate_common, validate_affine, NOISE_MODES
from ._runs import ExperimentRun, source_digest, experiment_settings, save_results, append_summary, log_configuration

DEFAULT_ARCHIVE = Path.cwd() / 'agents/data/ucr/UCRArchive_2018.zip'
REPRESENTATIONS = ('waveform', 'pca', 'autoencoder', 'fourier', 'wavelet')


def build_parser(representation, description=None, defaults=None):
    if representation not in REPRESENTATIONS:
        raise ValueError(representation)
    parser = argparse.ArgumentParser(description=description)
    common_defaults = {'fit_gp': representation == 'waveform',
                       'alignment': 'dtw_gp' if representation == 'waveform' else 'none'}
    common_defaults.update(defaults or {})
    add_common_arguments(parser, defaults=common_defaults)
    parser.add_argument('--archive', type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument('--download', action='store_true')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--datasets', nargs='+', help='Names, or all for the whole archive.')
    selection.add_argument('--dataset', dest='single_dataset', help=argparse.SUPPRESS)
    parser.add_argument('--output-dir', type=Path,
                        default=DEFAULT_ARCHIVE.parents[2] / f'ucr_{representation}_results')
    parser.add_argument('--max-points', '--points', type=int, default=64)
    parser.add_argument('--max-series', type=int)
    parser.add_argument('--sigma-mode', choices=NOISE_MODES, default='cluster')
    parser.add_argument('--gamma-mode', choices=NOISE_MODES,
                        default='fixed' if representation == 'waveform' else 'cluster')
    parser.add_argument('--process-noise', '--latent-covariance-scale', type=float, default=1.)
    parser.add_argument('--observation-noise-scale', type=float, default=1.)
    parser.add_argument('--sigma-prior', type=float, default=.001)
    parser.add_argument('--gamma-prior', '--gamma', type=float, default=.001)
    if representation in ('pca', 'autoencoder'):
        parser.add_argument('--components', '--latent-dim', type=int, default=12)
        parser.add_argument('--calibration-samples', '--pca-samples', '--ae-beats', type=int, default=400)
        parser.add_argument('--noise-floor', type=float, default=1e-6)
    if representation == 'autoencoder':
        parser.add_argument('--epochs', '--ae-epochs', type=int, default=60)
        parser.add_argument('--learning-rate', type=float, default=.02)
        parser.add_argument('--noise-floor-fraction', type=float, default=.05)
    if representation == 'fourier':
        parser.add_argument('--window-size', type=int, default=16)
        parser.add_argument('--harmonics', type=int)
    if representation == 'waveform':
        parser.add_argument('--lengthscale', type=float, default=.2)
        parser.add_argument('--gp-noise', type=float, default=.01)
        for name, value, typ in (('dtw-points', 24, int), ('dtw-inducing', 10, int),
                                ('dtw-lengthscale', .2, float), ('dtw-amplitude', .1, float),
                                ('dtw-anchor-noise', .05, float)):
            parser.add_argument('--' + name, type=typ, default=value)
    parser.set_defaults(representation=representation, **(defaults or {}))
    return parser


def make_config(grid, args, decoder=None):
    """Construct the single HDPGPC recipe; no wrapper model is introduced."""
    policy = alignment_policy(args)
    dimension = len(grid) if decoder is None else decoder.latent_dim
    sigma = (args.sigma_prior * args.observation_noise_scale if decoder is None
             else float(np.diag(decoder.Sigma).mean()))
    gamma = (args.gamma_prior * args.process_noise if decoder is None
             else float(np.diag(decoder.latent_covariance).mean()))
    kernel = (ConstantKernel(1., (1e-3, 1e3)) * RBF(getattr(args, 'lengthscale', .2), (.01, 2.))
              + WhiteKernel(getattr(args, 'gp_noise', .01), (1e-5, 1.)))
    config = HDPGPCConfig(
        x_basis=grid, verbose=args.verbose,
        hdp=HDPConfig(hdp_hyp=args.hdp_preset, hmm_switch=False, max_models=args.max_clusters),
        emission=GPEmissionConfig(kernels=[kernel], fit_gp=args.fit_gp,
                                  ini_sigma=sigma, bound_sigma=(1e-8, 1e8)),
        dynamics=DynamicsConfig(
            ini_gamma=gamma, bound_gamma=(1e-8, 1e8), annealing=False,
            bayesian_params=True, free_deg_MNIV=args.lds_degrees, lds_update=lds_update(args),
            lds_scoring=args.lds_scoring, lds_step_size=args.lds_step_size),
        inference=InferenceConfig(
            schedule=args.schedule, warp=policy != 'none', hard_assignments=args.hard_assignments,
            offline_strategy=getattr(args, 'offline_strategy', 'legacy'),
            split_merge_refine_steps=getattr(args, 'split_merge_refine_steps', 1),
            split_merge_stability_patience=getattr(args, 'split_merge_stability_patience', 3),
            split_merge_score_rtol=getattr(args, 'split_merge_score_rtol', 1e-3),
            birth_lds_kl_per_dimension=args.birth_lds_kl_per_dimension,
            n_explore_steps=args.exploration_steps,
            it_limit=args.iterations if args.schedule == 'offline' else None),
        alignment=AlignmentConfig(policy=policy, x_basis_warp=grid if policy != 'none' else None,
            **{name: getattr(args, name) for name in ('dtw_points', 'dtw_inducing',
                'dtw_lengthscale', 'dtw_amplitude', 'dtw_anchor_noise') if hasattr(args, name)}),
        multi_output=MultiOutputConfig(n_outputs=1, share_gp=True, use_snr=False),
        numerics=NumericalConfig(seed=args.seed, num_threads=1, dtype='float64', deterministic=True))
    if decoder is not None:
        config = with_fixed_observation_model(config, decoder,
                    sigma_policy=NoisePolicyConfig(mode=args.sigma_mode))
        gamma_value = decoder.latent_covariance.copy()
    else:
        config = replace(config, observation=ObservationConfig(
            C_policy=MatrixPolicyConfig(mode='fixed', value=np.eye(dimension)),
            Sigma_policy=NoisePolicyConfig(mode=args.sigma_mode, value=sigma * np.eye(dimension))))
        gamma_value = gamma * np.eye(dimension)
    return replace(config, dynamics=replace(config.dynamics,
        A_policy=MatrixPolicyConfig(mode='fixed', value=np.zeros((dimension, dimension))),
        Gamma_policy=NoisePolicyConfig(mode=args.gamma_mode, value=gamma_value)))


def calibrate(train, grid, args, destination):
    """Return a fixed decoder, metadata and optional calibration graphics."""
    kind = args.representation
    metadata, figures = {}, None
    if kind == 'waveform':
        return None, metadata, figures
    if kind == 'pca':
        dim = min(args.components, len(grid) - 1, min(args.calibration_samples, len(train)) - 1)
        estimate = estimate_pca_observation(train, grid, components=dim,
            max_beats=args.calibration_samples, random_state=args.seed, noise_floor=args.noise_floor)
        decoder = AffineGaussianObservation(estimate.components,
            args.observation_noise_scale * estimate.residual_covariance, grid, estimate.mean,
            args.process_noise * np.diag(estimate.latent_variances))
        metadata = {'calibration_train_indices': estimate.beat_indices,
                    'explained_variance_ratio': estimate.explained_variance_ratio}
    elif kind == 'autoencoder':
        estimate = estimate_autoencoder_observation(train, grid, components=args.components,
            max_samples=args.calibration_samples, epochs=args.epochs, learning_rate=args.learning_rate,
            random_state=args.seed, noise_floor=args.noise_floor,
            noise_floor_fraction=args.noise_floor_fraction, process_noise_scale=args.process_noise,
            observation_noise_scale=args.observation_noise_scale)
        decoder = estimate.model
        np.savez_compressed(destination / 'training.npz', losses=estimate.losses, indices=estimate.indices)
        metadata = {'calibration_train_indices': estimate.indices, 'final_scaled_mse': estimate.losses[-1]}
        def figures():
            from ..diagnostics._calibration_plots import plot_autoencoder_calibration
            return {'autoencoder_calibration': plot_autoencoder_calibration(
                train, estimate.network, estimate.indices, estimate.losses)}
    elif kind == 'fourier':
        decoder = make_fourier_observation(grid, window_size=args.window_size,
            n_harmonics=args.harmonics, Sigma=args.sigma_prior * args.observation_noise_scale,
            latent_covariance=args.gamma_prior * args.process_noise)
    else:
        decoder = make_wavelet_observation(grid, Sigma=args.sigma_prior * args.observation_noise_scale,
                                           latent_covariance=args.gamma_prior * args.process_noise)
    np.savez_compressed(destination / 'observation_calibration.npz', **decoder.state_dict())
    metadata['latent_dimension'] = decoder.latent_dim
    return decoder, metadata, figures


def run_dataset(archive, name, pair, args, run):
    data = prepare_dataset(archive, pair, args.max_points)
    if args.representation == 'wavelet' and len(data.train[0]) & (len(data.train[0]) - 1):
        data = prepare_dataset(archive, pair, 2 ** int(np.floor(np.log2(len(data.train[0])))))
    grid = np.linspace(0., 1., data.train.shape[1])[:, None]
    train, test = data.train, data.test
    if args.normalize_samples:
        train, train_centers, train_scales = normalize_samples(train[:, :, None]); train = train[..., 0]
        test, test_centers, test_scales = normalize_samples(test[:, :, None]); test = test[..., 0]
        np.savez_compressed(run.path / 'normalization.npz', train_centers=train_centers,
                            train_scales=train_scales, test_centers=test_centers, test_scales=test_scales)
    decoder, metadata, extra_figures = calibrate(train, grid, args, run.path)
    values = np.concatenate((train, test))
    labels = np.concatenate((data.train_labels, data.test_labels))
    rng = np.random.default_rng(args.seed)
    selected = (rng.choice(len(values), args.max_series, replace=False)
                if args.max_series and args.max_series < len(values) else np.arange(len(values)))
    selected = rng.permutation(selected)
    values, labels = values[selected, :, None], labels[selected]
    config = make_config(grid, args, decoder)
    (run.path / 'config.yaml').write_text(config.to_yaml(), encoding='utf-8')
    log_configuration(config, args, source='TRAIN calibration' if decoder is not None else 'configured')
    # The same public call is used by every experiment variant.
    model = HDPGPC(config).fit(values, grid=grid, birth_diagnostic_limit=args.birth_plots)
    codes = (None if decoder is None else
             (values[..., 0] - decoder.offset) @ np.linalg.pinv(decoder.C).T)
    metadata.update(train=len(train), test=len(test), raw_points=data.raw_points,
        train_center=data.center, train_scale=data.scale,
        waveform_points=len(grid), missing_rows=data.missing_rows,
        evaluation='transductive TRAIN+TEST clustering; calibration uses TRAIN only')
    mask = selected >= len(train)
    if mask.any():
        from sklearn.metrics import adjusted_rand_score
        metadata['ari_test_transductive'] = float(adjusted_rand_score(labels[mask], model.labels_[mask]))
    return save_results(run, model, values, labels, grid, args, ordered=False,
        metadata=metadata, codes=codes, projection_name=args.representation.upper(),
        row_metadata={'source_row': selected, 'split': np.where(mask, 'TEST', 'TRAIN')},
        extra_figures=extra_figures)


def run_ucr(args, parser=None):
    import torch
    torch.set_num_threads(1)
    validate_common(args)
    if (args.max_points < 3 or (args.max_series is not None and args.max_series < 2)
            or any(not np.isfinite(x) or x <= 0 for x in (args.process_noise,
                   args.observation_noise_scale, args.sigma_prior, args.gamma_prior))):
        raise ValueError('Invalid resolution, subset or variance controls')
    if args.representation != 'waveform':
        validate_affine(args)
    if args.birth_plots and args.schedule != 'offline':
        raise ValueError('Birth diagnostic plots require offline inference')
    if args.download:
        download_archive(args.archive)
    if not args.archive.is_file():
        raise FileNotFoundError(f'{args.archive}; use --download')
    fingerprint = source_digest(args.archive)
    if args.single_dataset:
        args.datasets = [args.single_dataset]
    settings = experiment_settings(args)
    reports = []
    with ZipFile(args.archive) as archive:
        pairs = archive_datasets(archive)
        chosen = list(dict.fromkeys(sorted(pairs) if not args.datasets or args.datasets == ['all'] else args.datasets))
        if set(chosen) - set(pairs):
            raise ValueError(f'Unknown datasets: {sorted(set(chosen) - set(pairs))}')
        for name in chosen:
            run = ExperimentRun(args.output_dir, name, f'ucr_{args.representation}', settings, fingerprint)
            if args.resume and run.complete():
                report = json.loads((run.path / 'results.json').read_text(encoding='utf-8'))
            else:
                try:
                    with run:
                        report = run_dataset(archive, name, pairs[name], args, run)
                except Exception:
                    report = json.loads((run.path / 'results.json').read_text(encoding='utf-8'))
            append_summary(args.output_dir, report)
            reports.append(report)
            print(f'{name}: {report["status"]}; {run.path}')
    return reports
