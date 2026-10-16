"""Comparable MIT-BIH and Daphnet record experiments using ordinary HDPGPC."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from .. import HDPGPC
from ..config import (HDPGPCConfig, HDPConfig, GPEmissionConfig, DynamicsConfig,
    MatrixPolicyConfig, NoisePolicyConfig, AlignmentConfig,
    InferenceConfig, MultiOutputConfig, NumericalConfig, LDS_PRESETS, with_lds_preset,
    with_fixed_observation_model)
from ..datasets import (load_mitbih_record, load_daphnet_record, window_daphnet_record,
                         download_daphnet, daphnet_records, DAPHNET_CHANNELS)
from ..emissions import estimate_pca_observation, estimate_autoencoder_observation, AffineGaussianObservation
from ..dynamics import estimate_lds_noise_priors
from ..preprocessing import prepare_sequences, normalize_samples
from ._arguments import add_common_arguments, alignment_policy, lds_update, validate_common, validate_affine, NOISE_MODES
from ._runs import ExperimentRun, source_digest, experiment_settings, save_results, append_summary, log_configuration

ROOT = Path.cwd()


def build_parser(dataset='mitbih', representation='waveform', description=None, defaults=None):
    parser = argparse.ArgumentParser(description=description)
    selected = {'hard_assignments': True, 'hdp_preset': 'more',
                'fit_gp': representation == 'waveform'}
    if dataset == 'mitbih' and representation == 'waveform':
        selected['alignment'] = 'dtw_gp'
    selected.update(defaults or {})
    add_common_arguments(parser, defaults=selected, ordered=True)
    parser.add_argument('--records', nargs='+', default=['S02R01'] if dataset == 'daphnet' else ['100'])
    parser.add_argument('--output-dir', '--plot-dir', type=Path,
                        default=ROOT / 'agents' / f'{dataset}_{representation}_runs')
    parser.add_argument('--max-series', '--max-windows', type=int)
    parser.add_argument('--max-points', '--points', type=int, default=64 if dataset == 'daphnet' else None)
    parser.add_argument('--scaling', choices=('none', 'first_window'),
                        default='first_window' if dataset == 'daphnet' else 'none')
    parser.add_argument('--scale-waveforms', dest='scaling', action='store_const', const='first_window', help=argparse.SUPPRESS)
    parser.add_argument('--no-scale-waveforms', dest='scaling', action='store_const', const='none', help=argparse.SUPPRESS)
    parser.add_argument('--unit-grid', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--lds-preset', choices=LDS_PRESETS, default='only_plastic')
    if representation == 'waveform':
        parser.add_argument('--representation', choices=('waveform', 'pca'), default='waveform')
    parser.add_argument('--a-mode', choices=('cluster', 'shrinkage', 'shared', 'fixed'))
    parser.add_argument('--c-mode', choices=('cluster', 'shrinkage', 'shared', 'fixed'))
    parser.add_argument('--gamma-mode', choices=NOISE_MODES)
    parser.add_argument('--shrink-strength', type=float, default=10.)
    parser.add_argument('--lds-inducing-points', type=int, default=0)
    parser.add_argument('--sigma-prior', type=float, default=.1 if dataset == 'daphnet' else 50.)
    parser.add_argument('--gamma-prior', type=float, default=.1 if dataset == 'daphnet' else 10.)
    parser.add_argument('--lengthscale', type=float, default=.15 if dataset == 'daphnet' else 1.)
    parser.add_argument('--gp-noise', type=float, default=.01 if dataset == 'daphnet' else 1.)
    parser.add_argument('--lengthscale-bounds', nargs=2, type=float, default=(.02, 2.) if dataset == 'daphnet' else (.1, 10.))
    parser.add_argument('--gp-noise-bounds', nargs=2, type=float, default=(1e-5, 1.) if dataset == 'daphnet' else (1e-4, 1e8))
    parser.add_argument('--gp-outputscale', type=float, default=1.)
    parser.add_argument('--gp-outputscale-bounds', nargs=2, type=float, default=(.001, 1e3) if dataset == 'daphnet' else (.001, 1e4))
    noise_bounds = (1e-6, 1e3) if dataset == 'daphnet' else ((.001, 1e8) if representation == 'waveform' else (1e-8, 1e8))
    parser.add_argument('--sigma-bounds', nargs=2, type=float, default=noise_bounds)
    parser.add_argument('--gamma-bounds', nargs=2, type=float, default=noise_bounds)
    parser.add_argument('--calibration-samples', '--ae-beats', '--pca-samples', type=int, default=400)
    parser.add_argument('--calibration-sampling', '--ae-sampling', choices=('uniform', 'prefix'))
    parser.add_argument('--components', '--latent-dim', '--pca-components', type=int, default=12)
    parser.add_argument('--epochs', '--ae-epochs', type=int, default=60)
    parser.add_argument('--learning-rate', type=float, default=.02)
    parser.add_argument('--noise-floor', type=float, default=1e-6)
    parser.add_argument('--noise-floor-fraction', type=float, default=.05)
    parser.add_argument('--sigma-mode', choices=NOISE_MODES)
    parser.add_argument('--calibrate-lds-noise-priors', action=argparse.BooleanOptionalAction, default=False)
    if dataset == 'daphnet':
        parser.add_argument('--source', type=Path, default=ROOT / 'agents/data/daphnet/daphnet.zip')
        parser.add_argument('--download', action='store_true')
        parser.add_argument('--list-records', action='store_true')
        parser.add_argument('--segments', choices=('longest', 'all'), default='longest')
        parser.add_argument('--channel', choices=DAPHNET_CHANNELS, default='ankle_vertical')
        parser.add_argument('--window-seconds', type=float, default=2.)
        parser.add_argument('--stride-seconds', type=float)
    else:
        parser.add_argument('--data-dir', type=Path, default=ROOT / 'hdpgpc/data/mitbih')
        parser.add_argument('--channels', nargs='+', type=int, default=[0])
        parser.add_argument('--channel', type=int, help=argparse.SUPPRESS)
    parser.set_defaults(dataset=dataset, representation=representation, **(defaults or {}))
    return parser


def make_config(grid, args, outputs=1, decoder=None, noise_scales=None):
    policy = alignment_policy(args)
    sigma, gamma = noise_scales or (args.sigma_prior, args.gamma_prior)
    basis = grid
    if args.lds_inducing_points:
        if not 1 <= args.lds_inducing_points <= len(grid):
            raise ValueError('LDS inducing-point count must be between one and waveform dimension')
        basis = grid[np.linspace(0, len(grid) - 1, args.lds_inducing_points, dtype=int)]
    config = HDPGPCConfig(
        x_basis=basis, verbose=args.verbose,
        hdp=HDPConfig(hdp_hyp=args.hdp_preset, hmm_switch=True, max_models=args.max_clusters),
        emission=GPEmissionConfig(kernels=[ConstantKernel(args.gp_outputscale, tuple(args.gp_outputscale_bounds))
            * RBF(args.lengthscale, tuple(args.lengthscale_bounds)) + WhiteKernel(args.gp_noise, tuple(args.gp_noise_bounds))],
            fit_gp=args.fit_gp, ini_lengthscale=args.lengthscale, bound_lengthscale=tuple(args.lengthscale_bounds),
            ini_sigma=sigma, bound_sigma=tuple(args.sigma_bounds), inducing_points=len(basis) < len(grid)),
        dynamics=DynamicsConfig(ini_gamma=gamma, bound_gamma=tuple(args.gamma_bounds),
            annealing=False, bayesian_params=True, free_deg_MNIV=args.lds_degrees,
            lds_update=lds_update(args), lds_scoring=args.lds_scoring, lds_step_size=args.lds_step_size),
        alignment=AlignmentConfig(policy=policy, x_basis_warp=grid if policy != 'none' else None),
        inference=InferenceConfig(schedule=args.schedule, warp=policy != 'none',
            offline_strategy=getattr(args, 'offline_strategy', 'legacy'),
            split_merge_refine_steps=getattr(args, 'split_merge_refine_steps', 3),
            split_merge_stability_patience=getattr(args, 'split_merge_stability_patience', 3),
            split_merge_score_rtol=getattr(args, 'split_merge_score_rtol', 1e-3),
            hard_assignments=args.hard_assignments, n_explore_steps=args.exploration_steps,
            birth_lds_kl_per_dimension=args.birth_lds_kl_per_dimension,
            offline_representer_selection=args.offline_representer_selection,
            it_limit=args.iterations if args.schedule == 'offline' else None),
        multi_output=MultiOutputConfig(n_outputs=outputs, share_gp=True, use_snr=False),
        numerics=NumericalConfig(seed=args.seed, num_threads=1, dtype='float64', deterministic=True))
    if decoder is not None:
        return with_fixed_observation_model(config, decoder,
                    sigma_policy=NoisePolicyConfig(mode=args.sigma_mode or 'fixed'))
    config = with_lds_preset(config, args.lds_preset, shrink_strength=args.shrink_strength)
    dynamics, observation = config.dynamics, config.observation
    for name, mode in (('A', args.a_mode), ('C', args.c_mode)):
        if mode:
            policy = MatrixPolicyConfig(mode=mode, strength=args.shrink_strength if mode == 'shrinkage' else 1.)
            if name == 'A':
                dynamics = replace(dynamics, A_policy=policy)
            else:
                observation = replace(observation, C_policy=policy)
    if args.gamma_mode:
        dynamics = replace(dynamics, Gamma_policy=NoisePolicyConfig(mode=args.gamma_mode))
    if args.sigma_mode:
        observation = replace(observation, Sigma_policy=NoisePolicyConfig(mode=args.sigma_mode))
    return replace(config, dynamics=dynamics, observation=observation)


def run_record(values, labels, grid, args, run, metadata, row_metadata, colors=None):
    """The same prepare -> calibrate -> configure -> fit -> report stages as UCR."""
    if args.max_series:
        values, labels = values[:args.max_series], labels[:args.max_series]
        row_metadata = {key: value[:args.max_series] for key, value in row_metadata.items()}
    if len(values) < 2:
        raise ValueError('At least two observations are required')
    if args.max_points and args.dataset != 'daphnet' and args.max_points < values.shape[1]:
        from scipy.signal import resample_poly
        from math import gcd
        divisor = gcd(args.max_points, values.shape[1])
        old_grid = grid
        values = resample_poly(values, args.max_points // divisor, values.shape[1] // divisor, axis=1, padtype='line')
        grid = np.linspace(old_grid[0, 0], old_grid[-1, 0], args.max_points)[:, None]
    if args.normalize_samples:
        values, centers, scales = normalize_samples(values)
        np.savez_compressed(run.path / 'normalization.npz', centers=centers, scales=scales)
    elif args.scaling == 'first_window':
        centers, scales = values[0].mean(0), values[0].std(0)
        values, _ = prepare_sequences(values, channels=tuple(range(values.shape[2])), scale_from_first=True)
        np.savez_compressed(run.path / 'normalization.npz', centers=centers, scales=scales)
    decoder, codes, extra_figures = None, None, None
    if args.representation != 'waveform':
        if values.shape[2] != 1 or alignment_policy(args) != 'none' or args.lds_inducing_points:
            raise ValueError('Fixed affine records require one output, no warp and no inducing grid')
        validate_affine(args)
        if args.a_mode or args.c_mode or args.gamma_mode:
            raise ValueError('Calibrated record emissions fix C and learn A/Gamma per cluster; select only --sigma-mode')
        sampling = args.calibration_sampling or ('uniform' if args.schedule == 'offline' else 'prefix')
        if args.schedule != 'offline' and sampling != 'prefix':
            raise ValueError('Ordered online calibration must use a prefix (replayed during fitting)')
        if args.representation == 'autoencoder':
            estimate = estimate_autoencoder_observation(values, grid, components=args.components,
                max_samples=args.calibration_samples, epochs=args.epochs, learning_rate=args.learning_rate,
                sampling=sampling, random_state=args.seed, noise_floor=args.noise_floor,
                noise_floor_fraction=args.noise_floor_fraction)
            decoder = estimate.model
            np.savez_compressed(run.path / 'training.npz', losses=estimate.losses, indices=estimate.indices)
            def extra_figures():
                from ..diagnostics._calibration_plots import plot_autoencoder_calibration
                return {'autoencoder_calibration': plot_autoencoder_calibration(
                    values, estimate.network, estimate.indices, estimate.losses)}
            metadata.update(calibration_indices=estimate.indices, sampling=sampling)
        else:
            dimension = min(args.components, len(grid) - 1, min(args.calibration_samples, len(values)) - 1)
            estimate = estimate_pca_observation(values, grid, components=dimension,
                max_beats=args.calibration_samples, random_state=args.seed, sampling=sampling,
                noise_floor=args.noise_floor)
            decoder = AffineGaussianObservation(estimate.components, estimate.residual_covariance,
                                               grid, estimate.mean, np.diag(estimate.latent_variances))
            metadata.update(calibration_indices=estimate.beat_indices, sampling=sampling,
                            explained_variance_ratio=estimate.explained_variance_ratio)
        np.savez_compressed(run.path / 'observation_calibration.npz', **decoder.state_dict())
        codes = (values[..., 0] - decoder.offset) @ np.linalg.pinv(decoder.C).T
    noise_scales = None
    if args.calibrate_lds_noise_priors:
        if args.schedule != 'offline' or decoder is not None or args.lds_preset == 'static_mixture':
            raise ValueError('LDS noise calibration requires an offline dynamic waveform model')
        estimate = estimate_lds_noise_priors(values, max_beats=args.calibration_samples,
            segment_length=min(20, args.calibration_samples), random_state=args.seed,
            require_positive_components=False)
        pooled = .5 * (estimate.observation + estimate.process)
        if not np.isfinite(pooled) or pooled <= 0:
            raise ValueError('Calibrated pooled LDS variance is not positive')
        noise_scales = pooled, pooled
        metadata['calibrated_noise_prior'] = pooled
    config = make_config(grid, args, values.shape[2], decoder, noise_scales)
    (run.path / 'config.yaml').write_text(config.to_yaml(), encoding='utf-8')
    log_configuration(config, args, source='calibration' if noise_scales or decoder is not None else 'configured')
    model = HDPGPC(config).fit(values, grid=grid, birth_diagnostic_limit=args.birth_plots)
    metadata.update(evaluation='in-sample ordered clustering; calibration prefixes are replayed',
                    waveform_points=len(grid))
    return save_results(run, model, values, labels, grid, args, ordered=True,
        metadata=metadata, row_metadata=row_metadata, codes=codes,
        projection_name=args.representation.upper(), annotation_colors=colors,
        extra_figures=extra_figures, positive_label=2 if args.dataset == 'daphnet' else None)


def run_records(args, parser=None):
    import torch
    torch.set_num_threads(1)
    validate_common(args)
    if (args.max_series is not None and args.max_series < 2) or (args.max_points is not None and args.max_points < 2):
        raise ValueError('At least two observations/points are required')
    if any(not np.isfinite(x) or x <= 0 for x in (args.sigma_prior, args.gamma_prior, args.lengthscale, args.gp_noise)):
        raise ValueError('Prior and GP scales must be positive and finite')
    if args.birth_plots and args.schedule != 'offline':
        raise ValueError('Birth diagnostic plots require offline inference')
    for bounds, initial in ((args.lengthscale_bounds, args.lengthscale), (args.gp_noise_bounds, args.gp_noise),
                            (args.gp_outputscale_bounds, args.gp_outputscale),
                            (args.sigma_bounds, args.sigma_prior), (args.gamma_bounds, args.gamma_prior)):
        if not 0 < bounds[0] <= initial <= bounds[1] or not np.isfinite(bounds).all():
            raise ValueError('Initial GP and noise values must lie within finite positive bounds')
    settings = experiment_settings(args)
    reports = []
    for name, values, labels, grid, metadata, rows, fingerprint, colors in _record_jobs(args):
        run = ExperimentRun(args.output_dir, name, f'{args.dataset}_{args.representation}', settings, fingerprint)
        if args.resume and run.complete():
            report = json.loads((run.path / 'results.json').read_text(encoding='utf-8'))
        else:
            try:
                with run:
                    report = run_record(values, labels, grid, args, run, metadata, rows, colors)
            except Exception:
                report = json.loads((run.path / 'results.json').read_text(encoding='utf-8'))
        append_summary(args.output_dir, report)
        reports.append(report)
        print(f'{name}: {report["status"]}; {run.path}')
    return reports


def _record_jobs(args):
    """Load one record at a time so archive runs do not retain all records."""
    if args.dataset == 'daphnet':
        if args.download:
            download_daphnet(args.source)
        names = daphnet_records(args.source)
        if args.list_records:
            print(' '.join(names)); return
        selected = names if args.records == ['all'] else [name.upper().removesuffix('.TXT') for name in args.records]
        fingerprint = source_digest(args.source)
        for name in dict.fromkeys(selected):
            record = load_daphnet_record(args.source, name)
            segments = window_daphnet_record(record, channel=args.channel, window_seconds=args.window_seconds,
                stride_seconds=args.stride_seconds, points=args.max_points)
            segments = [segment for segment in segments if len(segment.values) >= 2]
            if not segments:
                raise ValueError(f'{name}: no eligible contiguous segment')
            if args.segments == 'longest':
                segments = [max(segments, key=lambda segment: len(segment.values))]
            for segment in segments:
                yield (f'{name}_segment_{segment.segment:02d}', segment.values, segment.annotations,
                    segment.grid, {'record': name, 'segment': segment.segment,
                        'channel': args.channel, 'window_seconds': args.window_seconds,
                        'source_sampling_hz': 64},
                    {'source_row': segment.start_rows, 'start_seconds': segment.start_seconds,
                     'end_seconds': segment.end_seconds, 'freeze_fraction': segment.freeze_fraction},
                    fingerprint, {1: 'tab:blue', 2: 'tab:red'})
    else:
        names = args.records
        if names == ['all']:
            names = sorted(path.stem for path in args.data_dir.glob('*.npy') if path.stem.isdigit())
        channels = [args.channel] if args.channel is not None else args.channels
        if len(channels) > 1 and args.schedule != 'offline':
            raise ValueError('The characterized online path requires one channel')
        for name in dict.fromkeys(names):
            record = load_mitbih_record(args.data_dir, name)
            values, grid = prepare_sequences(record.waveforms, channels=channels, unit_grid=args.unit_grid)
            fingerprint = hashlib.sha256((source_digest(args.data_dir / f'{name}.npy')
                + source_digest(args.data_dir / f'{name}_labels.npy')).encode()).hexdigest()
            yield (name, values, record.annotations, grid,
                {'channels': channels, 'omitted_trailing_annotations': record.omitted_trailing_annotations},
                {'source_row': np.arange(len(values))}, fingerprint, None)
