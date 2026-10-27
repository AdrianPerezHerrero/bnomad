"""CAS-Colon fixed affine image experiments with native HDP-GPC trajectories."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .. import HDPGPC
from ..config import MatrixPolicyConfig, NoisePolicyConfig, with_mfa_emission
from ..datasets.cas_colon import cas_colon_annotations, load_cas_colon_frames
from ..diagnostics import (model_latent_trajectories, plot_latent_trajectories,
                           plot_image_clustering_summary)
from ..emissions import (AffineGaussianObservation, estimate_pca_observation,
                         estimate_autoencoder_observation, estimate_sfa_observation,
                         estimate_linear_prediction_observation, estimate_mfa_observation)
from . import records
from ._arguments import validate_common, validate_affine
from ._runs import ExperimentRun, experiment_settings, save_results, append_summary, log_configuration


def build_parser(description=None, defaults=None, *, representation='pca'):
    if representation not in ('pca', 'autoencoder', 'sfa', 'linear_prediction', 'mfa'):
        raise ValueError('Unknown CAS-Colon observation representation')
    selected = dict(records=['1'], components=10, calibration_samples=400,
                    sigma_prior=1., gamma_prior=.01, sigma_mode='fixed',
                    offline_strategy='split_merge', split_merge_refine_steps=1,
                    exploration_steps=5, hdp_preset='balanced',
                    fit_gp=False, alignment='none', lds_plots=False,
                    lds_dynamics_plots=False)
    if representation in ('autoencoder', 'linear_prediction'):
        selected['epochs'] = 250
    if representation == 'mfa':
        selected.update(sigma_mode='cluster_diagonal', hard_assignments=False)
    selected.update(defaults or {})
    parser = records.build_parser('cas_colon', representation, description, selected)
    parser.set_defaults(data_dir=(defaults or {}).get(
        'data_dir', Path.cwd() / 'agents/data/cas_colon/CAS-ColonDB'))
    parser.add_argument('--annotations', type=Path, help='Override Label/frame/Frame_Label.csv.')
    parser.add_argument('--image-size', nargs=2, type=int, default=(32, 32), metavar=('HEIGHT', 'WIDTH'))
    parser.add_argument('--crop', nargs=4, type=int, metavar=('LEFT', 'TOP', 'RIGHT', 'BOTTOM'),
                        help='Source-pixel crop before grayscale conversion and resizing.')
    parser.add_argument('--frame-step', type=int, default=1, help='Select one frame every N filename seconds.')
    parser.add_argument('--gap-policy', choices=('split', 'error'), default='split')
    parser.add_argument('--frames-per-cluster', type=int, default=10,
                        help='Maximum randomly sampled gallery frames per cluster (default: 10; uses --seed).')
    parser.add_argument('--trajectory-plots', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--hmm-switch', action=argparse.BooleanOptionalAction, default=representation != 'mfa')
    if representation == 'mfa':
        parser.add_argument('--dynamic', action=argparse.BooleanOptionalAction, default=False,
                            help='Learn factor LDS A/Gamma; otherwise independent N(0,I) factors.')
        parser.add_argument('--shared-emission', action=argparse.BooleanOptionalAction, default=False,
                            help='Pool learned C and diagonal Sigma across clusters; retain cluster means (offline batch only).')
        parser.add_argument('--mean-precision', type=float, default=1.)
        parser.add_argument('--loading-precision', type=float, default=1.)
    if representation == 'sfa':
        parser.add_argument('--sfa-lag', type=int, default=1,
                            help='Temporal lag in selected frame steps; pairs never cross segment boundaries.')
        parser.add_argument('--sfa-whitening-components', type=int, default=64,
                            help='Maximum pilot whitening rank before selecting slow features (default: 64).')
        parser.add_argument('--sfa-rank-tolerance', type=float, default=1e-6,
                            help='Relative singular-value cutoff for numerical covariance rank.')
    if representation == 'linear_prediction':
        parser.add_argument('--prediction-weight', type=float, default=1.,
                            help='Weight of normalized latent prediction error; zero gives a PCA control.')
        parser.add_argument('--prediction-lag', type=int, default=1,
                            help='Constant prediction lag in selected frame steps (default: 1).')
        parser.add_argument('--prediction-whitening-components', type=int, default=64,
                            help='Maximum pilot whitening rank before learning predictive features.')
        parser.add_argument('--prediction-rank-tolerance', type=float, default=1e-6)
        parser.add_argument('--prediction-ridge', type=float, default=1e-6,
                            help='Small ridge on the calibration predictor; does not regularize LDS A.')
    for action in parser._actions:
        if action.dest == 'schedule':
            action.choices = ('offline',)
            action.help = 'This first image experiment retains complete offline state histories.'
        elif action.dest == 'a_mode':
            action.choices = ('cluster', 'fixed')
            action.help = 'Learn A per cluster (default); fixed selects A=0 for an independence comparison.'
        elif action.dest == 'gamma_mode':
            action.choices = ('cluster', 'fixed')
            action.help = 'Cluster by default; fixed Gamma requires --a-mode fixed.'
    return parser


def make_config(grid, args, decoder):
    """Reuse the record affine recipe, with optional independence controls."""
    base_args = argparse.Namespace(**vars(args))
    if args.representation == 'mfa':
        base_args.c_mode = None
    config = records.make_config(grid, base_args, decoder=decoder)
    if args.representation == 'mfa':
        config = replace(config, hdp=replace(config.hdp, hmm_switch=args.hmm_switch))
        initialization = replace(decoder, Sigma=config.observation.Sigma_policy.value)
        return with_mfa_emission(config, initialization, dynamic=args.dynamic,
            mean_precision=args.mean_precision, loading_precision=args.loading_precision,
            shared=args.c_mode == 'shared')
    dynamics = config.dynamics
    if args.a_mode == 'fixed':
        dynamics = replace(dynamics, A_policy=MatrixPolicyConfig(
            mode='fixed', value=np.zeros((decoder.latent_dim, decoder.latent_dim))))
    if args.gamma_mode == 'fixed':
        if args.a_mode != 'fixed':
            raise ValueError('Fixed Gamma with this affine decoder requires --a-mode fixed')
        dynamics = replace(dynamics, Gamma_policy=NoisePolicyConfig(mode='fixed'))
    return replace(config, dynamics=dynamics,
                   hdp=replace(config.hdp, hmm_switch=args.hmm_switch))


def _validate(args):
    validate_common(args)
    validate_affine(args)
    if args.schedule != 'offline':
        raise ValueError('CAS-Colon trajectory reports currently require offline inference')
    if args.representation not in ('pca', 'autoencoder', 'sfa', 'linear_prediction', 'mfa'):
        raise ValueError('Unknown CAS-Colon observation representation')
    if args.representation == 'mfa':
        if args.shared_emission:
            if args.c_mode not in (None, 'shared') or args.sigma_mode not in ('cluster_diagonal', 'shared_diagonal'):
                raise ValueError('--shared-emission requires shared C and shared_diagonal Sigma')
            args.c_mode, args.sigma_mode = 'shared', 'shared_diagonal'
        if (args.c_mode, args.sigma_mode) not in (
                (None, 'cluster_diagonal'), ('cluster', 'cluster_diagonal'), ('shared', 'shared_diagonal')):
            raise ValueError('MFA requires cluster C/cluster_diagonal Sigma or shared C/shared_diagonal Sigma')
        if args.c_mode == 'shared' and args.lds_update not in (None, 'conjugate_batch'):
            raise ValueError('Shared MFA requires --lds-update conjugate_batch')
        if args.a_mode is not None or args.gamma_mode is not None:
            raise ValueError('Use --dynamic to learn MFA A/Gamma, or --no-dynamic for A=0/Gamma=I')
        if args.hmm_switch and not args.dynamic and args.transition_topology == 'ergodic':
            raise ValueError('Static MFA uses independent assignments; use --dynamic --hmm-switch for HMM factors')
    if args.representation == 'linear_prediction' and (
            args.prediction_lag < 1 or args.prediction_whitening_components < args.components
            or not np.isfinite(args.prediction_rank_tolerance) or not 0 < args.prediction_rank_tolerance < 1
            or not np.isfinite(args.prediction_weight) or args.prediction_weight < 0
            or not np.isfinite(args.prediction_ridge) or args.prediction_ridge <= 0
            or args.epochs < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0):
        raise ValueError('Invalid prediction weight, lag, whitening rank, tolerance, ridge or training controls')
    if args.representation == 'sfa' and (
            args.sfa_lag < 1 or args.sfa_whitening_components < args.components
            or not np.isfinite(args.sfa_rank_tolerance) or not 0 < args.sfa_rank_tolerance < 1):
        raise ValueError('SFA requires positive lag, whitening components >= components and rank tolerance in (0,1)')
    if args.representation == 'autoencoder' and (
            args.epochs < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not np.isfinite(args.noise_floor_fraction) or args.noise_floor_fraction < 0):
        raise ValueError('Autoencoder epochs/rate must be positive and noise floor fraction nonnegative')
    if (min(args.image_size) < 2 or args.frame_step < 1 or args.frames_per_cluster < 1
            or args.components < 1 or args.calibration_samples < 2
            or (args.max_series is not None and args.max_series < 2)
            or not np.isfinite(args.noise_floor) or args.noise_floor <= 0
            or any(not np.isfinite(x) or x <= 0 for x in (args.sigma_prior, args.gamma_prior))):
        raise ValueError('Image, calibration and prior controls must be finite and positive')
    if (args.normalize_samples or args.scaling != 'none' or args.unit_grid or args.max_points is not None
            or args.lds_inducing_points or args.calibrate_lds_noise_priors
            or (args.c_mode not in (None, 'cluster', 'shared') if args.representation == 'mfa'
                else args.c_mode not in (None, 'fixed')) or args.channels != [0] or args.channel is not None):
        raise ValueError('Use grayscale [0,1] frames, --image-size/--crop, fixed affine C and no extra inducing grid; '
                         'waveform/grid scaling, channels, --max-points and waveform noise calibration do not apply')
    if args.birth_plots:
        raise ValueError('Birth-warp plots do not apply to flattened images; use --trajectory-plots')
    if args.calibration_sampling not in (None, 'uniform', 'prefix'):
        raise ValueError('Image calibration sampling must be uniform or prefix')


def run_sequence(sequence, args, run, *, segment):
    values = sequence.values
    grid = np.arange(values.shape[1], dtype=float)[:, None]
    dimension = min(args.components, values.shape[1] - 1,
                    min(args.calibration_samples, len(values)) - 1)
    sampling = args.calibration_sampling or 'uniform'
    calibration_start = perf_counter()
    calibration_encoder = None
    if args.representation == 'mfa':
        decoder = estimate_mfa_observation(values, grid, components=dimension,
            max_samples=args.calibration_samples, sampling=sampling,
            random_state=args.seed, noise_floor=args.noise_floor)
        indices = (np.arange(min(len(values), args.calibration_samples)) if sampling == 'prefix'
            else np.sort(np.random.default_rng(args.seed).choice(len(values),
                size=min(len(values), args.calibration_samples), replace=False)))
        representation_name = 'Shared affine MFA' if args.c_mode == 'shared' else 'Learned affine MFA'
        calibration_metadata = dict(initialization='spectral', dynamic=args.dynamic,
            decoder_is_fixed=False, factor_coordinates_are_cluster_local=args.c_mode != 'shared')
        codes = (values[:, :, 0]-decoder.offset) @ np.linalg.pinv(decoder.C).T
        calibration_description = 'spectral prior initialization; cluster means/loadings/diagonal noise are learned'
    elif args.representation == 'linear_prediction':
        estimate = estimate_linear_prediction_observation(values, grid, components=dimension,
            max_samples=args.calibration_samples, epochs=args.epochs, learning_rate=args.learning_rate,
            prediction_weight=args.prediction_weight, prediction_ridge=args.prediction_ridge,
            whitening_components=args.prediction_whitening_components,
            rank_tolerance=args.prediction_rank_tolerance, sampling=sampling, random_state=args.seed,
            times=sequence.seconds, lag=args.prediction_lag, noise_floor=args.noise_floor)
        decoder, indices = estimate.model, estimate.indices
        calibration_encoder = estimate.encoder
        codes = estimate.transform(values)
        representation_name = 'Linear prediction'
        best_losses = estimate.losses[estimate.best_iteration]
        loss_columns = ['total', 'reconstruction', 'prediction', 'predictor_ridge']
        calibration_metadata = dict(encoder='linear_prediction', lag=args.prediction_lag,
            prediction_weight=args.prediction_weight, prediction_ridge=args.prediction_ridge,
            numerical_rank=estimate.numerical_rank, whitening_dimension=estimate.whitening_dimension,
            rank_tolerance=args.prediction_rank_tolerance, calibration_pair_count=len(estimate.pair_indices),
            calibration_lag_seconds=float(estimate.pair_durations[0]), epochs_requested=args.epochs,
            optimizer_steps=len(estimate.losses) - 1, best_iteration=estimate.best_iteration,
            learning_rate=args.learning_rate, loss_columns=loss_columns,
            pca_control_losses=estimate.losses[0], best_losses=best_losses,
            raw_encoder_is_gaussian_measurement=False)
        np.savez_compressed(run.path / 'linear_prediction_calibration.npz',
            encoder=estimate.encoder, predictor=estimate.predictor, offset=decoder.offset,
            indices=indices, pair_indices=estimate.pair_indices,
            pair_times=sequence.seconds[estimate.pair_indices], pair_durations=estimate.pair_durations,
            codes=codes, losses=estimate.losses, loss_columns=np.asarray(loss_columns),
            best_iteration=estimate.best_iteration, numerical_rank=estimate.numerical_rank,
            whitening_dimension=estimate.whitening_dimension,
            prediction_weight=args.prediction_weight, prediction_ridge=args.prediction_ridge)
        calibration_description = (f'{len(estimate.pair_indices)} temporal pairs, '
            f'objective {estimate.losses[0, 0]:.5g} -> {best_losses[0]:.5g}, '
            f'reconstruction fraction={best_losses[1]:.5g}, best step={estimate.best_iteration}')
    elif args.representation == 'sfa':
        estimate = estimate_sfa_observation(values, grid, components=dimension,
            max_samples=args.calibration_samples, sampling=sampling, random_state=args.seed,
            times=sequence.seconds, lag=args.sfa_lag,
            whitening_components=args.sfa_whitening_components,
            rank_tolerance=args.sfa_rank_tolerance, noise_floor=args.noise_floor)
        decoder, indices = estimate.model, estimate.indices
        calibration_encoder = estimate.encoder
        codes = estimate.transform(values)
        representation_name = 'Linear SFA'
        calibration_metadata = dict(encoder='linear_sfa', lag=args.sfa_lag,
            derivative_time_units='video seconds', derivative_variances=estimate.derivative_variances,
            calibration_pair_count=len(estimate.pair_indices),
            numerical_rank=estimate.numerical_rank, whitening_dimension=estimate.whitening_dimension,
            rank_tolerance=args.sfa_rank_tolerance,
            raw_encoder_is_gaussian_measurement=False)
        np.savez_compressed(run.path / 'sfa_calibration.npz', encoder=estimate.encoder,
            offset=decoder.offset, indices=indices, pair_indices=estimate.pair_indices,
            pair_times=sequence.seconds[estimate.pair_indices], codes=codes,
            derivative_variances=estimate.derivative_variances,
            numerical_rank=estimate.numerical_rank, whitening_dimension=estimate.whitening_dimension,
            lag=args.sfa_lag, rank_tolerance=args.sfa_rank_tolerance)
        calibration_description = (f'{len(estimate.pair_indices)} genuine temporal pairs, '
                                  f'derivative variances={np.round(estimate.derivative_variances, 4).tolist()}')
    elif args.representation == 'autoencoder':
        estimate = estimate_autoencoder_observation(values, grid, components=dimension,
            max_samples=args.calibration_samples, epochs=args.epochs, learning_rate=args.learning_rate,
            sampling=sampling, random_state=args.seed, noise_floor=args.noise_floor,
            noise_floor_fraction=args.noise_floor_fraction, noise_structure='diagonal')
        decoder = estimate.model
        indices = estimate.indices
        representation_name = 'Affine autoencoder'
        calibration_metadata = dict(encoder='tied_linear_orthonormal', epochs=args.epochs,
            learning_rate=args.learning_rate, final_scaled_mse=float(estimate.losses[-1]),
            noise_floor_fraction=args.noise_floor_fraction)
        np.savez_compressed(run.path / 'training.npz', losses=estimate.losses, indices=indices)
        import torch
        with torch.no_grad():
            codes = estimate.network.encode(torch.as_tensor(values[:, :, 0], dtype=torch.float64)).numpy()
        calibration_description = f'final scaled MSE={estimate.losses[-1]:.5g}'
    else:
        estimate = estimate_pca_observation(values, grid, components=dimension,
            max_beats=args.calibration_samples, sampling=sampling,
            random_state=args.seed, noise_floor=args.noise_floor)
        decoder = AffineGaussianObservation(estimate.components, estimate.residual_covariance,
                                           grid, estimate.mean, np.diag(estimate.latent_variances))
        indices = estimate.beat_indices
        representation_name = 'PCA'
        calibration_metadata = dict(explained_variance_ratio=estimate.explained_variance_ratio)
        codes = (values[:, :, 0] - decoder.offset) @ np.linalg.pinv(decoder.C).T
        calibration_description = f'explained variance={estimate.explained_variance_ratio:.4f}'
    calibration_metadata['calibration_seconds'] = perf_counter() - calibration_start
    dimension = decoder.latent_dim
    config = make_config(grid, args, decoder)
    prior_state = decoder.state_dict()
    prior_state['Sigma'] = config.observation.Sigma_policy.value
    np.savez_compressed(run.path / 'observation_calibration.npz', **prior_state)
    (run.path / 'config.yaml').write_text(config.to_yaml(), encoding='utf-8')
    if args.verbose != 'silent':
        print(f'CAS-Colon video {sequence.video}, segment {segment}: {len(values)} ordered frames, '
              f'image={tuple(args.image_size)}, {representation_name}={dimension}, {calibration_description}')
    log_configuration(config, args, source=f'frame calibration (Sigma multiplier={args.sigma_prior:g})')
    if args.representation == 'mfa' and args.verbose != 'silent':
        print(f'MFA affine row prior: mean precision={args.mean_precision:g}; '
              f'loading precision={args.loading_precision:g}; emission scope={config.observation.C_policy.mode}; '
              f'factors={"native LDS" if args.dynamic else "independent N(0,I)"}')
    model = HDPGPC(config).fit(values, grid=grid)
    if args.representation == 'mfa':
        # The usual LDS artifact keeps A/C/Gamma/Sigma; also retain learned
        # intercepts and joint conjugate posterior parameters for MFA analysis.
        np.savez_compressed(run.path / 'mfa_posteriors.npz', **{
            f'cluster_{cluster}_{name}': value.detach().cpu().numpy()
                if hasattr(value, 'detach') else np.asarray(value)
            for cluster, gp in enumerate(model._backend.gpmodels[0])
            for name, value in dict(
                offset=gp.fixed_observation_model.offset,
                joint_mean=gp.observation_params.m_mean,
                column_precision=gp.observation_params.m_r_cov,
                noise_scale=gp.observation_params.scale,
                degrees=gp.observation_params.n0).items()})
    if args.representation == 'mfa' and args.c_mode == 'shared':
        owner = model._backend.observation_policies[0]
        posterior = owner.posterior
        np.savez_compressed(run.path / 'mfa_shared_posterior.npz',
            joint_mean=posterior.m_mean.detach().cpu().numpy(),
            column_precision=posterior.m_r_cov.detach().cpu().numpy(),
            noise_scale=posterior.scale.detach().cpu().numpy(), degrees=posterior.n0,
            intercept_count=owner.intercepts)
    trajectories = model_latent_trajectories(model)
    np.savez_compressed(run.path / 'latent_trajectories.npz', times=sequence.seconds,
        **{f'cluster_{cluster}_{name}': array for cluster, path in trajectories.items()
           for name, array in path.items()})

    def summary_plotter(fitted, observations, labels, **kwargs):
        return plot_image_clustering_summary(fitted, observations, labels,
            image_shape=tuple(args.image_size), times=sequence.seconds,
            frames_per_cluster=args.frames_per_cluster, trajectories=trajectories,
            evaluation_mask=sequence.evaluation_mask, representation_name=representation_name,
            random_state=args.seed, calibration_encoder=calibration_encoder, **kwargs)

    def extra_figures():
        figures = ({'latent_trajectories': plot_latent_trajectories(model, times=sequence.seconds,
                    title=f'CAS-Colon video {sequence.video}', trajectories=trajectories,
                    min_cluster_samples=args.min_cluster_samples)}
                if args.trajectory_plots else {})
        if args.representation == 'autoencoder':
            from ..diagnostics._calibration_plots import plot_autoencoder_calibration
            figures['autoencoder_calibration'] = plot_autoencoder_calibration(
                values, estimate.network, indices, estimate.losses, image_shape=tuple(args.image_size))
        if args.representation == 'linear_prediction':
            from ..diagnostics._calibration_plots import plot_linear_prediction_calibration
            figures['linear_prediction_calibration'] = plot_linear_prediction_calibration(
                values, estimate, image_shape=tuple(args.image_size))
        return figures

    metadata = dict(video=sequence.video, segment=segment, image_shape=list(args.image_size),
        grayscale=True, pixel_range=[0, 1], crop=args.crop, frame_step=args.frame_step,
        time_units='seconds encoded in official frame filenames (one-second bins)',
        representation=args.representation, observation_scope=config.observation.C_policy.mode, calibration_indices=indices,
        calibration_sampling=sampling, latent_dimension=dimension, **calibration_metadata,
        sigma_prior_multiplier=args.sigma_prior, sigma_prior_diagonal_mean=config.emission.ini_sigma,
        ambiguous_seconds=sequence.seconds[~sequence.evaluation_mask],
        evaluation_frames=int(sequence.evaluation_mask.sum()),
        evaluation='in-sample ordered clustering; decoder calibration uses this selected segment without labels',
        dynamics_step='between successive observations included in the same cluster, not elapsed wall-clock time')
    return save_results(run, model, values, sequence.annotations, grid, args, ordered=True,
        metadata=metadata, codes=codes, projection_name=representation_name,
        row_metadata={'second': sequence.seconds, 'filename': sequence.filenames,
                      'annotation_options': ['|'.join(options) for options in sequence.annotation_options],
                      'source_variants': ['|'.join(names) for names in sequence.source_variants],
                      'evaluation_included': sequence.evaluation_mask},
        summary_plotter=summary_plotter, extra_figures=extra_figures,
        evaluation_mask=sequence.evaluation_mask, equal_duration_baseline=True)


def run_cas_colon(args, parser=None):
    import torch
    torch.set_num_threads(1)
    _validate(args)
    annotation_path, index = cas_colon_annotations(args.data_dir, args.annotations)
    videos = sorted(index, key=int) if args.records == ['all'] else [str(int(video)) for video in args.records]
    settings = experiment_settings(args)
    reports = []
    for video in dict.fromkeys(videos):
        sequences = load_cas_colon_frames(args.data_dir, video, frame_records=index,
            image_size=tuple(args.image_size), crop=args.crop, frame_step=args.frame_step,
            max_frames=args.max_series, gap_policy=args.gap_policy)
        for segment, sequence in enumerate(sequences):
            if len(sequence.images) < 2:
                if args.verbose != 'silent':
                    print(f'CAS-Colon video {video}: skipped isolated frame at {sequence.seconds[0]}s')
                continue
            digest = hashlib.sha256(annotation_path.read_bytes())
            digest.update(sequence.images.tobytes())
            digest.update(sequence.seconds.tobytes())
            digest.update('\n'.join(sequence.filenames).encode())
            run = ExperimentRun(args.output_dir, f'video_{video}_segment_{segment:02d}',
                                f'cas_colon_{args.representation}', settings, digest.hexdigest())
            if args.resume and run.complete():
                report = json.loads((run.path / 'results.json').read_text(encoding='utf-8'))
            else:
                try:
                    with run:
                        report = run_sequence(sequence, args, run, segment=segment)
                except Exception:
                    report = json.loads((run.path / 'results.json').read_text(encoding='utf-8'))
            append_summary(args.output_dir, report)
            reports.append(report)
            print(f'{run.dataset}: {report["status"]}; {run.path}')
    if not reports:
        raise ValueError('No contiguous sequence has at least two selected frames')
    return reports
