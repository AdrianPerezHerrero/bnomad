"""Cluster known-LDS synthetic images through the ordinary HDPGPC API."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from .. import HDPGPC
from ..config import (HDPGPCConfig, HDPConfig, GPEmissionConfig, DynamicsConfig,
                      MatrixPolicyConfig, NoisePolicyConfig, AlignmentConfig,
                      InferenceConfig, MultiOutputConfig, NumericalConfig,
                      with_fixed_observation_model)
from ..datasets import generate_switching_lds_images, load_switching_lds_images
from ..diagnostics import (model_latent_trajectories, plot_latent_trajectories,
                           plot_image_clustering_summary)
from ..emissions import AffineGaussianObservation, estimate_pca_observation
from ._arguments import add_common_arguments, validate_common, validate_affine, lds_update, NOISE_MODES
from ._runs import (ExperimentRun, experiment_settings, save_results, append_summary,
                    log_configuration, source_digest, write_json)


def build_parser(description=None):
    parser = argparse.ArgumentParser(description=description or __doc__)
    add_common_arguments(parser, ordered=True, defaults=dict(
        schedule='offline', offline_strategy='split_merge', offline_representer_selection='fixed',
        split_merge_refine_steps=3, iterations=40, max_clusters=8,
        hdp_preset='balanced', lds_degrees=5, exploration_steps=10,
        hard_assignments=True, fit_gp=False, lds_dynamics_plots=False, warp_plots=False))
    # Image trajectories need complete state histories. Do not advertise online
    # reporting until that contract is supported by the image diagnostics.
    for action in parser._actions:
        if action.dest == 'schedule':
            action.choices = ('offline',)
        elif action.dest == 'split_merge_refine_steps':
            action.help = f'Local refinement passes per split/merge branch (default: {action.default}).'
        elif action.dest == 'lds_update':
            action.choices = ('conjugate_batch', 'conjugate_sequential', 'natural_gradient')
        elif action.dest == 'lds_scoring':
            action.choices = ('final',)
    parser.add_argument('--input', type=Path, help='Saved generator dataset.npz; otherwise generate in memory.')
    parser.add_argument('--output-dir', type=Path, default=Path('agents/synthetic_lds_clustering_runs'))
    parser.add_argument('--representation', choices=('oracle', 'pca'), default='oracle',
                        help='Known fixed affine decoder, or PCA learned from unlabeled calibration frames.')
    parser.add_argument('--components', type=int,
                        help='PCA components; default matches the saved/generated dimension (6 for fields/tissue, 4 for dynamics_only).')
    parser.add_argument('--calibration-samples', type=int, default=400)
    parser.add_argument('--calibration-sampling', choices=('uniform', 'prefix'), default='uniform')
    parser.add_argument('--noise-floor', type=float, default=1e-8, help='PCA residual variance floor.')
    parser.add_argument('--sigma-prior', type=float, default=1., help='Multiplier of known/calibrated observation covariance.')
    parser.add_argument('--gamma-prior', type=float,
                        help='Initial process variance; default is the true average variance in the decoder coordinates.')
    parser.add_argument('--sigma-mode', choices=NOISE_MODES, default='fixed')
    parser.add_argument('--a-mode', choices=('cluster', 'fixed'), default='cluster',
                        help='Learn cluster A, or fix A=0 for an independence control.')
    parser.add_argument('--gamma-mode', choices=('cluster', 'fixed'), default='cluster',
                        help='Fixed Gamma currently requires fixed A for affine observations.')
    parser.add_argument('--hmm-switch', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--shuffle', action=argparse.BooleanOptionalAction, default=False,
                        help='Permute frames as a temporal-order control; keep labels aligned for evaluation.')
    parser.add_argument('--max-series', type=int, help='Use an ordered prefix before optional shuffling.')
    parser.add_argument('--frames-per-cluster', type=int, default=6)
    parser.add_argument('--trajectory-plots', action=argparse.BooleanOptionalAction, default=True)
    generation = parser.add_argument_group('generation (only when --input is omitted)')
    generation.add_argument('--scenario', choices=('physical_fields', 'appearance_dynamics', 'dynamics_only'), default='physical_fields')
    generation.add_argument('--state-mode', choices=('switching', 'per_regime'), default='per_regime')
    generation.add_argument('--data-seed', type=int, default=42)
    generation.add_argument('--image-size', type=int, nargs=2, default=(32, 32))
    generation.add_argument('--regime-order', type=int, nargs='+', default=(0, 1, 2, 0, 2, 1))
    generation.add_argument('--segment-length', type=int, default=120)
    generation.add_argument('--rho', type=float, default=.995)
    generation.add_argument('--contrast', type=float, help='Oscillator pixel SD scale; default depends on scenario.')
    generation.add_argument('--appearance-strength', type=float, default=1.)
    generation.add_argument('--appearance-drift', type=float, help='Default .04 for fields, .002 for tissue.')
    generation.add_argument('--observation-std', type=float, default=.01)
    return parser


def make_config(args, decoder, gamma):
    """Use existing matrix policies, fixed decoder, inference and HDP settings."""
    sigma = float(np.diag(decoder.Sigma).mean())
    config = HDPGPCConfig(
        x_basis=decoder.grid, verbose=args.verbose,
        hdp=HDPConfig(hdp_hyp=args.hdp_preset, hmm_switch=args.hmm_switch, max_models=args.max_clusters,
                      transition_topology=getattr(args, 'transition_topology', 'ergodic')),
        # The native backend still constructs a GP container for affine LDS
        # emissions. Its kernel initialization must be complete even though
        # fit_gp=False and the observation likelihood uses the fixed decoder.
        emission=GPEmissionConfig(fit_gp=False, ini_lengthscale=1., bound_lengthscale=(.1, 10.),
            ini_outputscale=1., ini_sigma=sigma, bound_sigma=(1e-12, 1e8)),
        dynamics=DynamicsConfig(ini_gamma=gamma, bound_gamma=(1e-12, 1e8), annealing=False,
            bayesian_params=True, free_deg_MNIV=args.lds_degrees, lds_update=lds_update(args),
            lds_scoring=args.lds_scoring, lds_step_size=args.lds_step_size),
        alignment=AlignmentConfig(policy='none'),
        inference=InferenceConfig(schedule='offline', warp=False,
            offline_strategy=args.offline_strategy, it_limit=args.iterations,
            hard_assignments=args.hard_assignments, n_explore_steps=args.exploration_steps,
            offline_representer_selection=args.offline_representer_selection,
            split_merge_refine_steps=args.split_merge_refine_steps,
            split_merge_stability_patience=args.split_merge_stability_patience,
            split_merge_score_rtol=args.split_merge_score_rtol,
            birth_lds_kl_per_dimension=args.birth_lds_kl_per_dimension),
        multi_output=MultiOutputConfig(n_outputs=1, share_gp=True, use_snr=False),
        numerics=NumericalConfig(seed=args.seed, num_threads=1, dtype='float64', deterministic=True))
    config = with_fixed_observation_model(config, decoder,
                                         sigma_policy=NoisePolicyConfig(mode=args.sigma_mode))
    dynamics = replace(config.dynamics, Gamma_policy=NoisePolicyConfig(mode=args.gamma_mode))
    if args.a_mode == 'fixed':
        dynamics = replace(dynamics, A_policy=MatrixPolicyConfig(
            mode='fixed', value=np.zeros((decoder.latent_dim, decoder.latent_dim))))
    return replace(config, dynamics=dynamics)


def _validate(args):
    validate_common(args)
    validate_affine(args)
    if args.schedule != 'offline':
        raise ValueError('Synthetic image trajectory reports require offline inference')
    if args.gamma_mode == 'fixed' and args.a_mode != 'fixed':
        raise ValueError('The current affine path requires --a-mode fixed when --gamma-mode fixed')
    if (args.normalize_samples or args.birth_plots or args.warp_plots):
        raise ValueError('Do not normalize or 1D-warp flattened images; disable waveform birth/warp plots')
    if ((args.components is not None and args.components < 1)
            or args.calibration_samples < 2 or args.frames_per_cluster < 1
            or (args.max_series is not None and args.max_series < 2)
            or not 0 <= args.data_seed < 2**32
            or not np.isfinite(args.sigma_prior) or args.sigma_prior <= 0
            or not np.isfinite(args.noise_floor) or args.noise_floor <= 0
            or (args.gamma_prior is not None and (not np.isfinite(args.gamma_prior) or args.gamma_prior <= 0))):
        raise ValueError('Invalid dimensions, calibration budget, seed or covariance scales')


def _state_comparison(model, data, indices, decoder, paths, representation):
    """Evaluate states in true coordinates, using only MAP-owned retained states.

    Truth is used for diagnostics only; it never selects assignments or moves.
    PCA states are projected into the true subspace for an interpretable error.
    Exact parameter errors are reported only for the identical oracle basis.
    """
    inferred = np.full((len(indices), data.states.shape[1]), np.nan)
    transform = np.linalg.pinv(data.C) @ decoder.C
    shift = np.linalg.pinv(data.C) @ (decoder.offset - data.offset)
    labels = np.asarray(model.labels_, dtype=int)
    parameters = []
    for cluster, path in paths.items():
        positions = path['indices']
        owned = labels[positions] == cluster
        positions = positions[owned]
        inferred[positions] = path['means'][owned] @ transform.T + shift
        if not len(positions):
            continue
        truths = data.labels[indices[positions]]
        k = int(np.bincount(truths, minlength=len(data.regime_names)).argmax())
        gp = model._backend.gpmodels[0][cluster]
        A = gp.A[-1].detach().cpu().numpy()
        Gamma = gp.Gamma[-1].detach().cpu().numpy()
        record = dict(cluster=int(cluster), frames=len(positions), majority_regime=k,
                      majority_fraction=float(np.mean(truths == k)), spectral_radius=float(max(abs(np.linalg.eigvals(A)))))
        if representation == 'oracle':
            record.update(A_relative_error=float(np.linalg.norm(A-data.A[k]) / np.linalg.norm(data.A[k])),
                          Gamma_relative_error=float(np.linalg.norm(Gamma-data.Gamma[k]) / np.linalg.norm(data.Gamma[k])))
        parameters.append(record)
    available = np.isfinite(inferred).all(1)
    truth = data.states[indices]
    scores = dict(evaluated_frames=int(available.sum()),
                  latent_rmse=float(np.sqrt(np.mean((inferred[available]-truth[available])**2))) if available.any() else None,
                  zero_state_rmse=float(np.sqrt(np.mean(truth[available]**2))) if available.any() else None,
                  parameter_comparisons=parameters,
                  note='Smoothed in-sample states; majority mapping is diagnostic. PCA errors use projection into true coordinates.')
    return inferred, scores


def run_synthetic_clustering(args):
    import torch
    torch.set_num_threads(1)
    _validate(args)
    data = (load_switching_lds_images(args.input) if args.input else
            generate_switching_lds_images(image_size=tuple(args.image_size),
                regime_order=tuple(args.regime_order), segment_lengths=args.segment_length,
                rho=args.rho, contrast=args.contrast, observation_std=args.observation_std,
                state_mode=args.state_mode, seed=args.data_seed, scenario=args.scenario,
                appearance_strength=args.appearance_strength, appearance_drift=args.appearance_drift))
    indices = np.arange(min(len(data.images), args.max_series or len(data.images)))
    if args.shuffle:
        indices = np.random.default_rng(args.seed).permutation(indices)
    values, truth_labels = data.values[indices], data.labels[indices]
    labels = np.asarray(data.regime_names)[truth_labels]
    fingerprint = source_digest(args.input) if args.input else hashlib.sha256(
        data.images.tobytes() + data.C.tobytes() + data.A.tobytes() + data.labels.tobytes() +
        data.state_mode.encode()+data.scenario.encode()+data.initial_state_means.tobytes()+
        data.initial_state_covariances.tobytes()).hexdigest()
    name = ('' if data.scenario == 'dynamics_only' else f'{data.scenario}_') + f'{data.state_mode}_seed_{data.seed}'
    run = ExperimentRun(args.output_dir, name, 'synthetic_lds_images', experiment_settings(args), fingerprint)
    if args.resume and run.complete():
        report = json.loads((run.path / 'results.json').read_text(encoding='utf-8'))
        append_summary(args.output_dir, report)
        print(f'{name}: resumed; {run.path}')
        return report
    try:
        with run:
            metadata = dict(state_mode=data.state_mode, data_seed=data.seed, scenario=data.scenario,
                source=str(args.input.resolve()) if args.input else 'generated in memory',
                representation=args.representation, image_shape=list(data.images.shape[1:]),
                shuffle=args.shuffle, dynamics_step='successive observations assigned to each cluster',
                observation='Exact fixed affine Gaussian; no GP fit and no 1D image warp',
                evaluation='In-sample clustering; truth is used only by metrics/plots except explicitly oracle C/noise calibration')
            if args.representation == 'oracle':
                decoder = data.observation_model()
            else:
                estimate = estimate_pca_observation(values, data.grid,
                    components=args.components if args.components is not None else data.C.shape[1],
                    max_beats=args.calibration_samples, sampling=args.calibration_sampling,
                    random_state=args.seed, noise_floor=args.noise_floor)
                decoder = AffineGaussianObservation(estimate.components, estimate.residual_covariance,
                    data.grid, estimate.mean, np.diag(estimate.latent_variances))
                metadata.update(calibration_fit_rows=estimate.beat_indices,
                                explained_variance_ratio=estimate.explained_variance_ratio)
            decoder = replace(decoder, Sigma=decoder.Sigma * args.sigma_prior)
            # In PCA coordinates a scalar process variance changes scale. Use
            # the true covariance projected into that basis as a calibration
            # control, or the practitioner's explicit absolute variance.
            transform = np.linalg.pinv(decoder.C) @ data.C
            calibration_gamma = np.mean([np.trace(transform @ g @ transform.T) / decoder.latent_dim
                                         for g in data.Gamma])
            gamma = float(calibration_gamma if args.gamma_prior is None else args.gamma_prior)
            metadata.update(gamma_prior=gamma, gamma_prior_source=(
                'truth-informed mean variance in decoder coordinates' if args.gamma_prior is None else 'explicit'),
                sigma_prior_multiplier=args.sigma_prior,
                sigma_prior_diagonal_mean=float(np.diag(decoder.Sigma).mean()),
                source_state_matches_cluster_convention=data.state_mode == 'per_regime' and not args.shuffle)
            config = make_config(args, decoder, gamma)
            config.validate(for_build=True)
            (run.path / 'config.yaml').write_text(config.to_yaml(), encoding='utf-8')
            np.savez_compressed(run.path / 'observation_calibration.npz', **decoder.state_dict())
            np.savez_compressed(run.path / 'ground_truth.npz', images=data.images[indices],
                clean_images=data.clean_images[indices], latent_states=data.states[indices],
                labels=truth_labels, source_frames=indices, segment_ids=data.segment_ids[indices],
                source_previous_indices=data.previous_indices[indices],
                A=data.A, Gamma=data.Gamma, C=data.C, Sigma_diagonal=data.Sigma_diagonal,
                offset=data.offset, initial_covariance=data.initial_covariance,
                initial_state_means=data.initial_state_means,
                initial_state_covariances=data.initial_state_covariances,
                regime_mean_images=data.regime_mean_images, scenario=data.scenario,
                regime_names=np.asarray(data.regime_names))
            if args.verbose != 'silent':
                print(f'Synthetic images: {len(values)} frames, shape={data.images.shape[1:]}, '
                      f'representation={args.representation}, latent dimension={decoder.latent_dim}')
                print(f'Generation={data.scenario}/{data.state_mode}; shuffled={args.shuffle}; Gamma prior={gamma:g}')
                if data.state_mode != 'per_regime':
                    print('The generated global switching state differs from the model\'s resumed cluster states.')
            log_configuration(config, args, source=metadata['gamma_prior_source'])
            model = HDPGPC(config).fit(values, grid=data.grid)
            trajectories = model_latent_trajectories(model)
            inferred, comparison = _state_comparison(model, data, indices, decoder, trajectories, args.representation)
            write_json(run.path / 'ground_truth_comparison.json', comparison)
            np.savez_compressed(run.path / 'latent_trajectories.npz',
                inferred_states_in_true_coordinates=inferred, source_frames=indices,
                **{f'cluster_{cluster}_{key}': array for cluster, path in trajectories.items()
                   for key, array in path.items()})
            metadata['ground_truth_comparison'] = comparison
            def summary_plotter(fitted, observations, annotations, **kwargs):
                return plot_image_clustering_summary(fitted, observations, annotations,
                    image_shape=data.images.shape[1:], times=np.arange(len(values)),
                    frames_per_cluster=args.frames_per_cluster, trajectories=trajectories,
                    representation_name=args.representation.capitalize(), random_state=args.seed,
                    time_label='Fit row', time_unit='', reference_title='True dynamical regimes (evaluation only)',
                    image_limits=(min(0., float(values.min())), max(1., float(values.max()))), **kwargs)
            def extra_figures():
                from ..diagnostics._synthetic_images import plot_switching_lds_recovery
                figures = {'ground_truth_recovery': plot_switching_lds_recovery(
                    data, indices, model.labels_, inferred)}
                if args.trajectory_plots:
                    figures['latent_trajectories'] = plot_latent_trajectories(
                        model, times=np.arange(len(values)), trajectories=trajectories,
                        title=name, min_cluster_samples=args.min_cluster_samples,
                        time_label='Fit row', time_unit='')
                return figures
            report = save_results(run, model, values, labels, data.grid, args,
                ordered=not args.shuffle, metadata=metadata,
                codes=(values[:, :, 0]-decoder.offset) @ np.linalg.pinv(decoder.C).T,
                projection_name=args.representation.capitalize(),
                row_metadata={'source_frame': indices, 'true_regime_id': truth_labels,
                              'segment': data.segment_ids[indices]},
                summary_plotter=summary_plotter, extra_figures=extra_figures)
    except Exception:
        report_path = run.path / 'results.json'
        if not report_path.is_file():
            raise
        report = json.loads(report_path.read_text(encoding='utf-8'))
    append_summary(args.output_dir, report)
    print(f'{run.dataset}: {report["status"]}; {run.path.resolve()}')
    return report


def main():
    import matplotlib
    matplotlib.use('Agg')
    parser = build_parser()
    args = parser.parse_args()
    if run_synthetic_clustering(args)['status'] != 'ok':
        raise SystemExit(1)
