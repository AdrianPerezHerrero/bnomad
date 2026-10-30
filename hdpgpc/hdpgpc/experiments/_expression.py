"""Shared temporal expression-state comparisons; dataset extraction remains in datasets."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from .. import HDPGPC
from ..config import (HDPGPCConfig, HDPConfig, GPEmissionConfig, DynamicsConfig,
    AlignmentConfig, InferenceConfig, NumericalConfig, MatrixPolicyConfig,
    NoisePolicyConfig, with_mfa_emission, with_fixed_observation_model)
from ..diagnostics import cluster_label_statistics, model_latent_trajectories, plot_latent_trajectories
from ..emissions import (estimate_mfa_observation, estimate_pca_observation,
                         estimate_autoencoder_observation, AffineGaussianObservation)
from ..preprocessing import fit_pca_projection
from ._arguments import add_common_arguments, validate_common, validate_affine, lds_update, NOISE_MODES
from ._runs import (ExperimentRun, experiment_settings, save_results, append_summary,
                    log_configuration, write_json, save_mfa_posteriors)


DEFAULTS = dict(schedule='offline', offline_strategy='split_merge',
    split_merge_refine_steps=1, exploration_steps=5, max_clusters=20,
    hdp_preset='balanced', fit_gp=False, alignment='none', hard_assignments=False,
    lds_plots=False, lds_dynamics_plots=False)


def build_parser(description=None, defaults=None, *, dataset='expression', representation='mfa'):
    if representation not in ('mfa', 'pca', 'autoencoder'):
        raise ValueError('Expression representation must be mfa, pca or autoencoder')
    parser = argparse.ArgumentParser(description=description or __doc__)
    add_common_arguments(parser, defaults={**DEFAULTS, **(defaults or {})}, ordered=True)
    parser.add_argument('--data-dir', type=Path, default=Path.cwd() / f'agents/data/{dataset}')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--output-dir', '--plot-dir', type=Path,
                        default=Path.cwd() / f'agents/{dataset}_{representation}_runs')
    parser.add_argument('--configurations', nargs='+', choices=('independent', 'dynamic'),
                        default=['independent', 'dynamic'],
                        help='Both use HMM assignments; dynamic learns A, independent fixes A=0.')
    parser.add_argument('--projection-components', type=int, default=30,
                        help='Preliminary PCA dimensions, capped at numerical rank; 0 bypasses this projection.')
    parser.add_argument('--components', '--latent-dim', type=int, default=10,
                        help='Latent state dimension, smaller than the observation dimension.')
    parser.add_argument('--calibration-samples', type=int, default=400)
    parser.add_argument('--sigma-prior', type=float,
                        default=0.001 if representation == 'mfa' else 1.,
                        help='Multiplier of the calibrated observation covariance.')
    parser.add_argument('--gamma-prior', type=float, default=0.01)
    parser.add_argument('--noise-floor', type=float, default=1e-6)
    if representation == 'mfa':
        parser.add_argument('--mean-precision', type=float, default=1.)
        parser.add_argument('--loading-precision', type=float, default=1.)
        parser.add_argument('--shared-emission', action=argparse.BooleanOptionalAction, default=True)
    else:
        parser.add_argument('--sigma-mode', choices=NOISE_MODES, default='fixed',
                            help='Observation covariance policy; defaults to fixed calibrated Sigma.')
    if representation == 'autoencoder':
        parser.add_argument('--epochs', '--ae-epochs', type=int, default=250)
        parser.add_argument('--learning-rate', type=float, default=.02)
        parser.add_argument('--noise-floor-fraction', type=float, default=.05)
    parser.add_argument('--hmm-switch', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--target', choices=('broad', 'stage'), default='broad',
                        help='Primary reporting label only; both targets are scored after fitting.')
    parser.add_argument('--max-series', type=int, help='Chronological prefix of sampling stages.')
    parser.add_argument('--trajectory-plots', action=argparse.BooleanOptionalAction, default=True)
    parser.set_defaults(dataset=dataset, representation=representation)
    if defaults:
        parser.set_defaults(**defaults)
    return parser


def _validate(args):
    validate_common(args)
    validate_affine(args)
    if args.schedule != 'offline':
        raise ValueError('Expression comparison requires --schedule offline')
    if args.normalize_samples or args.birth_plots:
        raise ValueError('Per-snapshot normalization and birth-warp plots do not apply to gene expression')
    if args.representation == 'mfa' and args.shared_emission and args.lds_update not in (None, 'conjugate_batch'):
        raise ValueError('Shared MFA requires --lds-update conjugate_batch')
    if args.transition_topology == 'left_to_right' and not args.hmm_switch:
        raise ValueError('left_to_right requires HMM assignments')
    if (args.components < 1 or args.projection_components < 0
            or (args.projection_components > 0 and args.projection_components <= args.components)
            or args.calibration_samples <= args.components
            or (args.max_series is not None and args.max_series < args.components + 2)):
        raise ValueError('Require projection-components=0 or > components, calibration dimensions '
                         '> components and enough sampling stages')
    precisions = ((args.mean_precision, args.loading_precision) if args.representation == 'mfa' else ())
    if any(not np.isfinite(value) or value <= 0 for value in (
            args.sigma_prior, args.gamma_prior, args.noise_floor, *precisions)):
        raise ValueError('Noise/prior scales and precisions must be finite and positive')
    if args.representation == 'autoencoder' and (
            args.epochs < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not np.isfinite(args.noise_floor_fraction) or args.noise_floor_fraction < 0):
        raise ValueError('Autoencoder epochs/rate must be positive and noise floor fraction nonnegative')


def make_config(args, initialization, *, dynamic):
    """Shared configuration for learned MFA or fixed PCA/autoencoder emissions."""
    from sklearn.gaussian_process.kernels import RBF
    initialization = replace(initialization, Sigma=initialization.Sigma * args.sigma_prior)
    config = HDPGPCConfig(
        x_basis=np.arange(initialization.latent_dim, dtype=float)[:, None],
        verbose=args.verbose,
        emission=GPEmissionConfig(kernels=[RBF(1.)], fit_gp=False, ini_sigma=args.sigma_prior),
        dynamics=DynamicsConfig(ini_gamma=args.gamma_prior, annealing=False,
            bayesian_params=True, free_deg_MNIV=args.lds_degrees,
            lds_update=lds_update(args), lds_scoring='final', lds_step_size=args.lds_step_size),
        alignment=AlignmentConfig(policy='none'),
        hdp=HDPConfig(hdp_hyp=args.hdp_preset, hmm_switch=args.hmm_switch,
            max_models=args.max_clusters, transition_topology=args.transition_topology),
        inference=InferenceConfig(schedule='offline', warp=False, it_limit=args.iterations,
            offline_strategy=args.offline_strategy, n_explore_steps=args.exploration_steps,
            offline_representer_selection=args.offline_representer_selection,
            split_merge_refine_steps=args.split_merge_refine_steps,
            split_merge_stability_patience=args.split_merge_stability_patience,
            split_merge_score_rtol=args.split_merge_score_rtol,
            hard_assignments=args.hard_assignments,
            birth_lds_kl_per_dimension=args.birth_lds_kl_per_dimension),
        numerics=NumericalConfig(seed=args.seed, num_threads=1, dtype='float64', deterministic=True))
    if args.representation == 'mfa':
        config = with_mfa_emission(config, initialization, dynamic=dynamic,
            shared=args.shared_emission, mean_precision=args.mean_precision,
            loading_precision=args.loading_precision)
    else:
        config = with_fixed_observation_model(config, initialization,
            sigma_policy=NoisePolicyConfig(mode=args.sigma_mode))
        if not dynamic:
            config = replace(config, dynamics=replace(config.dynamics,
                A_policy=MatrixPolicyConfig(mode='fixed', value=np.zeros((initialization.latent_dim,)*2))))
    # The MFA convenience helper defaults static factors to IID assignments.
    # This experiment deliberately keeps the requested assignment process in
    # both arms: A=0/Gamma=I removes factor evolution, not HMM dependence.
    return replace(config, hdp=replace(config.hdp, hmm_switch=args.hmm_switch))


def _calibrate_observation(scores, grid, args):
    """Calibrate once per time course so both factor configurations share a decoder."""
    if args.representation == 'mfa':
        return estimate_mfa_observation(scores, grid, components=args.components,
            max_samples=args.calibration_samples, random_state=args.seed, noise_floor=args.noise_floor), None
    if args.representation == 'pca':
        estimate = estimate_pca_observation(scores, grid, components=args.components,
            max_beats=args.calibration_samples, random_state=args.seed, noise_floor=args.noise_floor)
        decoder = AffineGaussianObservation(estimate.components, estimate.residual_covariance,
            grid, estimate.mean, np.diag(estimate.latent_variances))
    else:
        estimate = estimate_autoencoder_observation(scores, grid, components=args.components,
            max_samples=args.calibration_samples, epochs=args.epochs, learning_rate=args.learning_rate,
            random_state=args.seed, noise_floor=args.noise_floor,
            noise_floor_fraction=args.noise_floor_fraction, noise_structure='diagonal')
        decoder = estimate.model
    return decoder, estimate


def _run_configuration(data, values, projection, scores,
                       initialization, args, run, configuration, calibration=None):
    from ..diagnostics._expression_plots import plot_expression_clustering_summary
    dynamic = configuration == 'dynamic'
    grid = np.arange(scores.shape[1], dtype=float)[:, None]
    observations = scores[:, :, None]
    n = len(scores)
    config = make_config(args, initialization, dynamic=dynamic)
    np.savez_compressed(run.path / 'prepared_data.npz', gene_ids=data.gene_ids,
        gene_names=data.gene_names, **data.prepared_arrays,
        **{name: array[:n] for name, array in data.time_arrays.items()})
    prior_state = initialization.state_dict()
    prior_state['Sigma'] = config.observation.Sigma_policy.value
    projection_state = (dict(gene_center=projection.mean, gene_loadings=projection.loadings,
        projection_scale=projection.scale, explained_variance_ratio=projection.explained_variance_ratio,
        numerical_rank=projection.numerical_rank) if projection is not None else {})
    np.savez_compressed(run.path / 'observation_calibration.npz',
        projection_applied=projection is not None, scores=scores, **projection_state,
        representation=args.representation,
        **{f'{"mfa" if args.representation == "mfa" else "decoder"}_{key}': value
           for key, value in prior_state.items()})
    calibration_metadata = {}
    if args.representation == 'autoencoder':
        np.savez_compressed(run.path / 'training.npz', losses=calibration.losses, indices=calibration.indices)
        calibration_metadata = dict(encoder='tied_linear_orthonormal', epochs=args.epochs,
            learning_rate=args.learning_rate, final_scaled_mse=float(calibration.losses[-1]))
    elif args.representation == 'pca':
        np.savez_compressed(run.path / 'pca_calibration.npz', indices=calibration.beat_indices,
            latent_variances=calibration.latent_variances,
            explained_variance_ratio=calibration.explained_variance_ratio)
        calibration_metadata = dict(explained_variance_ratio=calibration.explained_variance_ratio)
    (run.path / 'config.yaml').write_text(config.to_yaml(), encoding='utf-8')
    representation = (data.metadata['expression_units'] if projection is None else 'All-gene PCA')
    reduction = ('PCA disabled' if projection is None else
                 f'PCA={scores.shape[1]}, explained variance={projection.explained_variance_ratio.sum():.4f}')
    print(f'{data.title} {configuration}: {len(data.gene_ids)} genes retained; {n} sampling stages; '
          f'{reduction}; state dimension={len(config.x_basis)}')
    log_configuration(config, args,
        source=f'same all-gene observations and {args.representation} calibration in both arms')
    model = HDPGPC(config).fit(observations, grid=grid)
    if args.representation == 'mfa':
        save_mfa_posteriors(model, run.path)
    trajectories = model_latent_trajectories(model)
    np.savez_compressed(run.path / 'latent_trajectories.npz', times=data.hours[:n],
        **{f'cluster_{cluster}_{name}': value for cluster, path in trajectories.items()
           for name, value in path.items()})
    targets = {name: labels[:n] for name, labels in data.targets.items()}
    masks = {name: mask[:n] for name, mask in data.evaluation_masks.items()}
    metrics = {}
    for name, labels in targets.items():
        selected = masks.get(name, np.ones(n, dtype=bool))
        metrics[name] = (cluster_label_statistics(model.labels_[selected], labels[selected])[0]
                         if selected.any() else None)
    write_json(run.path / 'label_metrics.json', metrics)

    def summary_plotter(fitted, fitted_values, labels, **kwargs):
        return plot_expression_clustering_summary(fitted, fitted_values, labels,
            stages=data.stage_names[:n], hours=data.hours[:n],
            projected=projection is not None, expression_units=data.metadata['expression_units'],
            reference_title=data.metadata.get('target_labels', {}).get(args.target, 'Developmental reference'),
            **kwargs)

    def extra_figures():
        figures = ({'latent_trajectories': plot_latent_trajectories(model, times=data.hours[:n],
                    title=f'{data.title} {configuration}', trajectories=trajectories,
                    min_cluster_samples=args.min_cluster_samples,
                    time_label='Hours post-fertilisation (unequal sampling intervals)',
                    time_unit='h', observation_label='stages')}
                if args.trajectory_plots else {})
        if args.representation == 'autoencoder':
            from ..diagnostics._calibration_plots import plot_autoencoder_calibration
            figure = plot_autoencoder_calibration(scores, calibration.network,
                calibration.indices, calibration.losses)
            figure.axes[1].set(xlabel='Observation component' if projection is not None else 'Gene index',
                               ylabel='Expression observation')
            figures['autoencoder_calibration'] = figure
        return figures

    metadata = dict(data.metadata, configuration=configuration, dynamic=dynamic,
        representation=args.representation, decoder_is_fixed=args.representation in ('pca', 'autoencoder'),
        emission_family=config.observation.family,
        observation_calibration=calibration_metadata,
        source_sha256=data.source_sha256, gene_count=len(data.gene_ids), selected_genes=False,
        constant_genes_retained=int(np.count_nonzero(np.ptp(values, axis=0) == 0)),
        fitted_source_samples=int(data.time_arrays['replicate_counts'][:n].sum()),
        projection_applied=projection is not None,
        projection_components=scores.shape[1] if projection is not None else 0,
        observation_dimension=scores.shape[1], factor_components=len(config.x_basis),
        projection_explained_variance=(float(projection.explained_variance_ratio.sum())
                                       if projection is not None else None),
        projection_calibration=('all fitted sampling-stage means, retrospective; labels unused'
                                if projection is not None else None),
        label_metrics=metrics, primary_target=args.target,
        factor_dynamics_step='successive observations included in a cluster; not elapsed hours',
        evaluation='in-sample reference agreement; no held-out classification',
        evaluated_observations={name: int(masks.get(name, np.ones(n, bool)).sum()) for name in targets})
    return save_results(run, model, observations, targets[args.target], grid, args,
        ordered=True, codes=scores, projection_name=representation, metadata=metadata,
        row_metadata={'stage': data.stage_names[:n], 'broad_category': targets['broad'],
                      'hours_post_fertilisation': data.hours[:n],
                      'reference_stage': targets['stage'],
                      'stage_evaluation_included': masks.get('stage', np.ones(n, bool)),
                      'evaluation_included': masks.get(args.target, np.ones(n, bool)),
                      'replicates': data.time_arrays['replicate_counts'][:n]},
        evaluation_mask=masks.get(args.target),
        summary_plotter=summary_plotter, extra_figures=extra_figures)



def run_expression_comparison(data, args, *, comparison_name='comparison'):
    import torch
    torch.set_num_threads(1)
    _validate(args)
    values = data.values[:args.max_series] if args.max_series is not None else data.values
    projection = (fit_pca_projection(values, components=args.projection_components)
                  if args.projection_components > 0 else None)
    scores = projection.transform(values) if projection is not None else values
    if scores.shape[1] <= args.components or len(scores) <= args.components:
        raise ValueError('MFA needs more observation dimensions and sampling stages than factors')
    grid = np.arange(scores.shape[1], dtype=float)[:, None]
    initialization, calibration = None, None
    fingerprint = hashlib.sha256(json.dumps(data.source_sha256, sort_keys=True).encode()).hexdigest()
    reports = []
    for configuration in dict.fromkeys(args.configurations):
        settings = experiment_settings(args)
        settings.pop('configurations', None)
        settings['configuration'] = configuration
        run = ExperimentRun(args.output_dir, data.name,
                            f'{args.dataset}_{args.representation}_{configuration}', settings, fingerprint)
        if args.resume and run.complete():
            report = json.loads((run.path / 'results.json').read_text(encoding='utf-8'))
            print(f'Using saved results from: {run.path.resolve()}')
        else:
            try:
                with run:
                    print(f'Results directory: {run.path.resolve()}')
                    if initialization is None:
                        initialization, calibration = _calibrate_observation(scores, grid, args)
                    report = _run_configuration(data, values,
                        projection, scores, initialization, args, run, configuration, calibration)
                    print(f'Results saved to: {run.path.resolve()}')
            except Exception:
                report = json.loads((run.path / 'results.json').read_text(encoding='utf-8'))
        append_summary(args.output_dir, report)
        reports.append(report)
    comparison = {'dataset': data.name, 'source_sha256': data.source_sha256, 'gene_selection': False,
        'representation': args.representation,
        'dataset_accession': data.metadata['dataset_accession'],
        'target_source': data.metadata['target_source'],
        'gene_count': len(data.gene_ids), 'sampling_stages': len(scores),
        'projection_applied': projection is not None,
        'projection_components': scores.shape[1] if projection is not None else 0,
        'observation_dimension': scores.shape[1],
        'factor_components': args.components,
        'projection_explained_variance': (float(projection.explained_variance_ratio.sum())
                                         if projection is not None else None),
        'runs': [{key: report.get(key) for key in (
            'experiment', 'run_id', 'status', 'clusters', 'metrics', 'seconds', 'output_directory', 'error')}
            | {'label_metrics': report.get('metadata', {}).get('label_metrics', {}),
               'evaluated_observations': report.get('metadata', {}).get('evaluated_observations', {})}
            for report in reports]}
    write_json(Path(args.output_dir) / f'{comparison_name}.json', comparison)
    if args.plots and all(report['status'] == 'ok' for report in reports):
        import matplotlib.pyplot as plt
        from ..diagnostics._expression_plots import plot_expression_comparison
        figure = plot_expression_comparison(reports, data.stage_names[:len(scores)], title=data.title)
        try:
            figure.savefig(Path(args.output_dir) / f'{comparison_name}.png', dpi=150)
        finally:
            plt.close(figure)
    print(f'Comparison results saved to: {Path(args.output_dir).resolve()}')
    return reports
