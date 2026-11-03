"""Unsupervised flutter motion-window clustering with a common fixed linear AE."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from .. import HDPGPC
from ..datasets.flutter import (FLUTTER_RECORDS, download_flutter, load_flutter_record,
                                window_flutter_record)
from ..diagnostics import model_latent_trajectories, plot_latent_trajectories
from ..emissions import estimate_autoencoder_observation
from ._affine import make_config
from ._arguments import add_common_arguments, validate_common, validate_affine, NOISE_MODES
from ._runs import ExperimentRun, experiment_settings, save_results, append_summary, log_configuration, write_json


DEFAULTS = dict(schedule='offline', offline_strategy='split_merge', iterations=25,
    split_merge_refine_steps=3, exploration_steps=7, max_clusters=10,
    hdp_preset='balanced', lds_degrees=10, fit_gp=False, alignment='none',
    hard_assignments=False, projection_plots=True, lds_plots=False,
    lds_dynamics_plots=False, warp_plots=False)


def build_parser(description=None, defaults=None):
    parser = argparse.ArgumentParser(description=description or __doc__)
    add_common_arguments(parser, defaults={**DEFAULTS, **(defaults or {})}, ordered=True)
    parser.add_argument('--data-dir', type=Path, default=Path.cwd()/'agents/data/flutter')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--output-dir', '--plot-dir', type=Path,
                        default=Path.cwd()/'agents/flutter_autoencoder_runs')
    parser.add_argument('--records', nargs='+', choices=tuple(FLUTTER_RECORDS), default=list(FLUTTER_RECORDS))
    parser.add_argument('--calibration-records', nargs='+', choices=tuple(FLUTTER_RECORDS),
                        help='Default: selected records, pooled only for shared unsupervised calibration.')
    parser.add_argument('--window-samples', type=int, default=60,
                        help='Causal delay window, roughly 0.25 seconds at 240 Hz.')
    parser.add_argument('--step', '--frame-step', type=int, default=24,
                        help='Window advance in source samples; 24 is roughly 0.1 seconds.')
    parser.add_argument('--max-series', type=int, help='Window prefix per record, also applied to calibration.')
    parser.add_argument('--components', '--latent-dim', type=int, default=10)
    parser.add_argument('--calibration-samples', type=int, default=400)
    parser.add_argument('--epochs', '--ae-epochs', type=int, default=250)
    parser.add_argument('--learning-rate', type=float, default=.02)
    parser.add_argument('--noise-floor', type=float, default=1e-6)
    parser.add_argument('--noise-floor-fraction', type=float, default=.05)
    parser.add_argument('--sigma-prior', type=float, default=1.0, help='Calibrated observation covariance multiplier.')
    parser.add_argument('--gamma-prior', type=float, default=1.0, help='Initial per-coordinate process variance.')
    parser.add_argument('--sigma-mode', choices=NOISE_MODES, default='fixed')
    parser.add_argument('--configurations', nargs='+', choices=('independent', 'dynamic'), default=['dynamic'])
    parser.add_argument('--hmm-switch', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--trajectory-plots', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--stability-samples', type=int, default=200,
                        help='MNIW transition posterior draws per cluster; 0 saves point diagnostics only.')
    parser.set_defaults(dataset='flutter', representation='autoencoder', **(defaults or {}))
    return parser


def _validate(args):
    validate_common(args); validate_affine(args)
    if args.schedule != 'offline' or args.normalize_samples or args.birth_plots or args.warp_plots:
        raise ValueError('Flutter requires offline inference, preserved amplitudes and no warp/birth plots')
    if args.transition_topology == 'left_to_right' and not args.hmm_switch:
        raise ValueError('left_to_right requires HMM assignments')
    if (args.window_samples < 2 or args.step < 1 or args.components < 1
            or args.components >= 2*args.window_samples or args.calibration_samples <= args.components
            or args.epochs < 1 or args.stability_samples < 0
            or (args.max_series is not None and args.max_series <= args.components)):
        raise ValueError('Invalid flutter window, latent/calibration dimensions or diagnostic budget')
    for value in (args.learning_rate, args.noise_floor, args.sigma_prior, args.gamma_prior):
        if not np.isfinite(value) or value <= 0:
            raise ValueError('Learning rate and covariance scales must be finite and positive')
    if not np.isfinite(args.noise_floor_fraction) or args.noise_floor_fraction < 0:
        raise ValueError('Noise floor fraction must be finite and nonnegative')


def prepare_windows(records, args):
    result = {}
    for name, record in records.items():
        windows = window_flutter_record(record, window_samples=args.window_samples, step=args.step)
        if args.max_series is not None:
            windows = replace(windows, values=windows.values[:args.max_series],
                starts=windows.starts[:args.max_series], ends=windows.ends[:args.max_series],
                times=windows.times[:args.max_series])
        if len(windows.values) <= args.components:
            raise ValueError(f'{name}: not enough motion windows for the requested latent dimension')
        result[name] = windows
    return result


def calibrate_windows(windows, names, args):
    """Fixed global channel scaling and balanced, unlabeled per-record sampling."""
    raw = np.concatenate([windows[name].record.motion[:windows[name].ends[-1]+1] for name in names])
    center, scale = raw.mean(0), raw.std(0)
    scale = np.where(scale > np.finfo(float).eps, scale, 1.)
    rng = np.random.default_rng(args.seed)
    samples, sample_records, sample_ends = [], [], []
    if args.calibration_samples < len(names):
        raise ValueError('Calibration sample budget must cover every calibration record')
    for i, name in enumerate(names):
        data = windows[name]
        quota = args.calibration_samples//len(names)+(i < args.calibration_samples % len(names))
        chosen = np.sort(rng.choice(len(data.values), min(quota, len(data.values)), replace=False))
        values = (data.values[chosen].reshape(-1, args.window_samples, 2)-center)/scale
        samples.append(values.reshape(len(chosen), -1))
        sample_records.extend([name]*len(chosen)); sample_ends.extend(data.ends[chosen])
    batch = np.concatenate(samples)
    if len(batch) <= args.components:
        raise ValueError('Balanced calibration provides too few windows')
    grid = np.arange(batch.shape[1], dtype=float)[:, None]
    estimate = estimate_autoencoder_observation(batch, grid, components=args.components,
        max_samples=len(batch), epochs=args.epochs, learning_rate=args.learning_rate,
        random_state=args.seed, noise_floor=args.noise_floor,
        noise_floor_fraction=args.noise_floor_fraction, noise_structure='diagonal')
    return estimate, center, scale, batch, np.asarray(sample_records), np.asarray(sample_ends)


def _fit_record(windows, args, run, configuration, calibration, calibration_names):
    from ..diagnostics._flutter import (flutter_diagnostics, plot_flutter_summary,
                                       plot_flutter_embedding)
    from ..diagnostics._calibration_plots import plot_autoencoder_calibration
    estimate, center, scale, batch, sample_records, sample_ends = calibration
    decoder = estimate.model
    values = ((windows.values.reshape(-1, args.window_samples, 2)-center)/scale).reshape(len(windows.values), -1)
    grid = decoder.grid
    config = make_config(args, decoder, dynamic=configuration == 'dynamic')
    np.savez_compressed(run.path/'prepared_data.npz',
        source_times=windows.record.times[:windows.ends[-1]+1],
        source_motion=windows.record.motion[:windows.ends[-1]+1],
        window_starts=windows.starts, window_ends=windows.ends, window_times=windows.times,
        raw_windows=windows.values, scaled_windows=values)
    np.savez_compressed(run.path/'autoencoder_calibration.npz',
        **decoder.state_dict(), applied_Sigma=config.observation.Sigma_policy.value,
        channel_center=center, channel_scale=scale, losses=estimate.losses,
        calibration_indices=estimate.indices, calibration_records=sample_records,
        calibration_window_ends=sample_ends)
    log_configuration(config, args, source='common unlabeled raw-window autoencoder; fixed channel scaling')
    print(f'Flutter {windows.record.name} {configuration}: windows={len(values)}, '
          f'features={values.shape[1]}, latent dimension={decoder.latent_dim}; '
          f'published record context: {windows.record.reference}')
    model = HDPGPC(config).fit(values[:, :, None], grid=grid)
    paths = model_latent_trajectories(model)
    np.savez_compressed(run.path/'latent_trajectories.npz', times=windows.times,
        **{f'cluster_{cluster}_{name}': value for cluster, path in paths.items() for name, value in path.items()})
    codes = (values-decoder.offset) @ np.linalg.pinv(decoder.C).T
    diagnostics, reconstructed = flutter_diagnostics(model, windows, values, codes, decoder,
        center, scale, paths, samples=args.stability_samples, seed=args.seed)
    write_json(run.path/'dynamics.json', diagnostics)
    metadata = dict(configuration=configuration, representation='tied_linear_orthonormal_autoencoder',
        source_doi='10.1007/s44245-024-00043-w', source_sha256=windows.record.file_sha256,
        archive_sha256=windows.record.archive_sha256,
        airflow_speed=windows.record.airflow_speed, initial_condition=windows.record.initial_condition,
        reference=windows.record.reference, reference_scope='record-level description; no pointwise truth',
        calibration_records=calibration_names, calibration='balanced windows, retrospective and label-free',
        calibration_windows=len(batch), components=decoder.latent_dim, preliminary_projection=False,
        window_samples=args.window_samples, step=args.step,
        observation_layout='time-major, two channels interleaved; causal complete windows',
        channel_center=center, channel_scale=scale, per_window_normalization=False,
        overlapping_windows=args.step < args.window_samples,
        dynamics_clock=args.dynamics_clock,
        dynamics_clock_description=('Global regular sampling ticks' if args.dynamics_clock=='physical-clock'
                                    else 'Successive retained observations; gap validity recorded in dynamics.json'),
        interpretation='Latent window dynamics and smoothed in-sample reconstructions, not proof of physical instability or causal warning',
        diagnostics=diagnostics)

    def summary_plotter(fitted, observations, labels, **kwargs):
        return [plot_flutter_summary(windows, model.labels_, reconstructed, diagnostics,
                                     title=f'{windows.record.name}: {configuration}')]

    def figures():
        result = {}
        if args.trajectory_plots:
            result['latent_trajectories'] = plot_latent_trajectories(model, times=windows.times,
                title=f'Flutter {windows.record.name}: {configuration}; '+(
                    'global sampling steps' if args.dynamics_clock=='physical-clock' else 'retained-window steps'),
                trajectories=paths, min_cluster_samples=args.min_cluster_samples,
                time_label='Recorded window end time (s)', time_unit='s', observation_label='windows')
        figure = plot_autoencoder_calibration(batch, estimate.network, estimate.indices, estimate.losses)
        figure.axes[1].set(xlabel='Flattened delay-window coordinates (two channels)', ylabel='Scaled motion')
        result['autoencoder_calibration'] = figure
        if args.projection_plots:
            result['latent_embedding'] = plot_flutter_embedding(codes, model.labels_, windows.times,
                title=f'{windows.record.name}: common AE coordinates')
        return result

    return save_results(run, model, values[:, :, None], None, grid, args, ordered=True,
        metadata=metadata, codes=codes, row_metadata=dict(
            second=windows.times, window_start_sample=windows.starts, window_end_sample=windows.ends,
            record_reference=np.repeat(windows.record.reference, len(values))),
        summary_plotter=summary_plotter, extra_figures=figures)


def run_flutter(args):
    import torch
    from ..diagnostics._flutter import plot_flutter_overview, plot_flutter_structure
    torch.set_num_threads(1)
    _validate(args)
    if args.download:
        download_flutter(args.data_dir)
    names = list(dict.fromkeys(args.records))
    calibration_names = list(dict.fromkeys(args.calibration_records or names))
    sources = {name: load_flutter_record(args.data_dir, name) for name in dict.fromkeys(names+calibration_names)}
    windows = prepare_windows(sources, args)
    calibration = None
    fingerprint = hashlib.sha256(json.dumps({name: sources[name].file_sha256
        for name in sorted(sources)}, sort_keys=True).encode()).hexdigest()
    reports = []
    for name in names:
        for configuration in dict.fromkeys(args.configurations):
            settings = experiment_settings(args)
            settings.pop('configurations', None)
            settings.update(record=name, configuration=configuration, calibration_records=calibration_names)
            run = ExperimentRun(args.output_dir, f'flutter_{name}',
                                f'flutter_autoencoder_{configuration}', settings, fingerprint)
            if args.resume and run.complete():
                report = json.loads((run.path/'results.json').read_text())
                print(f'Using saved results from: {run.path.resolve()}')
            else:
                try:
                    with run:
                        print(f'Results directory: {run.path.resolve()}')
                        if calibration is None:
                            calibration = calibrate_windows(windows, calibration_names, args)
                        report = _fit_record(windows[name], args, run, configuration, calibration, calibration_names)
                        print(f'Results saved to: {run.path.resolve()}')
                except Exception:
                    report = json.loads((run.path/'results.json').read_text())
            append_summary(args.output_dir, report)
            reports.append(report)
    write_json(args.output_dir/'comparison.json', dict(
        source_doi='10.1007/s44245-024-00043-w', records=names,
        calibration_records=calibration_names,
        reference_scope='Published record descriptions; no pointwise reference segmentation',
        cluster_ids='local to each recording/configuration; not shared classes',
        runs=[{key: report.get(key) for key in ('dataset', 'run_id', 'experiment', 'status',
              'clusters', 'seconds', 'output_directory', 'error')} |
              {'diagnostics': report.get('metadata', {}).get('diagnostics', {})} for report in reports]))
    if args.plots and all(report['status'] == 'ok' for report in reports):
        import matplotlib.pyplot as plt
        figure = plot_flutter_overview(reports)
        figure.savefig(args.output_dir/'comparison.png', dpi=150)
        plt.close(figure)
        figure = plot_flutter_structure(reports)
        figure.savefig(args.output_dir/'autoencoder_structure.png', dpi=150)
        plt.close(figure)
    print(f'Flutter comparison saved to: {args.output_dir.resolve()}')
    return reports
