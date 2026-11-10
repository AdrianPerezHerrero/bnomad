"""XJTU-SY raw-window clustering with shared PCA, linear AE, Fourier or Haar observations."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .. import HDPGPC
from .._defaults import DEFAULT_SIGMA_PRIOR, DEFAULT_GAMMA_PRIOR
from ..datasets.xjtu_sy import (XJTU_RECORDS, XJTU_SOURCE, XJTU_SAMPLE_RATE,
    XJTU_SAMPLES, XJTU_PERIOD_SECONDS, download_xjtu_sy, load_xjtu_sy_windows)
from ..diagnostics import model_latent_trajectories, plot_latent_trajectories
from ..emissions import (estimate_autoencoder_observation, estimate_pca_observation,
                         AffineGaussianObservation)
from ._affine import make_config
from ._arguments import add_common_arguments, validate_common, validate_affine, NOISE_MODES
from ._runs import ExperimentRun, experiment_settings, save_results, append_summary, log_configuration, write_json

DEFAULTS = dict(schedule='offline', offline_strategy='split_merge', iterations=20,
    split_merge_refine_steps=3, exploration_steps=8, max_clusters=10,
    hdp_preset='balanced', lds_degrees=5, fit_gp=False, alignment='none',
    hard_assignments=False, projection_plots=True, lds_plots=False,
    lds_dynamics_plots=False, warp_plots=False)
REPRESENTATIONS = ('autoencoder', 'pca', 'fourier', 'wavelet')
BASIS_REPRESENTATIONS = ('fourier', 'wavelet')


def _log_emission_summary(representation, record, components, calibration_variance,
                          record_variance, channel_components=None):
    from ..diagnostics._xjtu_sy import representation_label
    def percentage(value):
        return 'n/a' if value is None else f'{value:.2%}'
    channels = ('' if channel_components is None else
                f' (horizontal={channel_components[0]}, vertical={channel_components[1]})')
    print(f'Emission={representation_label(representation)}; components={components}{channels}; '
          f'variance explained on scaled raw windows: calibration={percentage(calibration_variance)}, '
          f'{record}={percentage(record_variance)}', flush=True)


def build_parser(description=None, defaults=None, *, representation='autoencoder'):
    if representation not in REPRESENTATIONS:
        raise ValueError(f'XJTU-SY representation must be one of {REPRESENTATIONS}')
    parser = argparse.ArgumentParser(description=description or __doc__)
    basis_defaults = dict(dynamics_clock='physical-clock') if representation in BASIS_REPRESENTATIONS else {}
    add_common_arguments(parser, defaults={**DEFAULTS, **basis_defaults, **(defaults or {})}, ordered=True)
    parser.add_argument('--data-dir', type=Path, default=Path.cwd()/'agents/data/xjtu_sy')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--output-dir', '--plot-dir', type=Path, default=Path.cwd()/f'agents/xjtu_sy_{representation}_runs')
    parser.add_argument('--records', nargs='+', choices=tuple(XJTU_RECORDS), default=['Bearing1_1'])
    parser.add_argument('--calibration-records', nargs='+', choices=tuple(XJTU_RECORDS))
    parser.add_argument('--window-samples', type=int, default=512,
        help='Raw samples per channel from each 32768-sample burst; no cross-burst windowing.')
    parser.add_argument('--window-offset', type=int, default=0,
        help='Same zero-based raw-sample offset inside every burst.')
    parser.add_argument('--step', '--acquisition-step', type=int, default=1,
        help='Advance in numbered acquisitions (minutes), not raw vibration samples.')
    parser.add_argument('--max-series', type=int, help='Chronological acquisition prefix, also used for calibration.')
    parser.add_argument('--components', '--latent-dim', type=int, default=8,
        help='Total latent dimension; Fourier/Haar divide the budget between the two channels.')
    parser.add_argument('--calibration-samples', type=int, default=400)
    parser.add_argument('--epochs', '--ae-epochs', type=int, default=250)
    parser.add_argument('--learning-rate', type=float, default=.02)
    parser.add_argument('--noise-floor', type=float, default=1e-6)
    parser.add_argument('--noise-floor-fraction', type=float, default=.05)
    parser.add_argument('--sigma-prior', type=float, default=DEFAULT_SIGMA_PRIOR)
    parser.add_argument('--gamma-prior', type=float, default=DEFAULT_GAMMA_PRIOR)
    parser.add_argument('--sigma-mode', choices=NOISE_MODES, default='fixed')
    parser.add_argument('--configurations', nargs='+', choices=('independent','dynamic'), default=['dynamic'])
    parser.add_argument('--hmm-switch', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--trajectory-plots', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--stability-samples', type=int, default=200)
    if representation == 'fourier':
        parser.add_argument('--window-size', '--fourier-window-size', type=int,
            help='Fourier block length per channel; defaults to the full --window-samples window.')
        bins = parser.add_mutually_exclusive_group()
        bins.add_argument('--harmonics', type=int, help='Candidate bins 0..h in each Fourier block.')
        bins.add_argument('--frequency-bins', nargs='+', type=int,
            help='Candidate local Fourier bins; calibration selects whole sine/cosine pairs.')
    if representation == 'wavelet':
        parser.add_argument('--wavelet', choices=('haar',), default='haar')
    parser.set_defaults(dataset='xjtu_sy', representation=representation, **(defaults or {}))
    return parser


def _validate(args):
    validate_common(args); validate_affine(args)
    if args.schedule != 'offline' or args.normalize_samples or args.birth_plots or args.warp_plots:
        raise ValueError('XJTU-SY requires offline inference, preserved amplitudes and no warp/birth plots')
    if args.transition_topology == 'left_to_right' and not args.hmm_switch:
        raise ValueError('left_to_right requires HMM assignments')
    basis = args.representation in BASIS_REPRESENTATIONS
    if (not 2 <= args.window_samples <= min(2048, XJTU_SAMPLES) or args.window_offset < 0
            or args.window_offset+args.window_samples > XJTU_SAMPLES or args.step < 1
            or args.components < 1 or args.calibration_samples < 2
            or args.stability_samples < 0
            or (args.max_series is not None and args.max_series < 2)
            or (not basis and (args.components >= 2*args.window_samples
                or args.calibration_samples <= args.components
                or (args.max_series is not None and args.max_series <= args.components)))):
        raise ValueError('Invalid raw window, acquisition step, rank or budget (affine windows limited to 2048 samples/channel)')
    if basis:
        from ._xjtu_basis import validate_basis_options
        validate_basis_options(args)
    if any(not np.isfinite(value) or value <= 0 for value in
           (args.noise_floor,args.sigma_prior,args.gamma_prior)):
        raise ValueError('Covariance scales must be finite and positive')
    if args.representation == 'autoencoder' and (args.epochs < 1
            or not np.isfinite(args.learning_rate) or args.learning_rate <= 0):
        raise ValueError('Autoencoder epochs and learning rate must be finite and positive')
    if not np.isfinite(args.noise_floor_fraction) or args.noise_floor_fraction < 0:
        raise ValueError('Noise floor fraction must be finite and nonnegative')


def calibrate_windows(records, names, args):
    raw = np.concatenate([records[name].values.reshape(-1,2) for name in names])
    center, scale = raw.mean(0), raw.std(0)
    scale = np.where(scale > np.finfo(float).eps, scale, 1.)
    if args.calibration_samples < len(names):
        raise ValueError('Calibration budget must cover every calibration bearing')
    rng = np.random.default_rng(args.seed)
    batches, provenance, acquisitions = [], [], []
    for i,name in enumerate(names):
        data = records[name]
        quota = args.calibration_samples//len(names)+(i < args.calibration_samples % len(names))
        chosen = np.sort(rng.choice(len(data.values),min(quota,len(data.values)),replace=False))
        batch = (data.values[chosen].reshape(-1,args.window_samples,2)-center)/scale
        batches.append(batch.reshape(len(chosen),-1))
        provenance.extend([name]*len(chosen)); acquisitions.extend(data.acquisition_numbers[chosen])
    batch = np.concatenate(batches)
    if args.representation not in BASIS_REPRESENTATIONS and len(batch) <= args.components:
        raise ValueError('Too few calibration acquisitions for requested latent dimension')
    grid = np.arange(batch.shape[1],dtype=float)[:,None]
    if args.representation == 'pca':
        floor = max(args.noise_floor,args.noise_floor_fraction*float(np.var(batch)))
        estimate = estimate_pca_observation(batch,grid,components=args.components,
            max_beats=len(batch),random_state=args.seed,noise_floor=floor)
        decoder = AffineGaussianObservation(estimate.components,estimate.residual_covariance,
            grid,estimate.mean,np.diag(estimate.latent_variances))
    elif args.representation == 'autoencoder':
        estimate = estimate_autoencoder_observation(batch,grid,
            components=args.components,max_samples=len(batch),epochs=args.epochs,
            learning_rate=args.learning_rate,random_state=args.seed,noise_floor=args.noise_floor,
            noise_floor_fraction=args.noise_floor_fraction,noise_structure='diagonal')
        decoder = estimate.model
    else:
        from ._xjtu_basis import estimate_basis_observation
        estimate = estimate_basis_observation(batch, args)
        decoder = estimate.model
    return estimate,decoder,center,scale,batch,np.asarray(provenance),np.asarray(acquisitions)



def _fit_record(data,args,run,configuration,calibration,calibration_names):
    from ..diagnostics._xjtu_sy import (xjtu_diagnostics, plot_xjtu_summary, plot_xjtu_embedding,
        plot_xjtu_basis_calibration, representation_label, emission_reconstruction)
    from ..diagnostics._calibration_plots import plot_autoencoder_calibration
    estimate,decoder,center,scale,batch,sample_records,sample_acquisitions = calibration
    values=((data.values.reshape(-1,args.window_samples,2)-center)/scale).reshape(len(data.values),-1)
    from ._prior_screen import screen_experiment
    screened=screen_experiment(values[:,:,None],decoder.grid,args,run,
        lambda candidate:make_config(candidate,decoder,dynamic=configuration=='dynamic'))
    if screened is not None:
        return screened
    config=make_config(args,decoder,dynamic=configuration=='dynamic')
    np.savez_compressed(run.path/'prepared_data.npz',raw_windows=data.values,scaled_windows=values,
        acquisition_numbers=data.acquisition_numbers,window_times=data.times,
        window_starts=data.window_starts,window_ends=data.window_ends,
        acquisition_rms=data.acquisition_rms,acquisition_peak=data.acquisition_peak,
        source_file_sha256=np.asarray(data.file_sha256))
    state=dict(**decoder.state_dict(),applied_Sigma=config.observation.Sigma_policy.value,
        channel_center=center,channel_scale=scale,calibration_records=sample_records,
        calibration_acquisitions=sample_acquisitions)
    if args.representation == 'pca':
        state.update(calibration_indices=estimate.beat_indices,latent_variances=estimate.latent_variances,
                     explained_variance_ratio=estimate.explained_variance_ratio)
    elif args.representation == 'autoencoder':
        state.update(losses=estimate.losses,calibration_indices=estimate.indices)
    else:
        state.update(calibration_indices=estimate.indices,
            candidate_energy=estimate.candidate_energy,
            coefficient_channels=np.asarray([item['channel'] for item in estimate.coefficient_metadata]),
            coefficient_indices=np.asarray([item['basis_column'] for item in estimate.coefficient_metadata]),
            explained_variance_ratio=estimate.explained_variance_ratio)
        write_json(run.path/'basis.json',dict(representation=args.representation,
            selection='Maximum calibration energy within each channel at the requested rank',
            fourier_pairs_retained_together=args.representation=='fourier',
            candidate_columns_per_channel=estimate.candidate_energy.shape[1],
            retained_columns_per_channel=[len(columns) for columns in estimate.selected_columns],
            sample_rate_hz=XJTU_SAMPLE_RATE,coefficients=estimate.coefficient_metadata))
    np.savez_compressed(run.path/f'{args.representation}_calibration.npz',**state)
    label='linear AE' if args.representation == 'autoencoder' else representation_label(args.representation)
    log_configuration(config,args,source=f'unlabeled raw acquisition-window {label}; common channel scaling')
    calibration_variance=emission_reconstruction(batch,decoder)[2]
    record_variance=emission_reconstruction(values,decoder)[2]
    channel_components=([len(columns) for columns in estimate.selected_columns]
                        if args.representation in BASIS_REPRESENTATIONS else None)
    _log_emission_summary(args.representation,data.name,decoder.latent_dim,
                          calibration_variance,record_variance,channel_components)
    print(f'XJTU-SY {data.name} {configuration}: {len(values)} acquisitions, '
          f'{values.shape[1]} raw features, {decoder.latent_dim} latent dimensions; '
          f'one step advances {args.step:g} acquisition minute(s), not continuous vibration.')
    model=HDPGPC(config).fit(values[:,:,None],grid=decoder.grid)
    paths=model_latent_trajectories(model)
    np.savez_compressed(run.path/'latent_trajectories.npz',times=data.times,
        **{f'cluster_{cluster}_{name}':value for cluster,path in paths.items() for name,value in path.items()})
    diagnostics,codes=xjtu_diagnostics(model,data,values,decoder,center,scale,paths,
        samples=args.stability_samples,seed=args.seed,representation=args.representation)
    write_json(run.path/'dynamics.json',diagnostics)
    complete=bool(np.array_equal(data.acquisition_numbers,np.arange(1,data.published_acquisitions+1)))
    metadata=dict(configuration=configuration,representation=('tied_linear_orthonormal_autoencoder'
        if args.representation=='autoencoder' else args.representation),
        source=XJTU_SOURCE,source_fingerprint=data.source_fingerprint,
        reference_scope='Terminal bearing-level fault inspection; no pointwise health states',
        fault_element=data.fault_element,published_acquisitions=data.published_acquisitions,
        complete_run_to_failure=complete,terminal_acquisition_included=bool(data.acquisition_numbers[-1]==data.published_acquisitions),
        rpm=data.rpm,radial_load_kn=data.radial_load_kn,sample_rate_hz=XJTU_SAMPLE_RATE,
        burst_samples=XJTU_SAMPLES,acquisition_period_seconds=XJTU_PERIOD_SECONDS,
        time_origin='First scheduled acquisition; file n begins at relative (n-1)*60 seconds',
        acquisition_step=args.step,window_samples=args.window_samples,window_offset=args.window_offset,
        components=decoder.latent_dim,preliminary_projection=False,per_window_normalization=False,
        observation_layout='One within-burst raw window, time-major with horizontal/vertical interleaved',
        calibration_records=calibration_names,calibration='Balanced unlabeled retrospective acquisitions',
        calibration_explained_variance_ratio=calibration_variance,
        calibration_windows=len(batch),channel_center=center,channel_scale=scale,
        rms_peak_usage='Full-burst display metadata only; never model or calibration inputs',
        dynamics_clock=args.dynamics_clock,
        dynamics_clock_description=('Global regular sampling ticks' if args.dynamics_clock=='physical-clock'
                                    else 'Successive retained observations; gap validity recorded in dynamics.json'),
        interpretation='Phase-unsynchronized acquisition-window evolution, not fast mechanical mode identification or a validated remaining-life predictor',
        diagnostics=diagnostics)
    if args.representation in BASIS_REPRESENTATIONS:
        metadata.update(basis_selection='Unlabeled calibration energy; separate sensor bases',
            components_per_channel=channel_components,
            basis_specification='basis.json',phase_preserved=True,
            fourier_window_size=(args.window_size or args.window_samples) if args.representation=='fourier' else None,
            wavelet=args.wavelet if args.representation=='wavelet' else None)

    def summary_plotter(*unused,**kwargs):
        return [plot_xjtu_summary(data,model.labels_,diagnostics,title=f'{data.name}: {configuration}',
                                 representation=args.representation)]

    def figures():
        result={}
        if args.trajectory_plots:
            result['latent_trajectories']=plot_latent_trajectories(model,times=data.times/60,
                title=f'XJTU-SY {data.name}: {label} {configuration}; '+(
                    'global sampling steps' if args.dynamics_clock=='physical-clock' else 'retained-acquisition steps'),trajectories=paths,
                min_cluster_samples=args.min_cluster_samples,time_label='Acquisition time (min)',
                time_unit='min',observation_label='acquisitions')
        if args.projection_plots:
            result['latent_embedding']=plot_xjtu_embedding(codes,model.labels_,data.times,
                title=f'XJTU-SY {data.name}: common {label} coordinates',representation=args.representation)
        if args.representation == 'pca':
            from ..diagnostics._xjtu_sy import plot_xjtu_pca_calibration
            result['pca_calibration']=plot_xjtu_pca_calibration(batch,estimate)
        elif args.representation == 'autoencoder':
            figure=plot_autoencoder_calibration(batch,estimate.network,estimate.indices,estimate.losses)
            figure.axes[1].set(xlabel='Flattened raw acquisition-window coordinates',ylabel='Scaled acceleration')
            result['autoencoder_calibration']=figure
        else:
            result[f'{args.representation}_calibration']=plot_xjtu_basis_calibration(batch,estimate,
                representation=args.representation,sample_rate=XJTU_SAMPLE_RATE)
        return result

    return save_results(run,model,values[:,:,None],None,decoder.grid,args,ordered=True,metadata=metadata,
        codes=codes,row_metadata=dict(second=data.times,acquisition_number=data.acquisition_numbers,
            minute=data.times/60,window_start_sample=data.window_starts,window_end_sample=data.window_ends),
        summary_plotter=summary_plotter,extra_figures=figures)


def run_xjtu_sy(args):
    import torch
    torch.set_num_threads(1)
    _validate(args)
    names=list(dict.fromkeys(args.records))
    calibration_names=list(dict.fromkeys(args.calibration_records or names))
    records={}
    for name in dict.fromkeys(names+calibration_names):
        if args.download:
            download_xjtu_sy(args.data_dir,name,step=args.step,max_series=args.max_series)
        records[name]=load_xjtu_sy_windows(args.data_dir,name,window_samples=args.window_samples,
            window_offset=args.window_offset,step=args.step,max_series=args.max_series)
        if args.representation not in BASIS_REPRESENTATIONS and len(records[name].values)<=args.components:
            raise ValueError(f'{name}: too few acquisitions for requested latent rank')
    fingerprint=hashlib.sha256(json.dumps({name:records[name].source_fingerprint
        for name in sorted(records)},sort_keys=True).encode()).hexdigest()
    calibration=None; reports=[]
    for name in names:
        for configuration in dict.fromkeys(args.configurations):
            settings=experiment_settings(args);settings.pop('configurations',None)
            settings.update(record=name,configuration=configuration,calibration_records=calibration_names)
            run=ExperimentRun(args.output_dir,f'xjtu_sy_{name}',f'xjtu_sy_{args.representation}_{configuration}',settings,fingerprint)
            if args.resume and run.complete():
                report=json.loads((run.path/'results.json').read_text())
                print(f'Using saved results from: {run.path.resolve()}')
                if report.get('mode') != 'prior_screen':
                    metadata=report['metadata']
                    _log_emission_summary(args.representation,name,metadata['components'],
                        metadata.get('calibration_explained_variance_ratio'),
                        metadata.get('diagnostics',{}).get('reconstruction_explained_variance'),
                        metadata.get('components_per_channel'))
            else:
                try:
                    with run:
                        print(f'Results directory: {run.path.resolve()}')
                        if calibration is None:
                            calibration=calibrate_windows(records,calibration_names,args)
                        report=_fit_record(records[name],args,run,configuration,calibration,calibration_names)
                        print(f'Results saved to: {run.path.resolve()}')
                except Exception:
                    report=json.loads((run.path/'results.json').read_text())
            append_summary(args.output_dir,report);reports.append(report)
    write_json(args.output_dir/'comparison.json',dict(source=XJTU_SOURCE,representation=args.representation,records=names,
        calibration_records=calibration_names,reference_scope='No intermediate health-state targets',
        cluster_ids='Local to each bearing and configuration',runs=[{
            key:report.get(key) for key in ('dataset','run_id','experiment','status','clusters','seconds','output_directory','error')}
            | {'diagnostics':report.get('metadata',{}).get('diagnostics',{})} for report in reports]))
    if args.plots and not getattr(args, 'screen_priors', False) and all(report['status']=='ok' for report in reports):
        import matplotlib.pyplot as plt
        from ..diagnostics._xjtu_sy import plot_xjtu_overview
        figure=plot_xjtu_overview(reports)
        figure.savefig(args.output_dir/'comparison.png',dpi=150)
        plt.close(figure)
    print(f'XJTU-SY comparison saved to: {args.output_dir.resolve()}')
    return reports
