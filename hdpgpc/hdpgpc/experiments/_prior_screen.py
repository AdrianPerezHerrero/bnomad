"""Shared screen-only entry point for prepared experiment observations."""
from copy import copy
import csv


def add_screen_arguments(parser):
    group = parser.add_argument_group('Fast prior screening (no full-data fit)')
    group.add_argument('--screen-priors', action='store_true',
                       help='Calibrate and screen prior pairs, save recommendations, then stop before the full fit.')
    group.add_argument('--screen-seconds', type=float, default=120.,
                       help='Screening time budget checked between fits; excludes representation calibration (default: 120).')
    group.add_argument('--screen-samples', type=int, default=128,
                       help='Maximum training rows per independent chronological clip (default: 128).')
    group.add_argument('--screen-validation', type=int, default=32,
                       help='Chronological validation rows after each training clip (default: 32).')
    group.add_argument('--screen-blocks', type=int, default=5,
                       help='Independent clips spread across the record (default: 5).')
    group.add_argument('--screen-iterations', type=int, default=2)
    group.add_argument('--screen-grid-points', type=int, default=5,
                       help='Logarithmic values per active prior axis (default: 5, a 25-pair grid).')
    group.add_argument('--screen-max-fits', type=int, default=60,
                       help='Total pilot-fit budget, including refinement and repeat checks (default: 60).')
    group.add_argument('--screen-sigma-range', type=float, nargs=2, metavar=('LOW', 'HIGH'),
                       help='Override sigma screening bounds (default: 0.001 to 10.0).')
    group.add_argument('--screen-gamma-range', type=float, nargs=2, metavar=('LOW', 'HIGH'),
                       help='Override gamma screening bounds (default: 0.001 to 10.0).')


def options_from_args(args):
    from ..model_selection import NoisePriorScreenConfig
    return NoisePriorScreenConfig(train_size=args.screen_samples,
        validation_size=args.screen_validation, blocks=args.screen_blocks,
        iterations=args.screen_iterations, grid_points=args.screen_grid_points,
        max_fits=args.screen_max_fits,
        seconds=args.screen_seconds, sigma_range=args.screen_sigma_range,
        gamma_range=args.screen_gamma_range, seed=args.seed)


def validate_screen_arguments(args):
    if not getattr(args, 'screen_priors', False):
        return
    options_from_args(args)
    if getattr(args, 'calibrate_lds_noise_priors', False):
        raise ValueError('--screen-priors replaces --calibrate-lds-noise-priors; select one')
    if getattr(args, 'alignment', 'none') not in ('none', None) or getattr(args, 'warp', False):
        raise ValueError('--screen-priors requires --alignment none and no input warping')


def screen_experiment(values, grid, args, run, config_builder):
    """Return a standard screen report, or None for the unchanged normal workflow.

    config_builder receives a copy of args with only sigma/gamma changed. Its
    closed-over calibration must already be frozen. No label data are accepted.
    """
    if not getattr(args, 'screen_priors', False):
        return None
    from ..model_selection import screen_noise_priors
    from ._runs import write_json

    def factory(sigma, gamma):
        candidate = copy(args)
        candidate.sigma_prior, candidate.gamma_prior = sigma, gamma
        return config_builder(candidate)

    def progress(item):
        if args.verbose != 'silent':
            print(f"Prior screen {item['fit']}/{args.screen_max_fits}: "
                  f"sigma={item['sigma_prior']:.5g}, gamma={item['gamma_prior']:.5g}; "
                  f"{item['stage']} {item['status']} ({item['seconds']:.2f}s)", flush=True)

    base = factory(args.sigma_prior, args.gamma_prior)
    (run.path/'config.yaml').write_text(base.to_yaml(), encoding='utf-8')
    result = screen_noise_priors(values, factory, grid=grid,
        sigma_prior=args.sigma_prior, gamma_prior=args.gamma_prior,
        options=options_from_args(args), progress=progress)
    result['dataset'] = run.dataset
    result['run_id'] = run.run_id
    write_json(run.path/'prior_screen.json', result)
    columns = ('candidate', 'sigma_prior', 'gamma_prior', 'successful_fits',
               'mean_nll_per_feature', 'nll_standard_error', 'seed_assignment_agreement_ari',
               'nearby_assignment_agreement_ari', 'status')
    with (run.path/'prior_screen.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader(); writer.writerows(result['candidates'])
    artifacts = ['config.yaml', 'prior_screen.json', 'prior_screen.csv']
    if args.plots:
        _plot_screen(result, run.path/'prior_screen.png')
        artifacts.append('prior_screen.png')
    recommendation = result['recommendation']
    if recommendation is not None:
        print(f"Recommended screened pair: {recommendation['cli']} ({recommendation['confirmation']})", flush=True)
    print(f'Prior screening results saved to: {run.path.resolve()}', flush=True)
    report = {'schema_version': 1, 'dataset': run.dataset, 'run_id': run.run_id,
        'experiment': run.experiment, 'status': 'ok' if recommendation is not None else 'error',
        'mode': 'prior_screen', 'fitted': 0, 'metrics': {}, 'seconds': result['seconds'],
        'output_directory': str(run.path), 'artifacts': artifacts,
        'recommendation': recommendation, 'working_pairs': result['working_pairs']}
    if recommendation is None:
        report['error'] = 'No successful prior-screen validation fits; inspect prior_screen.json'
    write_json(run.path/'results.json', report)
    return report


def _plot_screen(result, path):
    import numpy as np
    import matplotlib.pyplot as plt
    fig, (axis, details) = plt.subplots(1, 2, figsize=(15, 6.5), layout='constrained',
                                      gridspec_kw={'width_ratios': (1.15, 1)})
    seed = result['options']['seed']
    successful, missing, colors = [], [], []
    for candidate in result['candidates']:
        # Every color uses exactly the same validation clip and training seed.
        # Do not substitute averages from candidates with additional checks.
        first = next((fit for fit in candidate['fits']
                      if fit['block'] == 0 and fit['seed'] == seed and fit['status'] == 'ok'), None)
        if first is None:
            missing.append(candidate)
        else:
            successful.append(candidate)
            colors.append(first['nll_per_feature'])
    if successful:
        scatter = axis.scatter([c['sigma_prior'] for c in successful], [c['gamma_prior'] for c in successful],
            c=colors, cmap='viridis_r', s=70)
        fig.colorbar(scatter, ax=axis, label='First-clip NLL per feature (lower is better)', fraction=.045, pad=.03)
    if missing:
        axis.scatter([c['sigma_prior'] for c in missing], [c['gamma_prior'] for c in missing],
                     marker='x', color='.6', label='No first-clip score')
    for index, pair in enumerate(result['working_pairs']):
        axis.scatter(pair['sigma_prior'], pair['gamma_prior'], facecolors='none',
                     edgecolors='black', s=200, linewidths=1.2,
                     label='Selection plateau (see right)' if index == 0 else None)
    best = result['recommendation']
    if best:
        axis.scatter(best['sigma_prior'], best['gamma_prior'], marker='*', color='#D55E00',
                     s=250, edgecolors='white', label='Recommended screened pair')
    axis.set(xscale='log', yscale='log', xlabel='--sigma-prior', ylabel='--gamma-prior',
             title=f'Initial validation: first clip, seed {seed}\nColors remain fixed after confirmation')
    axis.grid(alpha=.15)
    if best or missing or result['working_pairs']:
        axis.legend(fontsize=8)
    _plot_screen_confirmation(result, details)
    fig.suptitle(f"Prior screening: {result['fit_calls']} short fits in {result['seconds']:.1f}s", fontsize=15)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_screen_confirmation(result, axis):
    """Explain the existing selection without combining unlike color scores."""
    axis.set_axis_off()
    axis.text(0, 1, 'Confirmation results', transform=axis.transAxes, va='top', fontsize=13, weight='bold')
    promoted = [candidate for candidate in result['candidates']
                if any(fit['stage'] == 'confirm' and fit['status'] == 'ok' for fit in candidate['fits'])]
    promoted.sort(key=lambda candidate: candidate['mean_nll_per_feature'])
    best = result['recommendation']
    if not promoted:
        message = 'No successful confirmation checks completed.'
        if best:
            message += (f"\n\nProvisional recommendation:\nsigma={best['sigma_prior']:.5g}, "
                        f"gamma={best['gamma_prior']:.5g}"
                        '\n\nSelection uses the available first-clip results.')
        else:
            message += '\n\nNo valid recommendation is available.'
        axis.text(0, .89, message, transform=axis.transAxes, va='top', fontsize=10, wrap=True)
        return

    axis.text(0, .93, 'Means include all successful checks on the evaluated clips/seeds.',
              transform=axis.transAxes, va='top', fontsize=9)
    expected = 2*len(result['clips'])
    rows = []
    for candidate in promoted:
        se = candidate['nll_standard_error']
        nll = f"{candidate['mean_nll_per_feature']:.4f}"
        if se is not None:
            nll += f' ± {se:.4f}'
        agreement = candidate['seed_assignment_agreement_ari']
        checks = {(fit['block'], fit['seed']) for fit in candidate['fits'] if fit['status'] == 'ok'}
        rows.append([f"{candidate['sigma_prior']:.4g}", f"{candidate['gamma_prior']:.4g}", nll,
                     f'{agreement:.3f}' if agreement is not None else 'n/a', f'{len(checks)}/{expected}'])
    table = axis.table(cellText=rows, colLabels=('Sigma', 'Gamma', 'Mean NLL ± SE', 'Seed ARI', 'Checks'),
                       colWidths=(.14, .14, .36, .18, .18), cellLoc='center',
                       bbox=(0, .55, 1, .28))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor('.82')
        if row == 0:
            cell.set_facecolor('#eeeeee')
            cell.get_text().set_weight('bold')
        elif best and promoted[row-1]['candidate'] == best['candidate']:
            cell.set_facecolor('#fff0df')
            cell.get_text().set_weight('bold')
    axis.text(0, .51, 'Highlighted row: recommended pair. Checks: successful / expected.\n'
              'SE describes variation between fits; it is not a confidence interval.',
              transform=axis.transAxes, va='top', fontsize=9, color='.35')
    minimum = promoted[0]
    tolerance = max(minimum['nll_standard_error'] or 0., 1e-3)
    threshold = minimum['mean_nll_per_feature']+tolerance
    axis.text(0, .37, 'Selection rule', transform=axis.transAxes, va='top', fontsize=11, weight='bold')
    explanation = (f"Lowest promoted mean NLL: {minimum['mean_nll_per_feature']:.4f}\n"
                   f'Plateau: mean NLL ≤ {threshold:.4f}\n'
                   'Within that plateau: highest seed agreement, then lowest mean NLL.')
    if best:
        coverage = 'Complete' if best['confirmation_complete'] else 'Partial'
        explanation += (f"\n\nSelected: sigma={best['sigma_prior']:.5g}, gamma={best['gamma_prior']:.5g}"
                        f'\n{coverage} confirmation coverage.')
    axis.text(0, .30, explanation, transform=axis.transAxes, va='top', fontsize=10, linespacing=1.4)
    axis.text(0, 0, 'Candidates without repeat checks are not ranked by these averages.\n'
              'Untested combinations are not interpolated.', transform=axis.transAxes,
              va='bottom', fontsize=9, color='.35')
