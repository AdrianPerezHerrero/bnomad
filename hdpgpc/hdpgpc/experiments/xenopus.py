"""Xenopus dataset entry points for the common expression comparison."""
import argparse

from ..datasets.xenopus import download_xenopus, load_xenopus_expression
from ._expression import build_parser as _build_parser
from ._expression import _validate, run_expression_comparison


def build_parser(description=None, defaults=None, *, representation='mfa'):
    parser = _build_parser(description or __doc__, defaults, dataset='xenopus', representation=representation)
    parser.add_argument('--clutches', '--records', nargs='+', choices=('A', 'B'), default=['A'],
                        help='Independent polyA+ time courses; fit each clutch separately.')
    parser.add_argument('--exclude-qc-failures', action=argparse.BooleanOptionalAction, default=True,
                        help='Exclude samples identified as poor quality in the original paper (default).')
    parser.add_argument('--include-qc-failures', dest='exclude_qc_failures', action='store_false',
                        help='Retain the original paper\'s flagged samples for sensitivity analysis.')
    if defaults:
        parser.set_defaults(**defaults)
    return parser


def run_xenopus(args, *, sigma_prior_defaults=None):
    """Resolve optional per-clutch noise defaults before validation and saving."""
    clutch_runs = []
    for clutch in dict.fromkeys(args.clutches):
        options = argparse.Namespace(**vars(args))
        if options.sigma_prior is None:
            if sigma_prior_defaults is None or clutch not in sigma_prior_defaults:
                raise ValueError(f'No default sigma-prior for clutch {clutch}')
            options.sigma_prior = sigma_prior_defaults[clutch]
        _validate(options)
        clutch_runs.append((clutch, options))
    if args.download:
        download_xenopus(args.data_dir, clutches=args.clutches)
    reports = []
    for clutch, options in clutch_runs:
        data = load_xenopus_expression(args.data_dir, clutch,
            exclude_qc_failures=options.exclude_qc_failures)
        reports.extend(run_expression_comparison(data, options,
            comparison_name=f'comparison_clutch_{clutch}'))
    return reports
