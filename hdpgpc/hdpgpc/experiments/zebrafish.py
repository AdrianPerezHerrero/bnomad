"""Zebrafish dataset entry points for the shared expression experiment."""
from ..datasets import download_zebrafish, load_zebrafish_expression
from ._expression import build_parser as _build_parser
from ._expression import make_config, _validate, run_expression_comparison


def build_parser(description=None, defaults=None, *, representation='mfa'):
    return _build_parser(description, defaults, dataset='zebrafish', representation=representation)


def run_zebrafish(args):
    _validate(args)
    if args.download:
        download_zebrafish(args.data_dir)
    data = load_zebrafish_expression(args.data_dir)
    return run_expression_comparison(data.prepare_timecourse(), args)
