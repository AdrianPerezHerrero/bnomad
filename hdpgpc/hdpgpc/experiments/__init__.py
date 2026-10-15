"""Comparable experiment execution and reporting, separate from model inference.

Dataset preparation lives in datasets/preprocessing; observation models live in
emissions. This package owns CLI conventions, run provenance and artifact output.
"""

__all__ = ('ExperimentRun', 'save_results', 'add_common_arguments')


def __getattr__(name):
    if name == 'add_common_arguments':
        from ._arguments import add_common_arguments
        return add_common_arguments
    if name in ('ExperimentRun', 'save_results'):
        from . import _runs
        return getattr(_runs, name)
    raise AttributeError(name)
