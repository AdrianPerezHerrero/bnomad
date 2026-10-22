"""Public model and immutable result types. Native state modules are private."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._mfa import HDPMFA
    from ._mfa_results import MFAFactors

__all__ = ('HDPGPC', 'HDPMFA', 'MFAFactors', 'UpdateResult', 'Prediction', 'LatentState', 'WarpPosterior')


def __getattr__(name):
    if name == 'HDPMFA':
        from ._mfa import HDPMFA
        return HDPMFA
    if name == 'MFAFactors':
        from ._mfa_results import MFAFactors
        return MFAFactors
    if name == 'HDPGPC':
        from ._api import HDPGPC
        return HDPGPC
    if name in ('UpdateResult', 'Prediction', 'LatentState', 'WarpPosterior'):
        from . import _results
        return getattr(_results, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
