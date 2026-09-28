"""Public model and immutable result types. Native state modules are private."""

__all__ = ('HDPGPC', 'UpdateResult', 'Prediction', 'LatentState')


def __getattr__(name):
    if name == 'HDPGPC':
        from ._api import HDPGPC
        return HDPGPC
    if name in ('UpdateResult', 'Prediction', 'LatentState'):
        from . import _results
        return getattr(_results, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
