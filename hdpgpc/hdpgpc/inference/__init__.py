"""Explicit inference schedules over the existing HDP-GPC model state.

Public engines bind an owner without initializing numerical libraries. Private
schedule modules preserve research call order and delegate mathematics through
that owner to existing components. Legacy model entry points remain supported.
"""

from ._base import InferenceEngine

__all__ = ("InferenceEngine", "OnlineInference", "OfflineInference", "MFAOfflineInference")


def __getattr__(name):
    if name == 'MFAOfflineInference':
        from ._mfa import MFAOfflineInference
        return MFAOfflineInference
    if name == "OnlineInference":
        from ._engines import OnlineInference

        return OnlineInference
    if name == "OfflineInference":
        from ._engines import OfflineInference

        return OfflineInference
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
