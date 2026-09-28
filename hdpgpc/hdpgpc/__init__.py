"""HDP-GPC public API. Only the native modular implementation is supported.
Lazy imports do not initialize numerical libraries or process seeds."""

__all__ = ('HDPGPC', 'HDPGPCConfig', 'UpdateResult', 'Prediction', 'LatentState')


def __getattr__(name):
    if name == 'HDPGPCConfig':
        from .config import HDPGPCConfig
        return HDPGPCConfig
    if name in ('HDPGPC', 'UpdateResult', 'Prediction', 'LatentState'):
        from . import model
        return getattr(model, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
