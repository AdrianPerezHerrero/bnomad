"""HDP-GPC public API. Only the native modular implementation is supported.
Lazy imports do not initialize numerical libraries or process seeds."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config._mfa import HDPMFAConfig
    from .model._mfa import HDPMFA

# Retain the established star-import contract. New model families are available
# through explicit lazy imports and their model/config namespaces.
__all__ = ('HDPGPC', 'HDPGPCConfig', 'UpdateResult', 'Prediction', 'LatentState')


def __getattr__(name):
    if name == 'HDPMFAConfig':
        from .config._mfa import HDPMFAConfig
        return HDPMFAConfig
    if name == 'HDPMFA':
        from .model._mfa import HDPMFA
        return HDPMFA
    if name == 'HDPGPCConfig':
        from .config import HDPGPCConfig
        return HDPGPCConfig
    if name in ('HDPGPC', 'UpdateResult', 'Prediction', 'LatentState'):
        from . import model
        return getattr(model, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
