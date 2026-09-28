"""Optional identity/GP alignment and the native AMTGP implementation."""
__all__ = ('GaussianProcessWarp', 'GPAlignment', 'NoAlignment')

def __getattr__(name):
    if name in ('GPAlignment', 'NoAlignment'):
        from . import _components
        return getattr(_components, name)
    if name == 'GaussianProcessWarp':
        from ._amtgp import GaussianProcessWarp
        return GaussianProcessWarp
    raise AttributeError(name)
