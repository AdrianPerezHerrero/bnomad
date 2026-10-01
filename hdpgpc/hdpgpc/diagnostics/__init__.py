"""Research diagnostic utilities for fitted models and input warps."""

__all__ = ('plot_warp_diagnostics', 'model_warp_arrays')


def __getattr__(name):
    if name in __all__:
        from . import _warp_plots
        return getattr(_warp_plots, name)
    raise AttributeError(name)
