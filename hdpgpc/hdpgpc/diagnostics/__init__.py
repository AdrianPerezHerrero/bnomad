"""Research diagnostic utilities for fitted models, LDS matrices and input warps."""

__all__ = ('plot_warp_diagnostics', 'model_warp_arrays',
           'plot_lds_diagnostics', 'model_lds_arrays',
           'cluster_label_statistics', 'format_confusion_matrix',
           'format_lds_priors', 'Tee')


def __getattr__(name):
    if name in __all__:
        if name in ('cluster_label_statistics', 'format_confusion_matrix',
                    'format_lds_priors', 'Tee'):
            from . import _reporting
            return getattr(_reporting, name)
        if name in ('plot_warp_diagnostics', 'model_warp_arrays'):
            from . import _warp_plots
            return getattr(_warp_plots, name)
        from . import _lds_plots
        return getattr(_lds_plots, name)
    raise AttributeError(name)
