"""Research diagnostic utilities for fitted models, LDS matrices and input warps."""

__all__ = ('plot_warp_diagnostics', 'model_warp_arrays',
           'plot_clustering_summary',
           'plot_lds_diagnostics', 'model_lds_arrays',
           'plot_lds_dynamics', 'model_lds_dynamics',
           'plot_embedding_summary', 'model_latent_trajectories',
           'plot_latent_trajectories', 'plot_image_clustering_summary',
           'plot_birth_warp_diagnostic',
           'cluster_label_statistics', 'format_confusion_matrix',
           'format_lds_priors', 'Tee')


def __getattr__(name):
    if name in ('model_latent_trajectories', 'plot_latent_trajectories', 'plot_image_clustering_summary'):
        from . import _image_trajectories
        return getattr(_image_trajectories, name)
    if name == 'plot_clustering_summary':
        from ._cluster_plots import plot_clustering_summary
        return plot_clustering_summary
    if name in __all__:
        if name in ('cluster_label_statistics', 'format_confusion_matrix',
                    'format_lds_priors', 'Tee'):
            from . import _reporting
            return getattr(_reporting, name)
        if name in ('plot_warp_diagnostics', 'model_warp_arrays'):
            from . import _warp_plots
            return getattr(_warp_plots, name)
        if name == 'plot_embedding_summary':
            from . import _embedding_plots
            return getattr(_embedding_plots, name)
        if name == 'plot_birth_warp_diagnostic':
            from . import _birth_warp_plots
            return getattr(_birth_warp_plots, name)
        if name in ('plot_lds_dynamics', 'model_lds_dynamics'):
            from . import _lds_dynamics
            return getattr(_lds_dynamics, name)
        from . import _lds_plots
        return getattr(_lds_plots, name)
    raise AttributeError(name)
