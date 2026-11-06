"""Plot-only cluster selection based on final MAP membership counts."""
from numbers import Integral

import numpy as np


def cluster_plot_ids(assignments, min_cluster_samples=1):
    """Return sorted IDs with at least the requested number of assigned rows."""
    if (isinstance(min_cluster_samples, bool)
            or not isinstance(min_cluster_samples, Integral) or min_cluster_samples < 1):
        raise ValueError('min_cluster_samples must be a positive integer')
    ids, counts = np.unique(np.asarray(assignments).reshape(-1), return_counts=True)
    return ids[counts >= min_cluster_samples]


def empty_cluster_figure(title, min_cluster_samples):
    """Explain an empty plot selection instead of constructing a zero-row grid."""
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(10, 3), constrained_layout=True)
    axis.axis('off')
    axis.text(.5, .5, f'No clusters have at least {min_cluster_samples} assigned samples.',
              ha='center', va='center', transform=axis.transAxes)
    figure.suptitle(title)
    return figure


def uses_physical_clock(model):
    """Whether represented emissions actually retain physical-clock histories.

    Static independent factors can disable the clock despite the recipe setting.
    """
    from ..dynamics import _clock
    dynamics = getattr(model.config, 'dynamics', None)
    return (getattr(dynamics, 'clock', 'cluster-clock') == 'physical-clock' and
            all(_clock.enabled(output[int(cluster)])
                for output in model._backend.gpmodels for cluster in model.cluster_ids_))


def physical_clock_endpoints(model):
    """Last final-MAP observation per cluster for read-only early-stop plots."""
    if not uses_physical_clock(model):
        raise ValueError('early_stop requires physical-clock dynamics')
    labels = np.asarray(model.labels_, dtype=int)
    return {int(cluster): int(np.flatnonzero(labels == cluster)[-1])
            for cluster in np.unique(labels)}
