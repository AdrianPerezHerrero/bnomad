"""Deterministic temporal baselines; no model fitting or image features."""
from numbers import Integral

import numpy as np

from ..diagnostics import cluster_label_statistics


def equal_duration_baseline(labels, clusters, *, evaluation_mask=None):
    """Split a uniformly sampled sequence into K contiguous temporal bins.

    CAS-Colon segments have a fixed cadence, so durations differ by at most one
    frame when N is not divisible by K. Excluded reference labels still occupy
    time; only majority fitting and metric evaluation use the evaluation mask.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or not len(labels):
        raise ValueError('Baseline labels must be a nonempty one-dimensional sequence')
    if isinstance(clusters, bool) or not isinstance(clusters, Integral) or not 1 <= clusters <= len(labels):
        raise ValueError('Baseline cluster count must be an integer between 1 and the number of frames')
    evaluated = (np.ones(len(labels), dtype=bool) if evaluation_mask is None
                 else np.asarray(evaluation_mask, dtype=bool))
    if evaluated.shape != labels.shape or not evaluated.any():
        raise ValueError('Baseline evaluation mask must select aligned observations')
    assignments = np.arange(len(labels), dtype=np.int64)*int(clusters)//len(labels)
    scores, classes, confusion, mapping = cluster_label_statistics(
        assignments[evaluated], labels[evaluated])
    mapping = {k: mapping.get(k) for k in range(int(clusters))}
    scores.update(evaluation_frames=int(evaluated.sum()),
                  excluded_evaluation_frames=int((~evaluated).sum()))
    assigned_labels = np.array([mapping[int(k)] for k in assignments], dtype=object)
    segments = []
    for k in range(int(clusters)):
        indices = np.flatnonzero(assignments == k)
        segments.append(dict(cluster=k, start_index=int(indices[0]),
            stop_index_exclusive=int(indices[-1])+1, frames=int(len(indices)),
            evaluation_frames=int(evaluated[indices].sum()), majority_label=mapping[k]))
    return dict(method='equal_duration', clusters=int(clusters), fitted=len(labels),
        partition='floor(frame_index * K / N); fixed-cadence bins differ by at most one frame',
        evaluation='in-sample majority labels; same evaluation mask as fitted model',
        assignments=assignments, majority_labels=assigned_labels,
        metrics=scores, cluster_majority_labels=mapping, segments=segments,
        confusion_classes=classes, confusion=confusion)


def add_baseline_header(figure, model_scores, baseline):
    """Add the same K-matched comparison to every figure saved for a CAS run."""
    k = baseline['clusters']
    def metrics_line(name, scores):
        return (f'{name} (K={k}) | Purity {scores["purity"]:.3f} | '
                f'ARI {scores["adjusted_rand_index"]:.3f} | '
                f'NMI {scores["normalized_mutual_information"]:.3f}')
    title = figure._suptitle
    lines = title.get_text().splitlines() if title is not None else ['Clustering comparison']
    model_line = next((i for i, line in enumerate(lines)
        if all(key in line.lower() for key in ('purity', 'ari', 'nmi'))), None)
    baseline_line = metrics_line('Equal-duration baseline', baseline['metrics'])
    added_lines = 2 if model_line is None else 1
    if model_line is None:
        lines[1:1] = [metrics_line('Model', model_scores), baseline_line]
    else:
        lines.insert(model_line+1, baseline_line)
    text = '\n'.join(lines)
    if title is None:
        figure.suptitle(text, fontsize=11)
    else:
        # Preserve fixed-inch gallery title placement and each diagnostic's layout.
        title.set_text(text)
    if figure.get_layout_engine() is None:
        # Fixed-size galleries have no automatic layout to reserve header
        # space. Grow only the top margin, keeping every axes' physical size.
        title = figure._suptitle
        width, height = figure.get_size_inches()
        extra = added_lines * title.get_fontsize()/72 * 1.3 + .15
        positions = [(axis, axis.get_position(original=True).frozen()) for axis in figure.axes]
        top_margin = (1-title.get_position()[1])*height
        figure.set_size_inches(width, height+extra, forward=False)
        for axis, box in positions:
            axis.set_position([box.x0, box.y0*height/(height+extra),
                               box.width, box.height*height/(height+extra)])
        title.set_y(1-top_margin/(height+extra))


def plot_equal_duration_baseline(baseline, labels, *, times=None, title=''):
    """Reference, baseline cluster IDs and majority anatomical labels."""
    import matplotlib.pyplot as plt
    from ..diagnostics._image_trajectories import _colors
    labels = np.asarray(labels)
    supplied_times = times is not None
    times = np.arange(len(labels)) if times is None else np.asarray(times)
    classes = sorted(np.unique(labels))
    anatomical = {name: plt.get_cmap('tab10')(i % 10) for i, name in enumerate(classes)}
    assignments = baseline['assignments']
    cluster_colors = _colors(range(baseline['clusters']))
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, layout='constrained')
    axes[0].scatter(times, [classes.index(label) for label in labels],
                    c=[anatomical[label] for label in labels], s=10)
    axes[0].set_yticks(range(len(classes)), classes, fontsize=8)
    for tick, label in zip(axes[0].get_yticklabels(), classes):
        tick.set_color(anatomical[label])
    axes[0].set_title('Reference anatomical regions (evaluation only)')
    axes[1].scatter(times, assignments, c=[cluster_colors[int(k)] for k in assignments], s=10)
    axes[1].set_title('Equal-duration baseline: independent temporal cluster IDs')
    axes[2].scatter(times, assignments,
        c=[anatomical.get(label, '0.6') for label in baseline['majority_labels']], s=10)
    axes[2].set_title('Equal-duration baseline colored by majority label')
    for axis in axes[1:]:
        axis.set(ylabel='Cluster', yticks=range(baseline['clusters']))
    axes[2].set_xlabel('Video time (s)' if supplied_times else 'Observation index')
    fig.suptitle(f'{title} | Equal-duration baseline | {len(labels)} frames', fontsize=12)
    return fig
