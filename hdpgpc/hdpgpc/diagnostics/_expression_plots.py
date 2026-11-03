"""Temporal expression-state reports; genes are never selected for display."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _categorical_strip(axis, labels, *, title, palette):
    labels = np.asarray(labels)
    classes = list(dict.fromkeys(labels.tolist()))
    codes = np.array([classes.index(label) for label in labels])
    axis.imshow(codes[None], aspect='auto', interpolation='nearest', cmap=palette,
                vmin=-.5, vmax=max(len(classes)-.5, .5))
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    ends = np.r_[starts[1:], len(labels)]
    positions = ([(i, label) for i, label in enumerate(labels)] if len(labels) <= 25 else
                 [((start+end-1)/2, labels[start]) for start, end in zip(starts, ends)])
    for i, label in positions:
        axis.text(i, 0, str(label), ha='center', va='center', fontsize=7, rotation=35,
                  bbox=dict(facecolor='white', edgecolor='none', alpha=.75, pad=1))
    axis.set(yticks=[], title=title)


def plot_expression_clustering_summary(model, values, annotations, *, stages, hours,
                                       metrics=None, record='', projected=True,
                                       expression_units='log1p(CPM)', reference_title='Reference labels', **kwargs):
    """Shared save_results callback: observation heatmap and label/state strips.

    The horizontal coordinate is sampling-stage index, explicitly labelled with
    actual hours. Unequal intervals are never presented as equal elapsed time.
    """
    scores = np.asarray(values)[:, :, 0]
    figure, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True,
        gridspec_kw={'height_ratios': [4, 1, 1]}, constrained_layout=True)
    bound = max(float(np.abs(scores).max()), np.finfo(float).eps)
    image = axes[0].imshow(scores.T, aspect='auto', cmap='RdBu_r', vmin=-bound,
                           vmax=bound, interpolation='nearest')
    figure.colorbar(image, ax=axes[0], label='PCA score (one common scale)' if projected else expression_units)
    axes[0].set(ylabel='All-gene principal component' if projected else 'Gene index',
                title='Expression observations used by the model')
    _categorical_strip(axes[1], annotations, title=f'{reference_title} (evaluation only)', palette='tab20')
    _categorical_strip(axes[2], model.labels_, title='Inferred expression state', palette='tab20b')
    ticks = np.unique(np.linspace(0, len(scores)-1, min(18, len(scores)), dtype=int))
    axes[2].set(xticks=ticks,
        xticklabels=[f'{stages[i]}\n{hours[i]:g} h' for i in ticks],
        xlabel='Sampling stage (unequal time intervals; approximate hours post-fertilisation)')
    plt.setp(axes[2].get_xticklabels(), rotation=60, ha='right', fontsize=8)
    metric_text = '' if metrics is None else (
        f' | purity={metrics["purity"]:.3f}, ARI={metrics["adjusted_rand_index"]:.3f}, '
        f'NMI={metrics["normalized_mutual_information"]:.3f}')
    figure.suptitle(f'{record} | {model.n_clusters_} expression states{metric_text}')
    return [figure]


def plot_expression_comparison(reports, stages, *, title='Zebrafish'):
    """Paired scores, reference segmentation, and aligned inferred state paths."""
    figure, axes = plt.subplots(2 + len(reports), 1,
        figsize=(15, 5.6 + 1.6 * len(reports)), constrained_layout=True,
        gridspec_kw={'height_ratios': [3] + [1]*(1 + len(reports))})
    axes = np.atleast_1d(axes)
    names = [report['metadata']['configuration'] for report in reports]
    metrics = ('purity', 'adjusted_rand_index', 'normalized_mutual_information')
    x = np.arange(len(metrics))
    width = .75 / len(reports)
    reference = reports[0]['metadata'].get('target_labels', {}).get('broad', 'Broad reference targets')
    reference_labels = None
    for i, (name, report) in enumerate(zip(names, reports)):
        scores = report['metadata']['label_metrics']['broad']
        axes[0].bar(x + (i-(len(reports)-1)/2)*width,
                    [scores[key] for key in metrics], width, label=f'{name} (K={report["clusters"]})')
        with (Path(report['output_directory']) / 'assignments.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        labels = np.array([row.get('broad_category', row['true_label']) for row in rows])
        if len(rows) != len(stages):
            raise ValueError('Comparison assignments must align with the sampling stages')
        if reference_labels is None:
            reference_labels = labels
            _categorical_strip(axes[1], reference_labels,
                title=f'Reference segmentation: {reference}', palette='tab20')
        elif not np.array_equal(labels, reference_labels):
            raise ValueError('Comparison runs must use the same reference segmentation')
        axes[i+2].sharex(axes[1])
        _categorical_strip(axes[i+2], [int(row['cluster']) for row in rows],
                           title=f'{name}: inferred states', palette='tab20b')
    ticks = np.unique(np.linspace(0, len(stages)-1, min(18, len(stages)), dtype=int))
    boundaries = np.flatnonzero(reference_labels[1:] != reference_labels[:-1]) + .5
    for axis in axes[1:]:
        axis.set(xticks=ticks, xticklabels=np.asarray(stages)[ticks])
        plt.setp(axis.get_xticklabels(), rotation=50, ha='right', fontsize=8)
        for boundary in boundaries:
            axis.axvline(boundary, color='0.3', linestyle='--', linewidth=.8, alpha=.6)
    axes[0].set(xticks=x, xticklabels=['Purity', 'ARI', 'NMI'],
                ylabel='In-sample agreement', title=f'{reference}; labels unused in fitting')
    axes[0].legend()
    axes[-1].set_xlabel('Sampling index; tick labels give hours post-fertilisation (unequal intervals)')
    figure.suptitle(f'{title}: same observations and HMM priors; independent vs evolving factors')
    return figure
