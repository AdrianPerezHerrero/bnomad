# Working with a complete time-series record

`examples/full_record_clustering.py` is an ECG experiment: it selects MIT-BIH
records, reads their waveform and annotation files, builds a chosen recipe,
fits the model, logs scores, and saves plots. Its experiment controls remain
near the top of that script. The reusable preparation and reporting helpers
now live in the package, so another dataset can use them independently.

```python
from hdpgpc import HDPGPC
from hdpgpc.preprocessing import prepare_sequences
from hdpgpc.diagnostics import cluster_label_statistics, format_confusion_matrix

# signal: (sequences, points, channels), labels: one known label per sequence
values, grid = prepare_sequences(signal, channels=(0,), unit_grid=True)
model = HDPGPC(config).fit(values, grid=grid)
scores, classes, counts, cluster_labels = cluster_label_statistics(
    model.labels_, labels
)
print(scores)
print(format_confusion_matrix(classes, counts))
```

Construct `config` as in the [typed configuration guide](configuration.md).
`prepare_sequences` can
also set `scale_from_first=True` to center and standardize each selected
channel using the **first sequence's** statistics, applying that same transform
to all sequences. It returns a common grid from 0 to `points-1`, or from 0
to 1 with `unit_grid=True`. The waveform must have finite values and a common
point count. This helper does not read files or assume an ECG format.

`cluster_label_statistics` maps each fitted cluster to its majority reference
label, then returns purity, **in-sample** majority-label accuracy, adjusted
Rand index, normalized mutual information, class labels, and a confusion
matrix whose rows are true labels and columns are the cluster-majority
assigned labels. The majority mapping is fitted and scored on the same
labels; it is descriptive clustering evaluation, not held-out classification.
The ECG script reads annotation labels from its record files and, when there
is one extra annotation, crops the annotation vector to the number of
waveforms before using this helper.

`hdpgpc.diagnostics.format_lds_priors` formats selected LDS prior centers and
strengths for a run log; `Tee` duplicates output into a log stream. Plotting
helpers such as `plot_lds_diagnostics`, `plot_warp_diagnostics`, and
`model_lds_arrays` are also available from `hdpgpc.diagnostics`. The
record-specific composite report remains a plotting example. It is useful
to save the model configuration and label mapping with each figure, because
several reported scores depend on the evaluation choices.
