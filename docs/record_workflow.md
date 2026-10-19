# Working with a complete time-series record

`examples/mitbih_waveform_clustering.py` is an ECG experiment: it selects MIT-BIH
records, reads their waveform and annotation files, builds a chosen recipe,
fits the model, logs scores, and saves plots. Its controls use the same CLI
and `DEFAULTS` layout as Daphnet and the UCR examples. The reusable preparation,
configuration and reporting stages live in the package. See the
[common workflow and run-folder guide](experiment_workflows.md).

For a fixed PCA observation model, use
`python examples/mitbih_pca_clustering.py --records 100 --components 12`.
This shares the waveform runner's data loading, annotation alignment and reports.
It calibrates PCA once, then learns cluster dynamics in the reduced latent space.
Select `--sigma-mode cluster_diagonal` or `--sigma-mode cluster` to learn the
observation covariance, or leave it at the calibrated fixed covariance.
`--sigma-prior` multiplies the calibrated PCA/autoencoder covariance (default 1
in their dedicated runners). It affects the fixed Sigma or the learned Sigma
prior center. `--gamma-prior` sets Gamma's prior center to that scalar times
identity. The calibrated initial state covariance is separate.

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
The ECG script uses the [MIT-BIH loader](mitbih_annotations.md) to align the
archived annotation labels with waveform windows before fitting. The loader
removes one extra trailing annotation when the legacy extractor omitted its
waveform window.

`hdpgpc.diagnostics.format_lds_priors` formats selected LDS prior centers and
strengths for a run log; `Tee` duplicates output into a log stream. Plotting
helpers such as `plot_lds_diagnostics`, `plot_warp_diagnostics`,
`model_lds_dynamics`, and `plot_lds_dynamics` are also available from
`hdpgpc.diagnostics`. The
annotated composite report is `hdpgpc.diagnostics.plot_clustering_summary`.
It is useful
to save the model configuration and label mapping with each figure, because
several reported scores depend on the evaluation choices.

`model_lds_dynamics(model, grid=waveform_grid)` projects each cluster's
latest smoothed latent state and one-step LDS transition to the requested
observation grid. `plot_lds_dynamics(...)` returns one vertically stacked
figure with a six-panel section for each cluster: current and forecast
waveforms, expected change, separate uncertainty
components, the observation-space transition, process covariance, and leading
process-change modes. Each section keeps the same panel dimensions as the
former individual cluster plot. Pass the full waveform grid for an inducing-point LDS;
fixed PCA and autoencoder decoders use their calibrated waveform grid. See
the [LDS dynamics interpretation](lds_inference.md#observation-space-lds-dynamics)
for the formulas and limits.
