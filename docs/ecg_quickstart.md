# Small reproducible ECG example

From the repository with `hdpgpc` installed:

```sh
python examples/ecg_quickstart.py --output-dir agents/ecg_quickstart/my_run
```

The fit uses the public `HDPGPCConfig`, `HDPGPC.fit()` and `predictive()` APIs.
It needs only the core installation. A fresh run takes roughly eight seconds to
fit on the development CPU; downloads and plotting add time. Use a new output
directory for every run so earlier experiments are preserved.

![MIT-BIH 208 ECG clustering](assets/ecg_quickstart.svg)

## Data and preprocessing

The inputs are the **first 64 consecutive archived beats of record 208**, in
chronological order. The example does not select or balance beats by annotation.
The archive contains 35 `N`, 22 `V` and 7 `F` annotations in this window.

The source is the [MIT-BIH Arrhythmia Database, version 1.0.0](https://physionet.org/content/mitdb/1.0.0/),
by George Moody and Roger Mark, available through PhysioNet under the
[Open Data Commons Attribution License v1.0](https://physionet.org/content/mitdb/view-license/1.0.0/).
Please cite Moody and Mark (2005), DOI
[10.13026/C2F305](https://doi.org/10.13026/C2F305), and the dataset's requested references
when using these data in research.

This example uses the project's previously filtered and beat-centered `.npy`
arrays from repository revision `615109a9e474d37626abaff6c51f1c159b26124a`.
It records the source URLs and validates both files by SHA-256. The source
archive is about 4.3 MB and is downloaded only when local files are missing.
These processed arrays are distinct from the raw PhysioNet WFDB files.
The [annotation-alignment guide](mitbih_annotations.md) describes their extraction
and the loader's handling of omitted trailing windows.

Each archived window contains 90 samples around an annotation, from sample
`-27` through `62` at 360 Hz. The quickstart retains channel 0 and every second
sample, giving `X.shape == (64, 45, 1)`. It divides all selected beats by **one
shared maximum absolute amplitude**. The archive has already removed each beat's
channel mean; no additional per-beat scaling is applied. Plot amplitudes are
normalized archived ADC values, rather than calibrated millivolts. Plot time
is relative to the beat annotation; the sequence axis is a beat index because
the archive does not retain annotation timestamps.

To require an existing local copy:

```sh
python examples/ecg_quickstart.py \
  --data-dir hdpgpc/data/mitbih --no-download \
  --output-dir agents/ecg_quickstart/local_run
```

To test the pinned download independently of repository data:

```sh
python examples/ecg_quickstart.py \
  --data-dir agents/ecg_quickstart/data \
  --output-dir agents/ecg_quickstart/downloaded_run
```

## Model and observed result

The example fits static Bayesian GP emissions with an ergodic HMM, hard
assignments and the legacy offline search, capped at six iterations and eight
possible clusters. GP lengthscale starts at 0.25 on a within-beat coordinate
rescaled to `[0, 4]`, with bounds `[0.08, 0.8]`. Observation noise starts from
`0.01 I` and is **learned as a separate diagonal covariance in each cluster**.
`Sigma_policy=NoisePolicyConfig(mode='cluster_diagonal')` selects independent
conjugate noise posteriors, while `C_policy=MatrixPolicyConfig(mode='fixed')`
keeps the original static observation map at identity. LDS prior degrees are
`free_deg_MNIV=1` (reduced from 5), giving the initial Sigma less weight so the
observations can change it more strongly. The example uses
`lds_update='conjugate_batch'` so static emissions actually update Sigma after
each complete cluster pass; the static sequential path does not do that.
The GP kernel's white-noise parameter has bounds `[0.001, 0.05]`; these are not
clipping bounds on the learned Sigma posterior. HDP concentrations are explicitly set to
`gamma=1`, `trans_alpha=1`, `start_alpha=0.1`, and `kappa=0`. The full recipe is
saved in `config.json`. This is a morphology-clustering example; the static
emission model does not establish cardiac state dynamics.

The recorded run found five clusters. Purity was 0.96875, adjusted Rand index
0.87409, and normalized mutual information 0.83173. Cluster sizes were
37, 20, 4, 2 and 1. The 37-beat group contained all 35 `N` beats and two `F`
beats; `V` split into groups of 20 and two, and the remaining `F` beats split
into groups of four and one. The plot retains every group and reports its
annotation composition.

The learned diagonal variances vary both within waveforms and across clusters:

| Cluster | Minimum diagonal variance | Maximum diagonal variance |
| --- | ---: | ---: |
| 0 | 0.000929 | 0.023389 |
| 1 | 0.001670 | 0.013019 |
| 2 | 0.003295 | 0.011053 |
| 3 | 0.005000 | 0.019975 |
| 4 | 0.006458 | 0.010445 |

These values start from the same 0.01 prior variance. Larger clusters can
learn narrower noise bands, while smaller groups retain more of that prior.

Annotations never enter the model, preprocessing scale, priors or cluster count.
The displayed scores are evaluated on the fitted window. They describe this
small example and are not held-out classification performance. Shading is
`mean ± 2 sqrt(diag(Sigma_k))` using that cluster’s learned observation noise,
not a future prediction interval or a smoothed latent-state confidence interval.

## Report style

The figure uses the shared MIT-BIH `plot_clustering_summary` layout: assignments,
overlaid fitted means and row-normalized consecutive-beat transition counts,
followed by a three-column waveform gallery. Every cluster remains visible,
including the singleton. Gallery samples retain their own annotation hue even
when assigned to another majority class. Means and observation bands use the
usual ECG colors (`N`: blue, `V`: red, `F`: purple).

Assignment markers use the exact RGBA color of their cluster’s observation
uncertainty band, with opacity 0.22. Their hue describes the majority annotation;
cluster identity is given by the vertical cluster ID. The overview means and
cluster legend use the corresponding full-strength hues. Outer padding, panel
spacing, axis limits and title gaps keep the figure readable at README size.
All waveform panels share amplitude limits and label within-beat time in ms.

## Saved results and repeatability

The run directory contains:

- `ecg_clustering.svg` and `.png`: pale assignments, fitted means, empirical
  transition probabilities, and every cluster’s annotated waveform gallery.
- `input.npz`: the exact normalized inputs, grid and aligned annotations.
- `config.json`: the complete serializable model recipe.
- `result.json`: assignments, fitted means/covariances, learned `sigma_diagonal`
  for each cluster, noise policy/update mode, metrics, input/file hashes,
  data origin, source hash, environment and SVG hash.
- `fit.log`: the model's iteration output.

The recipe fixes CPU float64, seed 42 and one PyTorch/BLAS thread. Two fresh
processes on the same recorded environment produced identical scientific results
and identical SVG geometry and styles. SVG element identifiers may differ. Small floating-point differences can occur between
platforms or dependency builds; record `result.json` with any reproduced figure.

For longer records, other channels, alignment or learned dynamics, use
[`examples/mitbih_waveform_clustering.py`](../examples/mitbih_waveform_clustering.py)
and the [experiment workflow guide](experiment_workflows.md).
