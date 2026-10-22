# Public research API

The normal entry point is `HDPGPC`, configured with `HDPGPCConfig`. It is an input
and result facade over the native modular implementation. It does not contain another
inference algorithm, normalize emission scores into new probabilities, or run
extra filtering/smoothing passes to produce summaries.

```python
from pathlib import Path
from hdpgpc import HDPGPC, HDPGPCConfig

config = HDPGPCConfig.from_yaml(
    Path('examples/configurations/dynamic.yaml').read_text())
model = HDPGPC(config)
model.fit(X)                 # X: (observations, points) for one output
labels = model.labels_       # one current hard label per observation
```

For streaming, use the online example (it explicitly disables warp requests):

```python
config = HDPGPCConfig.from_yaml(
    Path('examples/configurations/online.yaml').read_text())
model = HDPGPC(config)
result = model.update(x)     # x: one waveform, shape (points,)
cluster = result.assigned_cluster

result = model.update(next_x, include_prediction=True, include_posterior=True)
mean = result.prediction.mean
latent_mean = result.posterior[0].filtered_mean
```

The [configuration guide](configuration.md) describes how to create a recipe
directly or migrate the actual parameters of an existing experiment. These
five-point examples supply their own small-data hyperparameters; they do not
replace any original constructor defaults.

## Public names and method contract

The root namespace exports exactly these new names through `__all__`:

| Name | Purpose |
| --- | --- |
| `HDPGPC` | Normal fit/update facade |
| `HDPGPCConfig` | Versioned construction recipe; alias of the config export |
| `UpdateResult` | One committed streaming update |
| `Prediction` | A component's current observation mean and covariance |
| `LatentState` | One output's latest stored filtered/smoothed state |

The model/result types are also exported from `hdpgpc.model`, which additionally
exports `WarpPosterior` for the opt-in GP-flow policy. Public methods and
properties on `HDPGPC` are:

| Name | Contract |
| --- | --- |
| `HDPGPC(config)` | Copy and validate the recipe; backend construction is deferred until fit/update |
| `fit(X, *, grid=None)` | Start a fresh model, run the selected schedule, return `self` |
| `update(x, *, grid=None, include_prediction=False, include_posterior=False)` | Commit one observation; return an independent `UpdateResult` |
| `labels_` | Read-only copy of the latest committed hard-label vector |
| `cluster_ids_` | Sorted unique indices represented in `labels_` |
| `n_clusters_` | Number of represented clusters, excluding empty reserved slots |
| `predictive(cluster, *, grid=None)` | Read a component's current observation distribution |
| `posterior(cluster)` | Copy its latest stored latent state for each output |
| `warp_posterior(x, *, grid=None, cluster=None)` | Fit candidate-conditioned warp posteriors from the current model state without committing `x`; requires `gp_flow` |
| `config` | An independent copy of the construction recipe |

There is deliberately no ambiguous `predict(X)` method: assigning new curves,
forecasting latent evolution, and querying a GP observation distribution are
different scientific operations. `predictive` performs the last of these.

The eight section dataclasses and their serialization/factory methods remain
public in `hdpgpc.config`. The previously documented component APIs remain
available for advanced use: emission, dynamics, alignment, HDP and inference
exports are listed in [architecture.md](architecture.md). The 0.3 engine interfaces replace the retired research entry points.

For record experiments, `hdpgpc.preprocessing.prepare_sequences` selects
channels, optional first-sequence scaling, and a common grid without assuming
an ECG file layout. `hdpgpc.diagnostics` exports label statistics, confusion
and LDS-prior formatting, log duplication, and fitted LDS/warp plots. See the
[complete-record workflow](record_workflow.md) for inputs and interpretation.

`hdpgpc.datasets.load_daphnet_record` reads an official ZIP or extracted
recording, and `window_daphnet_record` constructs chronological windows without
joining experimental gaps. `daphnet_records` lists IDs and `download_daphnet`
retrieves the official archive when requested. See the
[Daphnet waveform workflow](daphnet_workflow.md).

Fixed observation models are exported from `hdpgpc.emissions`:
`FixedObservationModel`, `AffineGaussianObservation`, `PCAObservationEstimate`,
`estimate_pca_observation`, and the Fourier/wavelet factories and basis functions.
Their implementations live in `emissions._fixed_observation`, `_pca`, `_fourier`
and `_wavelet`. The previous `hdpgpc.dynamics` observation exports and private
modules have been removed; callers must import these objects from `emissions`.
Serialized affine state dictionaries and configuration arrays are unchanged.
Python pickles referring to the removed module paths require migration; no
legacy module aliases are retained.
Latent filtering, smoothing and parameter estimation remain in `dynamics`.

`SFAObservationEstimate` and `estimate_sfa_observation` are exported from
`hdpgpc.emissions` (implementation `emissions._sfa`). Temporal calibration
returns the existing affine observation model plus raw slow-feature encoder,
pair indices and derivative eigenvalues. Install `estimate.model` with
`with_fixed_observation_model`; use `estimate.transform(values)` for raw SFA
features. See [Linear SFA](linear_sfa_observation.md) for sampling and the
distinction between raw encoding and Gaussian posterior inference.

`LinearPredictionObservationEstimate` and
`estimate_linear_prediction_observation` are exported from `hdpgpc.emissions`
(implementation `emissions._linear_prediction`). The estimate supplies the
ordinary affine `model`, raw `transform`, calibration `predictor` B, temporal
pairs and loss history. B is not installed as the LDS A. See
[linear prediction observations](linear_prediction_observation.md).

`hdpgpc.emissions.make_fourier_observation` creates an
`AffineGaussianObservation` from a full or selected real windowed Fourier basis.
`window_size` defaults to 16 non-overlapping points; setting it to the waveform
length reproduces the former global basis. Harmonic selections apply per window.
`real_fourier_basis` returns the synthesis matrix itself. Install the decoder
with `hdpgpc.config.with_fixed_observation_model`, the same helper used for
fixed linear autoencoders and new PCA configurations. See
[fixed windowed Fourier observation](fourier_observation.md).

`hdpgpc.emissions.make_wavelet_observation` creates the same affine observation
model using a full real orthonormal Haar basis. `real_wavelet_basis` returns
its synthesis matrix. Both require a uniformly spaced grid with a power-of-two
point count. See [fixed wavelet observation](wavelet_observation.md).

All names beginning with `_`, including `HDPGPC._backend`, are internal. The
implementation modules `model._api` and `model._results` are internal; import
their public classes through the exports above. Native tensors, mutable model
lists, optimizer diagnostics, proposal workspaces and caches are not new public
facade attributes. No generic forwarding `__getattr__` exposes the backend.

## Input shapes and coordinates

| Operation | Single output | Multiple outputs |
| --- | --- | --- |
| `fit(X)` | `(observations, points)` or `(observations, points, 1)` | `(observations, points, outputs)` |
| `update(x)` | `(points,)` or `(points, 1)` | `(points, outputs)` |

`X`/`x` denote waveform **values** in the new API. Time/domain coordinates use the
separate keyword `grid`. This differs intentionally from the legacy
`include_sample(x_train, y)` parameter naming.

For a shared grid, pass `(points,)` or `(points, 1)`. `fit` also accepts a separate
grid per observation with shape `(observations, points, 1)`. Without `grid`, the
facade uses `config.x_basis`. A list of identical per-cluster bases is also
unambiguous; different bases require explicit observation coordinates. When
inducing points use a smaller basis than the observations, provide the full
observation grid explicitly. No uniform spacing is guessed.

Array-like real inputs are copied to finite CPU float64 arrays and normalized to
the shapes expected by the legacy schedules. Tensor inputs are detached before
conversion. This is an observation API, not an autograd interface. Empty,
nonfinite, wrong-output-count and mismatched-grid inputs fail before inference.
Advanced native engines retain the original tensor/dtype/device behavior.
CUDA models use their existing conversion path after this public input adaptation;
the new facade has been characterized on CPU only.

## Schedule and lifecycle semantics

- `schedule='offline'`: `fit` calls `OfflineInference.fit` once with the configured
  `warp` and `it_limit`; `update` rejects the offline configuration.
- `schedule='online'` or `'online_fast'`: `fit` feeds observations sequentially
  through that existing schedule; subsequent `update` continues it.
- `schedule=None`: `fit` selects offline, while `update` on a fresh facade selects
  standard online. These defaults choose existing entry points; they do not
  change the constructor defaults or unify algorithms.
- `fit` always replaces the fitted backend with a fresh configured model. Existing
  results remain independent snapshots. Reproducibility settings apply at fresh
  construction as documented in the configuration guide, not at each update.
- `update` cannot continue an offline fit. The two schedules use different state
  caches, so the facade does not silently switch between them.
- Normal fitting/updating requires `classify=False`. Advanced classification,
  label reload and classification-learning use the advanced native engines described below.

`labels_` uses the most recent recorded label vector, never the possibly stale
`resp_last` cache. The one-observation offline exit does not append that vector;
in that case the facade reports the unique GP component that actually included
sample zero. It performs no additional classification or message pass. If a
future/untested path does not retain an unambiguous label for every observation,
label access raises rather than fabricating assignments.

On a numerical/inference exception, the original exception is propagated. The
backend may already be partially mutated; the facade marks that run failed and
refuses further updates or fitted summaries. A new `fit` starts fresh. Ordinary
input-validation errors do not invalidate an already successful model. Optional
prediction errors happen **after** a successful update has committed: a retry of
`update` would add another observation. Query with `predictive` separately when
that distinction matters.

## Streaming result: assignments and uncertainty

`UpdateResult` fields are frozen; its array fields are independent, read-only
NumPy snapshots:

| Field | Meaning |
| --- | --- |
| `observation_index` | Zero-based global observation index just committed |
| `assigned_cluster` | Latest committed cluster index, consistent with `labels_[-1]` |
| `cluster_ids` | Allocated indices for this result's responsibility columns, including any empty slots |
| `responsibilities` | One-hot vector for the retained hard assignment |
| `assignment_probabilities` | `None`: calibrated soft probabilities are unavailable from the retained state |
| `new_cluster` | Whether the update increased the backend's allocated cluster count |
| `prediction` | Optional `Prediction` for the assigned cluster, after committing the observation |
| `posterior` | Optional tuple of `LatentState`, one per output, after the update |

The existing `_safe_exp` performs argmax hardening. Neither softmaxing raw emission
scores nor describing the hard vector as calibrated uncertainty would faithfully
report that implementation. Hence hard responsibilities and unavailable soft
probabilities are explicit, separate concepts in this result.

The constructor creates the first cluster before any update, so the first
observation normally has `new_cluster=False`. Cluster indices are **current model
indices**, not permanent entity IDs: later reallocation/reordering may renumber
them or revisit historical labels. Old result objects remain historical snapshots.
For example, the small static streaming fixture records assignments
`[0, 0, 1, 1]` and birth flags `[False, False, True, False]`.

## Predictive and latent quantities

`Prediction.mean` has shape `(points, outputs)` and `covariance` has shape
`(outputs, points, points)`. `grid` has shape `(points, 1)`. Every output calls the
`GaussianProcessEmission.observe_last` on the same query grid. The default
query grid is the fixed decoder's waveform grid when configured, otherwise
the selected cluster's first-output GP basis. There is no
cross-output covariance matrix or mixture average.

This is a **current component observation distribution**, not a one-step future
LDS forecast or pre-update predictive assessment. At the native basis, the
legacy GP implementation returns `C[-1] @ f_star_sm[-1]` and `Sigma[-1]`.
The fixed affine decoder returns `offset + C @ f_star_sm[-1]` and
`C @ cov_f_sm[-1] @ C.T + Sigma`. At another
grid, the existing GP interpolation and stabilization formulas are used. With
alignment, coordinates are the model/aligned domain; no inverse warp back to raw
observation coordinates is performed.

`posterior(cluster)` returns one `LatentState` per output, allowing different
native bases. Each snapshot includes output index, last included observation
index, basis grid, filtered mean/covariance and stored smoothed mean/covariance.
Means are one-dimensional; covariance matrices remain two-dimensional. These are
the existing stored values, with no fresh smoothing or covariance correction.
Output ordering follows the backend; configured output reduction retains its
existing selected-output ordering and does not expose a new original-channel map.

Both kinds of summaries detach/copy arrays for reporting. They do not detach,
replace or mutate training tensors. Predictions/posteriors are opt-in on `update`
so normal streaming does not pay for their computation and matrix copies.

`InferenceConfig.offline_strategy` selects `'legacy'` (default) or
`'split_merge'` for offline fits. The latter returns detached proposal diagnostics
through `model.split_merge_history_`; legacy fits return an empty list.
`model.split_merge_convergence_` reports the stopping reason, iteration and
final assignment/score stability checks (an empty dictionary for legacy fits).
Split/merge outer stopping uses `split_merge_stability_patience=3` consecutive
stable iterations and `split_merge_score_rtol=1e-3` by default. See
[offline split–merge inference](offline_split_merge.md) for its controls and
relationship to LDS updates and the configured objective.

## Advanced native engines

For tensor-level classification, label-conditioned fitting and other scientific
workflows, see [the migration guide](migration.md). The engines are native:
`OnlineInference.update/update_fast`, `OfflineInference.fit/classify_batch/fit_labels`.
No old model imports, flat constructor adapters or old pickle paths are supported.
The mutable state returned by `config.build_model()` is opaque and private.

For actual 0.3 verification results see [the migration report](migration_0_3/report.md).

## Dataset experiments and reusable reporting

`hdpgpc.emissions.estimate_autoencoder_observation` calibrates the tied linear
autoencoder on explicitly supplied observations. It returns an
`AutoencoderObservationEstimate` containing the frozen network, exact affine
observation model, selected indices, losses and reconstruction residuals.
Use `with_fixed_observation_model(config, estimate.model, ...)` to install it.
The existing PCA/Fourier/wavelet factories share this observation structure.

`hdpgpc.datasets.ucr` supplies `PreparedDataset`, `archive_datasets`,
`prepare_dataset` and `download_archive`. Preparation returns TRAIN/TEST values
and labels plus TRAIN normalization statistics and interpolation metadata.
Importing the dataset namespace does not load numerical libraries.

`hdpgpc.diagnostics.plot_clustering_summary(model, values, annotations, ...)`
returns a figure per output, using the same overview/gallery for all datasets.
Pass `ordered=False` for class composition instead of temporal transitions.
LDS/warp diagnostics remain separate reusable functions.

`hdpgpc.experiments.ExperimentRun` owns one content-derived run directory and
log; `save_results` emits the common results/assignment/plot schema. The record
and UCR workflow modules expose `build_parser`, `make_config`, and their
respective `run_records`/`run_ucr` functions. They orchestrate the public model
API; they are not model implementations. See [the workflow guide](experiment_workflows.md).
