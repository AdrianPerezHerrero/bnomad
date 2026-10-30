# Comparable experiment scripts

Start with one dataset or a short record prefix. Every runner uses the ordinary
`HDPGPC(config).fit(values, grid=grid)` API and the same reporting code.

Dataset examples follow `<dataset>_<representation>_clustering.py`, using
`ucr`, `mitbih`, `daphnet` or `cas_colon` as the dataset prefix. `waveform` identifies the
direct GP/LDS representation; alignment and online/offline inference remain
selectable arguments within that example.

```powershell
python examples/ucr_pca_clustering.py --datasets Coffee --components 12
python examples/ucr_autoencoder_clustering.py --datasets Coffee --components 12
python examples/ucr_waveform_clustering.py --datasets Coffee --no-warp
python examples/daphnet_waveform_clustering.py --records S02R01 --warp
python examples/mitbih_waveform_clustering.py --records 100 --no-warp
python examples/mitbih_pca_clustering.py --records 100 --components 12
python examples/cas_colon_pca_clustering.py --records 1 --components 10
python examples/cas_colon_autoencoder_clustering.py --records 1 --components 10 --epochs 250
python examples/cas_colon_sfa_clustering.py --records 1 --components 10
python examples/cas_colon_linear_prediction_clustering.py --records 1 --components 20 --prediction-weight 0.1
python examples/mitbih_autoencoder_clustering.py --records 100 --calibration-samples 400
python examples/ucr_fourier_clustering.py --datasets ACSF1 --window-size 16
python examples/ucr_wavelet_clustering.py --datasets ACSF1
```

Use `--help` for each script's arguments. The scripts have the same layout:
dataset/representation selection, a `DEFAULTS` dictionary, parser construction,
then execution. The dictionary sets CLI defaults; it is not another model
configuration system. Shared builders construct the existing typed
`HDPGPCConfig`. Change `hdpgpc.experiments.records.make_config` or
`hdpgpc.experiments.ucr.make_config` for a recipe beyond the CLI controls.

## The stages

1. **Prepare:** dataset loaders in `hdpgpc.datasets` and transforms in
   `hdpgpc.preprocessing` produce `(observations, points, outputs)` and a grid.
2. **Calibrate:** PCA and tied linear autoencoder helpers in `hdpgpc.emissions`
   supply a fixed affine decoder. Fourier/wavelet factories supply fixed bases.
3. **Configure:** the appropriate builder returns an ordinary typed recipe.
4. **Fit:** call the public model API with the same input structure.
5. **Report:** `hdpgpc.experiments.save_results` records metrics, assignments,
   matrices and selected diagnostic figures, without refitting the model.

`hdpgpc.experiments` owns orchestration and files. It contains no alternative
inference engine. General plotting belongs to `hdpgpc.diagnostics`.

CAS-Colon PCA, affine autoencoder, Linear SFA and linear-prediction runners use an image gallery and retained latent-state trajectory plots
instead of a flattened-pixel waveform gallery. It retains chronological frames
and HMM transitions. See the [image experiment guide](cas_colon_pca.md) for data
layout, frame-gap handling, reconstruction/forecast panels and current restrictions.
The [autoencoder guide](cas_colon_autoencoder.md) describes its calibration,
covariance scales and exact affine limitations.
The [Linear SFA guide](linear_sfa_observation.md) explains calibration using
genuine temporal pairs and its fixed Gaussian decoder.
The [linear prediction guide](linear_prediction_observation.md) describes a
normalized affine objective combining reconstruction and latent prediction.

## Common arguments

| Purpose | Arguments |
| --- | --- |
| Inference | `--schedule offline/online/online_fast`, `--iterations`, `--hard-assignments` / `--no-hard-assignments` |
| Offline optimizer | `--offline-strategy legacy/split_merge`, `--split-merge-refine-steps`, `--offline-representer-selection sorted/fixed` |
| Split/merge stopping | `--split-merge-stability-patience` (3), `--split-merge-score-rtol` (0.001) |
| Exploration | `--hdp-preset`, `--max-clusters`, `--exploration-steps` |
| LDS | `--lds-update`, `--lds-scoring`, `--lds-degrees`, `--lds-step-size`, `--birth-lds-kl-per-dimension` |
| GP/alignment | `--fit-gp` / `--no-fit-gp`, `--alignment`, `--warp` / `--no-warp` |
| Resolution/subset | `--max-points`, `--max-series` |
| Calibration | `--components`, `--calibration-samples`; AE also `--epochs`, `--learning-rate` |
| Normalization | `--normalize-samples` / `--no-normalize-samples` |
| UCR covariance priors | `--sigma-prior`, `--gamma-prior`: variances for waveform mode; covariance multipliers for calibrated decoders |
| Record covariance priors | `--sigma-prior`: waveform variance or calibrated PCA/AE Sigma multiplier; `--gamma-prior`: process variance times identity |
| Diagnostics | `--plots`, `--projection-plots`, `--lds-plots`, `--lds-dynamics-plots`, `--warp-plots`, `--birth-plots N` |
| Cluster detail selection | `--min-cluster-samples P` (default 1); show detail panels only for clusters with at least P final assigned observations |
| Reproducibility | `--seed`, `--output-dir`, `--resume`, `--verbose silent/progress/debug` |

Boolean switches support their `--no-...` counterpart. `--no-plots` disables
all PNGs; saving closes figures without displaying them. Warp plots are made
only when alignment is active. Birth plots capture the first N proposals and
are currently available only offline. A requested diagnostic failure marks
the run as failed; missing captured birth curves are reported explicitly.

`--min-cluster-samples 5` omits clusters with fewer than five final MAP-assigned
observations from image/waveform galleries, fitted-mean/latent-path displays,
CAS-Colon trajectory detail rows and optional LDS matrix/dynamics plots. It is
a reporting threshold; it does not delete clusters or refit the model. Original
cluster IDs and colors are retained. Assignment timelines, transitions, label
composition, projection/warp overviews, scores and saved arrays include the
complete clustering. Figure headers identify the selection; `results.json`
records the threshold and shown/hidden detail cluster IDs under `plot_selection`.
If all clusters are below the threshold, the overview remains and detail-only
figures explain that no cluster meets the minimum. Counts use final assignments,
including ambiguous-label observations, rather than soft responsibility sums.

By default, offline uses `conjugate_batch`, and online schedules use
`conjugate_sequential`. An explicit batch request online is rejected. Fixed
affine observations require CPU, one output, final scoring, no warp, no GP
kernel fitting and a non-legacy LDS update. They use the exact existing affine
Gaussian likelihood; encoder coordinates are not treated as noiseless states.

The default offline optimizer remains `legacy`. Select `--offline-strategy
split_merge` for group split/merge search with the same GP/LDS and HDP tools.
`--split-merge-refine-steps` controls local refinement and `--exploration-steps`
limits candidate fits per outer iteration. See [offline split–merge inference](offline_split_merge.md)
for scoring, caching and stopping semantics.

The scripts retain aliases for earlier CLI names, including `--latent-dim`,
`--ae-beats`, `--ae-epochs`, `--pca-samples`, `--points`, `--max-windows`,
`--save-plots`, `--plot-lds`, `--plot-warps`, `--plot-dir`,
`--lds-degrees-of-freedom` and `--fixed-gp-hyperparameters`, where applicable.
The singular `--dataset` remains an alias for one UCR dataset.

## Comparisons that need different assumptions

| Runner | Dependence and representation | Default learned covariance |
| --- | --- | --- |
| ECG waveform | Chronological; `stable` LDS preset; DTW-GP enabled | Gamma and Sigma per cluster, with preset shrinkage |
| Daphnet waveform | Chronological; `only_plastic`; warp off | Gamma per cluster; Sigma fixed |
| ECG PCA | Chronological; fixed calibrated PCA C | Gamma per cluster; Sigma fixed (selectable) |
| ECG autoencoder | Chronological; fixed trained affine C | Gamma per cluster; Sigma fixed |
| CAS-Colon PCA / autoencoder / SFA / linear prediction | Chronological images; HMM on; fixed calibrated affine C | A/Gamma per cluster; Sigma fixed (selectable) |
| Zebrafish all-gene MFA | Chronological stage means; HMM on; same all-gene PCA and MFA priors in both arms | Shared C/diagonal Sigma; fixed A=0/Gamma=I versus learned A/Gamma |
| Xenopus all-gene MFA | Chronological measured polyA+ gene TPMs; each clutch fitted separately; shared expression runner | Same independent/dynamic factor comparison as zebrafish |
| Zebrafish / Xenopus PCA and autoencoder | Same expression preparation and HMM; fixed calibrated affine decoder | Gamma learned per cluster; A fixed at zero or learned; Sigma fixed by default |
| UCR waveform | HMM off; A=0, C=I; DTW-GP enabled | Sigma full per cluster; Gamma fixed |
| UCR PCA | HMM off; A=0; fixed PCA C | Gamma full and Sigma diagonal per cluster |
| UCR autoencoder | HMM off; A=0; fixed trained affine C | Gamma full per cluster; Sigma fixed |
| UCR windowed Fourier | HMM off; A=0; fixed windowed synthesis C | Gamma and Sigma full per cluster |
| UCR Haar wavelet | HMM off; A=0; fixed orthonormal C | Gamma full and Sigma diagonal per cluster |

Record runners retain chronological order. `--max-series` takes a prefix.
Daphnet excludes unlabelled gaps and fits each selected contiguous segment
with a fresh model; it never joins disconnected intervals. Records support
`--lds-preset`, optional `--a-mode`, `--c-mode`, `--gamma-mode`, `--sigma-mode`
overrides, `--shrink-strength`, `--lds-inducing-points`, and optional offline
`--calibrate-lds-noise-priors`. Native configuration validation rejects
incompatible policy combinations. A static mixture preset also disables HMM.
With a calibrated record decoder, C is fixed and A/Gamma are learned per
cluster; only Sigma's noise policy is selectable.

For record PCA, use:

```powershell
python examples/mitbih_pca_clustering.py --records 100 --components 12
python examples/mitbih_pca_clustering.py --records 100 --offline-strategy split_merge --sigma-mode cluster_diagonal
```

The dedicated PCA entry point defaults to 12 components, a 400-beat calibration
minibatch, 50 outer iterations, no warp and no GP kernel fitting. Calibration
reduces the component count if the waveform or minibatch is smaller. PCA's mean
and decoder remain fixed, A/Gamma learn per cluster, and Sigma defaults to the
calibrated fixed residual covariance. `--sigma-mode` selects a learned or shared
covariance when supported by the model. `--sigma-prior` multiplies the calibrated
Sigma before installing its fixed value or learned prior center. Dedicated PCA
and autoencoder record runners default to multiplier 1, preserving calibration;
0.1 reduces the covariance tenfold and 10 increases it tenfold. These are
covariance multipliers, not standard deviations or prior-strength controls.
For records, `--gamma-prior` continues to set Gamma's prior center to that value
times identity; it does not scale the calibrated initial state covariance.
The applied Sigma is recorded in `config.yaml`, `observation_calibration.npz`
and the report metadata; the initial log identifies its multiplier.
Outputs are saved under
`agents/mitbih_pca_runs/<record>/<run_id>/`. The waveform entry point's
`--representation pca` option remains available with explicit `--no-warp
--no-fit-gp`. With that waveform entry point, also pass `--sigma-prior 1` to
preserve calibration: its default remains the waveform variance default.

Online record calibration defaults to a prefix and rejects uniform sampling;
the prefix is replayed during fitting. Offline calibration can sample uniformly.
`--scaling first_window` applies the first observation's scale to all record
observations. `--normalize-samples` instead scales each observation separately,
removing amplitude differences. `--unit-grid` changes record coordinates to 0..1.
GP white-noise and lengthscale bounds are available in record runners through
`--gp-noise-bounds LOWER UPPER` and `--lengthscale-bounds LOWER UPPER`.
`--gp-outputscale`, its bounds, and `--sigma-bounds`/`--gamma-bounds` control
the remaining initial GP scale and LDS variance ranges.

UCR always calibrates normalization and learned decoders on TRAIN only, then
clusters TRAIN+TEST together in a seeded permutation. `--max-series` selects
a seeded random subset. Results, including TEST-only ARI, are transductive
clustering descriptions, not held-out classification accuracy. A=0 and HMM off
remove temporal conditioning; initialization and proposal schedules can still
depend on row order. `--sigma-mode` and `--gamma-mode` independently select
supported covariance policies. Noise scales are variances, not SDs.

UCR archive runners default to all datasets. Fourier and wavelet examples
default to ACSF1; `--datasets all` processes the whole archive. Record runners
accept `--records all`. Loading is sequential by dataset/record. Use
`--download` explicitly when a supported archive is absent.

Each script's `DEFAULTS` retains its representation-specific choices. Shared
fallbacks are 30 iterations, a 30-cluster cap, 10 exploration steps and progress
verbosity. Previous edited global constants are now CLI/default-dictionary
settings. The common layout and output schema replace the older flat filenames;
they do not claim identical numerical results for differing recipes.
PCA defaults to soft assignments; the other entry scripts retain hard
assignments. Set `--hard-assignments` or `--no-hard-assignments` explicitly for
comparisons. The waveform UCR script retains five exploration steps by default.

## Run folders and results

```text
output_directory/
  results.csv
  dataset_or_record/
    run_id/
      settings.json
      config.yaml
      clustering.log
      results.json
      assignments.csv
      lds_parameters.npz
      clusters_output_0.png
      projection.png
      lds_parameters_output_0_01.png   # when requested
      lds_dynamics_output_0.png        # when requested
      warps_output_0.png               # when enabled/requested
      birth_warp_01.png                # when captured/requested
```

Calibration and normalization NPZs are saved when applicable. The AE also
records training losses/indices and a calibration reconstruction plot.
Cluster summaries share the same function and fixed panel dimensions as the
ECG report. Ordered summaries show transitions; unordered summaries show class
composition. LDS and warp figures retain their existing detailed diagnostics.
The projection is a two-dimensional visualization of codes or waveform rows;
it does not change the fitted representation.
Its cluster legend is placed below the three data panels in up to six columns.
Additional legend rows increase the figure height, preserving space for the
projections and class-composition plot when many clusters are present.

`results.json` has a common schema: status, run ID, dimensions, metrics,
class-majority confusion matrix, cluster mapping, metadata, artifact list and
elapsed time. `assignments.csv` starts with fit row, true label, cluster and
cluster-majority label, followed by source row/split or timestamps. Labels are
used after fitting for reporting and plot colours. The majority mapping is
both fitted and scored in-sample.

The run ID hashes settings, source contents and package Python code. Changing
any of these produces another directory. `--resume` skips a successful run only
when every recorded artifact still exists. It does not resume interrupted
inference. Repeating the identical run without `--resume` reruns that directory.
The root table is updated by dataset/run ID without duplicate rows. An older
incompatible `results.csv` is preserved; the new table is then named
`results_standardized.csv`. Failures record tracebacks and error status;
remaining prepared datasets continue and the entry script exits nonzero if
any fit fails. Input-loading errors are raised directly.

## Known limitation found during this cleanup

The standard `online` waveform path can fail during reordering with hard
assignments or retain labels for removed components with soft assignments at
a small cluster cap. Both cases reproduce on the pre-cleanup implementation.
The runner detects the inconsistent final state and records a failed run before
writing assignments or metrics, rather than drawing a report for missing models.
The cleanup does not change cluster birth/pruning mathematics. The tested
offline, affine-online and Daphnet `online_fast` cases completed successfully;
they do not establish that every archive/settings combination will succeed.

## Automatic CAS-Colon equal-duration baseline

Every CAS-Colon representation (PCA, autoencoder, SFA, linear prediction and
MFA, including shared MFA) now computes a deterministic temporal baseline after
fitting. No extra flag is needed. For the fitted model's K occupied clusters,
the baseline divides the full fitted segment into K contiguous equal-duration
bins. CAS-Colon frames have a fixed cadence; when the frame count is not
divisible by K, bin lengths differ by at most one frame. It uses K from the
final assignments, not the maximum cluster cap or a plot's minimum-size filter.

Each bin is assigned its majority anatomical label using the same evaluation
mask as the model. Ambiguous/excluded frames still occupy their original time
positions but do not contribute to majority fitting or scores. A bin with no
evaluated frames has no majority label. Majority ties use the same deterministic
class order as the model. Purity measures the majority-label agreement; ARI and
NMI compare the unmerged temporal-bin IDs with reference labels. Two bins with
the same majority label remain distinct for ARI/NMI.

Model and baseline purity/ARI/NMI appear together in the saved plot headers,
including the image gallery, projections, latent trajectories, calibration
plots and every LDS diagnostic page. `equal_duration_baseline.png` also shows
the reference regions, baseline bin IDs and baseline-majority timeline using
the established anatomical/independent-cluster palettes. Fixed-size image
galleries gain header space without shrinking their thumbnails.

Each run saves:

- `equal_duration_baseline.json`: assignments, majority labels, metrics,
  per-bin boundaries/counts and confusion matrix.
- `equal_duration_assignments.csv`: per-frame bins, majority labels, reference
  labels, evaluation flags and timestamps.
- `results.json` → `baselines.equal_duration`: baseline definition and scores.
- `assignments.csv`: added `baseline_cluster` and `baseline_majority_label`
  columns alongside the fitted assignments.

Baseline metrics and assignments are saved and printed even with `--no-plots`;
PNG generation follows the existing plotting controls. The root results table
continues to summarize model scores. The baseline uses no pixels, inference,
random sampling or label-driven boundaries. Majority labels are assigned and
scored in sample, just as the model's majority-label mapping is. It provides a
matched-K temporal comparison, not a separately trained anatomical classifier.
Existing completed run folders are not modified; subsequent runs include the
baseline under their updated code-derived run ID.
