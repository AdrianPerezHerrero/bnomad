# Comparable experiment scripts

Start with one dataset or a short record prefix. Every runner uses the ordinary
`HDPGPC(config).fit(values, grid=grid)` API and the same reporting code.

Dataset examples follow `<dataset>_<representation>_clustering.py`, using
`ucr`, `mitbih` or `daphnet` as the dataset prefix. `waveform` identifies the
direct GP/LDS representation; alignment and online/offline inference remain
selectable arguments within that example.

```powershell
python examples/ucr_pca_clustering.py --datasets Coffee --components 12
python examples/ucr_autoencoder_clustering.py --datasets Coffee --components 12
python examples/ucr_waveform_clustering.py --datasets Coffee --no-warp
python examples/daphnet_waveform_clustering.py --records S02R01 --warp
python examples/mitbih_waveform_clustering.py --records 100 --no-warp
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

## Common arguments

| Purpose | Arguments |
| --- | --- |
| Inference | `--schedule offline/online/online_fast`, `--iterations`, `--hard-assignments` / `--no-hard-assignments` |
| Exploration | `--hdp-preset`, `--max-clusters`, `--exploration-steps` |
| LDS | `--lds-update`, `--lds-scoring`, `--lds-degrees`, `--lds-step-size`, `--birth-lds-kl-per-dimension` |
| GP/alignment | `--fit-gp` / `--no-fit-gp`, `--alignment`, `--warp` / `--no-warp` |
| Resolution/subset | `--max-points`, `--max-series` |
| Calibration | `--components`, `--calibration-samples`; AE also `--epochs`, `--learning-rate` |
| Normalization | `--normalize-samples` / `--no-normalize-samples` |
| Diagnostics | `--plots`, `--projection-plots`, `--lds-plots`, `--lds-dynamics-plots`, `--warp-plots`, `--birth-plots N` |
| Reproducibility | `--seed`, `--output-dir`, `--resume`, `--verbose silent/progress/debug` |

Boolean switches support their `--no-...` counterpart. `--no-plots` disables
all PNGs; saving closes figures without displaying them. Warp plots are made
only when alignment is active. Birth plots capture the first N proposals and
are currently available only offline. A requested diagnostic failure marks
the run as failed; missing captured birth curves are reported explicitly.

By default, offline uses `conjugate_batch`, and online schedules use
`conjugate_sequential`. An explicit batch request online is rejected. Fixed
affine observations require CPU, one output, final scoring, no warp, no GP
kernel fitting and a non-legacy LDS update. They use the exact existing affine
Gaussian likelihood; encoder coordinates are not treated as noiseless states.

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
| ECG autoencoder | Chronological; fixed trained affine C | Gamma per cluster; Sigma fixed |
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
python examples/mitbih_waveform_clustering.py --records 100 --representation pca --no-warp --no-fit-gp --components 12
```

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
