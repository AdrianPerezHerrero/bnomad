# Daphnet waveform baseline

This example applies the ordinary GP/LDS HDP-GPC model directly to acceleration
windows. There is no encoder, PCA, Fourier or custom fixed observation adapter.

```powershell
python examples/daphnet_clustering.py --download --records S02R01
python examples/daphnet_clustering.py --records S02R01 --schedule online_fast
python examples/daphnet_clustering.py --records S02R01 --warp
python examples/daphnet_clustering.py --records S02R01 --normalize-samples
python examples/daphnet_clustering.py --records S02R01 --lds-preset all_free
python examples/daphnet_clustering.py --records all --segments all --no-fit-gp
```

## Data and windows

The [official UCI Daphnet archive](https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait)
defaults to ignored `agents/data/daphnet/daphnet.zip`. `--download` fetches it
if absent; `--source` accepts another ZIP or extracted directory. Use
`--list-records` to list IDs. ZIP members are read without extraction.

Source documentation specifies 64 Hz acquisition and eleven columns: timestamp
in milliseconds, nine acceleration channels in milli-g, and annotation. The
loader converts time to seconds and acceleration to g. Labels retain these codes:

- 0: outside the experiment;
- 1: experimental non-freezing (standing, walking or turning);
- 2: freezing.

Label zero selects experimental intervals. Labels 1/2 only enter evaluation,
not model fitting or window selection. The default channel is ankle vertical;
`--channel` selects any of the nine axes. This one-channel baseline supports
both online and offline inference without claiming joint sensor coupling.

Default windows are two seconds, non-overlapping, with 128 native samples
anti-alias resampled to 64 points (32 Hz) using polyphase filtering. The GP grid
is local time in seconds. Window/stride durations must be multiples of 1/64 s.
`--points` may reduce resolution but cannot upsample. Window annotations use
raw majority vote (ties mean freezing); mixed windows remain included and their
exact `freeze_fraction` is exported. Incomplete trailing windows are omitted.

Windows never bridge annotation-zero periods, timestamp resets or gaps longer
than 1.5 sample periods. Each contiguous experimental segment gets a fresh
model. `--segments longest` is the default, selecting by length, not class;
`--segments all` fits every eligible segment separately. `--max-windows` selects
a chronological prefix. Subjects/runs and disconnected intervals are not joined
into an artificial trajectory. Subject-level evaluations should aggregate run
results as recommended in the source documentation.

Default scaling uses the first window's mean and standard deviation for all
windows. Relative amplitudes are retained; future observations do not calibrate
online normalization. `--scaling none` retains g units and may require changing
the prior scales. No class balancing is used.
`--normalize-samples` instead z-scores each window independently using its own
mean and standard deviation, overriding `--scaling`. This removes between-window
offset and amplitude differences. Constant windows become zero without division
by zero. The default is off; use `--no-normalize-samples` to disable it explicitly.
Normalization parameters are saved as `sample_normalization.npz` in the segment
folder, and logs/results identify the effective normalization mode. This uses
only the current window and is available with every inference schedule and warp.
Warp is off by default. `--warp` enables the existing DTW-GP alignment for
offline (including conjugate batch), online or online_fast; `--no-warp` disables
it. The model applies its existing GP warp penalty in proposal scoring.
Configuration, logs and results record the selected warp setting, and cluster
plots identify aligned model coordinates when it is enabled.

## Model choices

`make_config` in the script is the single configuration/API location. Defaults:

- Dynamic model and chronological HMM conditioning enabled.
- `only_plastic`: C=I and Sigma fixed, A/Gamma learned per cluster. Select
  `--lds-preset all_free` for unrestricted LDS learning; existing presets remain
  selectable, including static alternatives.
- Observation/process prior scales 0.1 in normalized waveform units, excess
  LDS degrees 20, final scoring and dimension-normalized proposal LDS KL.
- Offline conjugate batch or online conjugate sequential updates. Natural
  gradient is available via `--lds-update` and `--lds-step-size`.
- Initially fitted RBF/white-noise GP reused across proposals/clusters;
  `--no-fit-gp` retains a fixed kernel. `--lengthscale` is in seconds and
  `--gp-noise` is distinct from the LDS observation noise.
- Hard assignments, HDP preset `more`, ten birth exploration steps, ten-cluster
  cap and twenty offline iterations.

Use `--sigma-prior`, `--gamma-prior`, `--lds-degrees`, `--hdp-preset`,
`--hard-assignments`/`--no-hard-assignments`, and the iteration/cluster controls
for comparisons. Verbosity supports silent, progress and debug.

The LDS evolves waveform representers between observations, not an identified
biomechanical state. Cluster-member steps may span unequal time intervals.
Phase differences can generate clusters without alignment. Optional DTW-GP
warping permits local timing adjustment, but is not general shift invariance;
no freezing-specific features are supplied.

## Outputs and interpretation

Outputs are under `agents/daphnet_results/<record>/segment_<number>/`:

- `clustering.log`, `config.yaml` and `results.json`;
- `assignments.csv`, including source rows, timestamps, raw freezing fraction,
  window annotation, cluster and cluster-majority annotation;
- `lds_parameters.npz`, storing final A, C, Gamma and Sigma;
- `clusters_output_0.png`, reusing the annotated summary/gallery;
- `lds_parameters_<page>.png` and `lds_dynamics.png`, using existing diagnostics.

A root `summary.json` collects results. Plots use blue for non-freezing and red
for freezing, are saved and closed without an interactive window. `--no-plots`
disables graphics; `--no-lds-plots` keeps only the cluster plot.

Purity, ARI and NMI compare clusters with window annotations. The confusion
matrix compares actual annotation with each cluster's majority annotation;
freezing precision/recall/F1 are also reported. Mapping and evaluation use the
same fitted windows: these are descriptive in-sample scores, not held-out
freezing-detection accuracy. Missing positive-class recall/F1 is reported null.

Reusable `hdpgpc.datasets` helpers (`daphnet_records`, `load_daphnet_record`,
`window_daphnet_record`) separate preprocessing from model selection. A future
encoder variant can use these same waveforms and metadata.

For attribution, cite Bachlin et al., *Wearable Assistant for Parkinson's
Disease Patients With the Freezing of Gait Symptom*, IEEE TITB 14(2), 436-446,
2010 ([DOI](https://doi.org/10.1109/TITB.2009.2036165)), as requested in the archive.
