# XJTU-SY raw-window PCA and autoencoder clustering

```sh
python examples/xjtu_sy_autoencoder_clustering.py --download
python examples/xjtu_sy_autoencoder_clustering.py --download --configurations independent dynamic --resume
python examples/xjtu_sy_autoencoder_clustering.py --records Bearing1_1 Bearing1_4 --download --resume
```

The first command downloads and clusters the complete `Bearing1_1` lifetime.
All 15 bearing names are supported. Start with bearings from one operating
condition before interpreting differences between speed/load conditions.
The [dataset creator](https://biaowang.tech/xjtu-sy-bearing-datasets/)
provides operating conditions, acquisition settings, file counts and terminal
fault inspections. Cite Wang et al., *A Hybrid Prognostics Approach for
Estimating Remaining Useful Life of Rolling Element Bearings*, DOI
[10.1109/TR.2018.2882682](https://doi.org/10.1109/TR.2018.2882682), when using these data.

## PCA variant

```sh
python examples/xjtu_sy_pca_clustering.py --download --configurations independent dynamic --resume
python examples/xjtu_sy_pca_clustering.py --records Bearing1_1 Bearing1_2 Bearing1_3 Bearing1_4 Bearing1_5 --components 10 --sigma-prior 0.25 --gamma-prior 0.5 --configurations independent dynamic --resume
```

PCA and AE entry points share the same runner, raw acquisition windows,
calibration selection, global channel scaling, priors, plotting and resume
conventions. PCA directly fits a fixed affine decoder using the existing SVD
calibration helper; it applies no earlier projection. The same rank and noise
floor controls are available. `--epochs` and `--learning-rate` affect only the
AE. PCA residual covariance uses the native sample-variance estimate; both
variants apply the requested absolute/fractional noise floor.

PCA outputs default to `agents/xjtu_sy_pca_runs/`. Each run saves
`pca_calibration.npz` with decoder/noise, calibration provenance, latent variances
and calibration explained-variance ratio, plus `pca_calibration.png` with
cumulative variance and reconstruction examples. All standard assignments,
latent trajectories and comparison plots remain available. Diagnostics include
`reconstruction_explained_variance`/`reconstruction_scaled_mse` and correctly
named `pca_*` or `autoencoder_*` aliases. AE artifacts retain their original names.

The tied linear AE and PCA target similar low-rank reconstruction objectives.
PCA offers a direct SVD solution; changing to PCA does not remove the acquisition
phase uncertainty or automatically make the states valid health labels.

## Two different clocks

Each numbered CSV contains 32,768 samples at 25.6 kHz (1.28 seconds), with
horizontal and vertical acceleration in that order. A new burst is recorded
once per minute. There are approximately 58.72 seconds of unobserved vibration
between bursts; these gaps must not be treated as adjacent raw samples.

Each clustering observation is **one raw window inside one burst**. Defaults:
512 samples per channel, offset zero, flattened in time-major/channel-interleaved
order into 1024 features. Consecutive observations belong to consecutive
numbered acquisitions, approximately one minute apart. `--step` advances in
acquisitions, not raw samples. Times use the first scheduled acquisition as
origin: file n starts at `(n-1)*60` seconds; the window timestamp is its end.
This convention does not assert the exact delay between installation and file 1.

```sh
python examples/xjtu_sy_autoencoder_clustering.py --window-samples 1024 --window-offset 4096 --resume
python examples/xjtu_sy_autoencoder_clustering.py --step 2 --resume
```

The same offset is used for every acquisition. No window crosses a burst
boundary, no gaps are filled, and files are sorted by number. Missing requested
files cause an error. `--max-series` selects a chronological prefix for both
fitting and calibration; such a prefix is explicitly distinguished from a
complete run-to-failure record. Known terminal inspection remains bearing-level
metadata even when the terminal acquisition is absent from a selected prefix.

## Representation and dynamics

The existing tied **linear orthonormal autoencoder** learns a shared decoder,
which is frozen for exact Gaussian LDS inference. Defaults: eight latent
components, 250 epochs and at most 400 balanced calibration acquisitions.
There is no preliminary PCA, spectral transform, envelope extraction,
phase alignment, per-window normalization or hand-designed health indicator.
One global mean/scale per channel preserves amplitude differences. Each bearing
is fitted separately; shared decoder coordinates do not imply shared cluster IDs.

Calibration pools selected raw windows retrospectively without health labels.
Optional `--calibration-records` freezes the decoder/scaling on explicitly
selected training bearings. It is not a held-out detector unless training and
evaluation bearings are separated deliberately. The independent arm fixes A=0;
the dynamic arm learns A and Gamma. Both preserve the requested HMM topology.
Covariance controls and structural inference use the same shared configuration
builder as flutter and expression experiments.

Unlike consecutive flutter windows, acquisition bursts are **not synchronized
in vibration phase**. A reconstruction-trained AE may retain phase or noise,
while low-rank least squares can favor large end-of-life amplitudes. This is a
raw-data baseline, not a promise of successful degradation-state recovery.
Repeat with other offsets, window lengths, ranks and seeds before interpreting
small clusters as health stages. The implementation limits windows to 2048
samples/channel because the fixed affine observation covariance is materialized.

A describes evolution between **retained acquisition windows**, not fast
mechanical oscillation modes. If a cluster skips observations, its clock also
skips acquisition times. Diagnostics record this and withhold per-second rates
when the retained clock is irregular. Even with a regular clock, rho(A)>1 is a
model-conditional acquisition-coordinate property, not a physical failure
criterion. Variational uncertainty excludes representation/model-selection
uncertainty. Smoothed trajectories use future observations.

## References and plots

The source supplies terminal fault elements and lifetimes, not intermediate
healthy/degrading/failing boundaries. Reports therefore contain no fabricated
ARI, purity, health-stage labels or percentage-of-life segmentation targets.
Terminal fault type is evaluation metadata, never a training input.

Full-burst RMS and absolute peak are computed solely for plotting the lifetime
against inferred states. These summaries **do not enter the autoencoder,
channel scaling or clustering**. The summary also shows the published terminal
fault description and conditional transition-radius uncertainty. Latent plots
use acquisition time and break paths at MAP membership gaps.

Outputs default to `agents/xjtu_sy_autoencoder_runs/`:

- `comparison.json/.png` and the standard `results.csv` table.
- `xjtu_sy_<bearing>/<run_id>/`: configuration, settings/source fingerprint,
  log, unlabeled assignments, result metadata and LDS parameters.
- `prepared_data.npz`: actual raw/scaled windows, acquisition numbers/times,
  offsets, full-burst display summaries and SHA-256 hashes of source CSVs.
- `autoencoder_calibration.npz`: shared decoder/covariance, applied Sigma,
  channel scaling, losses, calibration bearing names and acquisition numbers.
- `latent_trajectories.npz/.png`, `latent_embedding.png`, `dynamics.json`,
  `autoencoder_calibration.png` and `clusters_output_0.png`.

`--no-trajectory-plots`, `--no-projection-plots` and `--no-plots` follow the
existing conventions. Complete resumed runs skip AE training and inference.
Logs print absolute result folders.

## Data cache and integrity

Default cache: `agents/data/xjtu_sy/<bearing>/<number>.csv`. Alternatively point
`--data-dir` at a locally extracted dataset with bearing folders directly below
it, below `35Hz12kN`/`37.5Hz11kN`/`40Hz10kN`, or below
`XJTU-SY_Bearing_Datasets/<condition>/`.

The creator's official mirrors package a large archive. `--download` uses the
[public archive mirror](https://huggingface.co/datasets/DavidNguyen/XJTU-SY_Bearing_Datasets)
at pinned revision `274226336036cd8deb2c40b4ba323af2d11a7290`, requesting only
selected ZIP members through bounded HTTP byte ranges. It refuses a server
that ignores Range requests rather than downloading the full 5.4 GB ZIP.
Members are ZIP-CRC checked and parsed before caching; no archive scripts run
and no arbitrary paths are extracted. `download_manifest.json` records source,
mirror revision, ZIP member/CRC and locally computed SHA-256 hashes. Cached
files are checked against that manifest, and interrupted downloads resume from
completed members. The creator provides no official per-file SHA manifest:
these local hashes verify cache integrity, not independent authenticity of the
third-party mirror.
