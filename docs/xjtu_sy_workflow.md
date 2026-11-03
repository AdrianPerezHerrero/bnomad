# XJTU-SY raw-window PCA, autoencoder, Fourier and wavelet clustering

Before inference, each runner prints the emission name, actual component count,
and explained variance on both the calibration windows and the current bearing's
scaled raw windows. Fourier/Haar also report the component count per sensor.
The line is saved in `clustering.log`; resumed runs print the saved values.

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

## Fourier and Haar variants

```sh
python examples/xjtu_sy_fourier_clustering.py --records Bearing1_1 Bearing1_4 --components 8 --resume
python examples/xjtu_sy_wavelet_clustering.py --records Bearing1_1 Bearing1_4 --components 8 --resume
python examples/xjtu_sy_fourier_clustering.py --records Bearing1_5 --components 16 --sigma-prior 0.1 --gamma-prior 0.5 --configurations independent dynamic --resume
```

These entry points use the same raw windows, scaling, balanced calibration,
configuration builder, inference and output schema as PCA/AE. Their experiment
default is **physical-clock**; select `--dynamics-clock cluster-clock` explicitly
for a clock comparison. Existing PCA/AE defaults are unchanged. Data download
uses the same `--download` option and cache.

Each sensor gets a separate native Fourier or Haar basis on its actual sampling
grid. The interleaved horizontal/vertical vector is never transformed as one
continuous signal. `--components` is the **total** latent dimension, default 8:
four coefficients per sensor. Odd budgets allocate one extra coefficient to the
horizontal channel. At least two components are required.

The basis orientation is fixed. The shared unlabeled calibration windows select
coordinates by centered coefficient energy, separately for each sensor. Fourier
sine/cosine pairs are selected together; an exact-rank energy maximization keeps
each pair intact. DC and Nyquist, when available, each cost one coordinate.
Some rank/candidate-bin combinations cannot retain whole pairs and are rejected.
Haar coefficients are selected individually from the native coarse-to-fine basis.
The chosen coordinates stay fixed across all requested bearings/configurations.
This selection is retrospective when calibration uses the evaluation lifetimes.

There is **no PCA preprocessing, learned rotation, per-window normalization,
phase removal, power spectrum or coefficient thresholding per observation**.
The model still scores and reconstructs raw waveforms with a fixed affine
Gaussian decoder. Calibration also estimates the waveform offset, diagonal
residual covariance and regularized latent covariance. Existing `--sigma-prior`,
`--sigma-mode`, `--gamma-prior` and LDS controls apply through the shared builder.
AE optimizer arguments are accepted for CLI consistency and do not train a network
in the basis variants.

Fourier-specific controls:

- `--window-size` / `--fourier-window-size`: non-overlapping Fourier block length
  **per sensor**, default the entire `--window-samples` raw window.
- `--harmonics h`: candidate bins 0 through h within each block.
- `--frequency-bins k ...`: explicit candidate local bins, mutually exclusive
  with `--harmonics`. Candidates are still subject to the total component budget.
- With no bin filter, all native bins are candidates. `basis.json` records actual
  physical frequencies, including shorter tail blocks.

Wavelets currently support `--wavelet haar`; `--window-samples` must be a power
of two. Full bases are available by requesting `--components 2*window_samples`
(supply the integer value, e.g. `--window-samples 16 --components 32`). Full bases
can have more coordinates than calibration acquisitions because their directions
are fixed, but estimating their dynamics requires enough evidence. At the usual
512 samples/channel, the full basis would give a 1024-dimensional LDS; the default
rank budget avoids that cost.

Outputs default to `agents/xjtu_sy_fourier_runs/` and
`agents/xjtu_sy_wavelet_runs/`. In addition to the usual saved results, each run
contains `basis.json` with selected channel/coefficient identities, the
`fourier_calibration.npz` or `wavelet_calibration.npz` decoder and provenance,
and a matching calibration plot. Noise floors keep a full-basis residual
covariance positive definite even when reconstruction is exact.

Signed Fourier coefficients retain phase and Haar coefficients retain event
position. These scripts test raw-waveform basis models; spectral-power/envelope
features would require a separate preparation path. Exact full-basis
reconstruction alone does not establish useful clustering or mechanical modes.

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

In the AE variant, the existing tied **linear orthonormal autoencoder** learns a shared decoder,
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

A describes evolution on the selected **acquisition clock**, not fast mechanical
oscillation modes. In cluster-clock mode a cluster skips acquisition times when
it skips observations. Physical-clock instead propagates it through all global
sampling ticks. Diagnostics record the convention and withhold per-second rates
when the provided clock is irregular. Even with a regular clock, rho(A)>1 is a
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
use acquisition time. Cluster-clock paths break at MAP membership gaps;
physical-clock paths show propagation/smoothing through unobserved ticks.

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

## Optional physical-clock evolution

Use `--dynamics-clock physical-clock` to propagate each component between all
global observation ticks, including unassigned observations. Cluster-clock
remains the PCA/AE default; the new Fourier/Haar runners default to physical-clock.
See [dynamics clocks](dynamics_clocks.md) for prior/scoring
semantics, regular-sampling assumptions and compatibility.
