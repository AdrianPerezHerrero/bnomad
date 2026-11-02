# Flutter motion clustering with an autoencoder

```sh
python examples/flutter_autoencoder_clustering.py --download
python examples/flutter_autoencoder_clustering.py --download --resume
python examples/flutter_autoencoder_clustering.py --records 0950 --resume
python examples/flutter_autoencoder_clustering.py --configurations independent dynamic --resume
```

This experiment uses the public supplement of [Santos et al. (2024)](https://doi.org/10.1007/s44245-024-00043-w).
The small publisher archive and each motion member have pinned SHA-256 checksums.
The loader reads only the known text members inside the ZIP and executes none
of the supplied scripts. Data cache: `agents/data/flutter/`.

## What is learned

Each observation is a **causal window of raw, synchronized motion**, not a set
of engineered velocities, frequencies or health indicators. Default windows
contain 60 samples from both channels, advancing by 24 samples: approximately
0.25 seconds of history and 0.1 seconds per LDS update. Source times are kept;
the dataset has approximately 240 Hz sampling with small record-specific
differences. Windows flatten in time-major, channel-interleaved order.

The existing **tied linear orthonormal autoencoder** learns a four-dimensional
subspace, then its affine decoder is frozen for exact Gaussian LDS inference.
It is a linear AE, not a newly introduced nonlinear emission model. There is
no preliminary PCA. Decoder fitting uses no response labels or physical
features. A deeper nonlinear AE would require a different inference contract.

One fixed channel mean/scale is fitted to the calibration recordings, so both
source motion columns contribute despite their different numerical units.
Amplitude changes within and across recordings survive this transformation;
there is no individual-window normalization, alignment or phase synchronization.
Motion columns remain in their supplied units; plots retain neutral channel
labels because the source text files have no column headers or unit metadata.

The calibration budget is balanced across source recordings before AE training.
All selected recordings share exactly the same decoder. Each recording is
clustered **separately**, preserving its own chronology and initial condition.
Local cluster IDs are not common physical-state classes across recordings.

## Source references and evaluation

| Record | Airflow speed | Published record context |
| --- | --- | --- |
| `0700_largeIC` | 7 m/s | Damped response, large initial condition |
| `0700_smallIC` | 7 m/s | Damped response, small initial condition |
| `0815` | 8.15 m/s in filename | Near-onset / marginal bounded response |
| `0840` | 8.4 m/s | Bounded limit-cycle response |
| `0900` | 9 m/s | Bounded limit-cycle response |
| `0950` | 9.5 m/s | Unstable response; test stopped to protect the apparatus |

These are record-level descriptions, **not sample-level segmentation labels**.
Exact instability-onset annotations and actual component breakage are not
provided. Accordingly reports save no purity/ARI/NMI or invented percentage-of-
lifetime labels. The reference description is shown next to inferred states.
Generic experiment saving now supports missing reference labels explicitly.

Default calibration pools selected recordings retrospectively. This evaluates
unsupervised representation/reconstruction and segmentation, not a held-out
failure detector. To freeze calibration on separate training recordings, use:

```sh
python examples/flutter_autoencoder_clustering.py --records 0950 --calibration-records 0700_largeIC 0700_smallIC --download
```

`--max-series` applies a chronological window prefix to fitting and calibration;
source channel scaling also uses only those selected prefixes. Shared
calibration may still use other selected recordings and is explicitly recorded.

## Dynamics and uncertainty

The dynamic arm learns A and Gamma per cluster. The independent arm fixes
A=0 and learns Gamma. Sigma is fixed to calibrated residual covariance by
default; `--sigma-prior` multiplies that covariance, while `--gamma-prior`
initializes the per-coordinate process variance. Neither arm is given the
published response descriptions as inputs or assignments.

`dynamics.json` includes each cluster's eigenvalues, spectral radius, MAP
support/recurrence, native transition posterior and optional MNIW posterior
samples. `--stability-samples` defaults to 200; 0 saves point estimates only.
The reported probability that rho(A)>1 is **conditional on the fitted
variational window model**, not a probability of physical breakage.

The LDS clock is successive observations *retained by that cluster*. If it skips
windows or has an irregular retained clock, conversion to growth per second is
withheld. A regular retained clock permits log(rho)/step_seconds, but the
interpretation is still the evolution of overlapping AE windows. Overlap,
external initial perturbations, short clusters, representation choice and
parameter uncertainty can produce misleading mechanical interpretations.
Smoothed reconstruction uses future observations and is not a causal forecast.

Compare the damped controls and bounded responses rather than treating every
rho>1 as a fault. Vary the window advance (including non-overlapping windows),
latent rank and noise priors before claiming physical mode recovery. An
unstable condition can be present from the start while its amplitude remains
small: correct inference need not invent a late healthy-to-fault boundary.

## Controls and artifacts

The scripts share the package's standard experiment arguments, logging,
configuration and save/resume conventions. Defaults are offline split/merge,
ergodic HMM assignments, at most six clusters, 15 outer iterations, three
exploration proposals, four latent dimensions and 250 AE epochs.
Use `--transition-topology left_to_right` only as an explicit ordered-control
experiment, not as evidence of monotonic damage.

```sh
python examples/flutter_autoencoder_clustering.py --components 6 --resume
python examples/flutter_autoencoder_clustering.py --step 60 --resume
python examples/flutter_autoencoder_clustering.py --no-birth-lds-kl-per-dimension --resume
```

Outputs default to `agents/flutter_autoencoder_runs/`:

- `comparison.json/.png`: per-record/configuration comparisons; IDs remain local.
- `autoencoder_structure.png`: common AE coordinates and learned coordinate
  norms over real time, without fitting extra display PCA or joining sequences.
- `flutter_<record>/<run_id>/`: standard configuration, log, assignments,
  results, LDS matrices and structural proposal history.
- `prepared_data.npz`: original motion/time prefix, causal window indices,
  raw windows and the actual scaled observations passed to inference.
- `autoencoder_calibration.npz`: shared fixed decoder, covariance, channel
  scaling, training losses and source recording/sample provenance.
- `latent_trajectories.npz`, `dynamics.json`: retained smoothed states and
  transition diagnostics including approximate posterior uncertainty.
- `latent_trajectories.png`: per-cluster latent paths, first three components
  over recorded window-end time with posterior uncertainty, and final-A
  one-step changes in the first two coordinates. Paths break at MAP membership
  gaps; arrows represent retained-window model steps, not fixed elapsed time.
  Enabled by default; disable with `--no-trajectory-plots` or `--no-plots`.
- Per-run plots: original motion, AE/LDS reconstruction, published context,
  inferred states, spectral-radius uncertainty, training loss and AE coordinates.

Complete resumed runs skip both AE training and inference. Logs print absolute
result folders. Runtime errors use the existing error report schema. Calibration
and LDS files are diagnostic artifacts rather than restart checkpoints.
