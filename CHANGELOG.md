# 0.3.0 — native implementation migration

- Added opt-in sigma/gamma prior screening with one frozen representation,
  short native fits, pre-update predictive validation, adaptive refinement,
  fit/time budgets and shared experiment reports. Both priors default to a
  five-point logarithmic grid over 1e-3–10.0, with five clips of up to 128 training
  and 32 validation observations, 60 total pilot fits and a 120-second budget
  including local refinement and repeat checks. Explicit
  ranges and budgets remain available. Plots use a common first-clip score for
  grid colors and a separate confirmation/selection panel. Normal inference is
  unchanged.

- Reconciled bundled MIT-BIH record 203 with the corrected reference by source
  sample: removed one obsolete artifact window/label and retained 2,979 paired
  beats. Saved annotation sample positions now propagate into record reports
  and data fingerprints; the unavailable filtered beat is documented.

- Fixed determinism verification to use the same case-relative baseline paths
  as unit tests and report exact per-case differences. Repeated captures still
  require byte-for-byte equality.
- Verified reduced-grid query covariance at the reference fitted white-noise
  value, accounting for the exact diagonal noise term while retaining the
  original residual covariance and prediction checks.

- Added opt-in offline latent-trajectory plots to MIT-BIH waveform, PCA and
  autoencoder scripts through the shared record runner, with both dynamics clocks.

- Fixed initial GP-prior recognition after kernel cloning so machine-rounding
  differences cannot add an unintended LDS prediction step. Added a one-ulp
  regression and retained the learned-covariance evolution check. Portability
  checks cover fitted noise metadata and saved warp optimizer controls while
  retaining scientific-output and exact-repeat checks.

- Standardized core and all dataset-runner sigma/gamma noise prior defaults to
  0.1, including Xenopus clutch defaults. Explicit recipes and fixed covariance
  policies retain their specified values; see docs/noise_priors.md.

- Audited current documentation, tutorials, repository URLs and bibliography;
  added an offline link/metadata check to installed-package validation.
- Scoped float32 covariance and optimized GP log-kernel fixture tolerances to
  observed cross-CPU rounding; model priors and frozen reference data are unchanged.
- Improved CI failure visibility and upgraded GitHub actions to Node 24.
- Accounted for Python 3.10 exception wording and Windows alignment optimizer
  trace/control precision while retaining scientific output checks and exact
  repeated-capture requirements.

- Added XJTU-SY Fourier/Haar example scripts through the shared runner, with
  channel-wise calibration-energy rank selection, paired Fourier coefficients,
  basis provenance, and physical-clock experiment defaults.

- Added opt-in physical-clock latent dynamics on a regular shared sampling grid,
  with prediction-only propagation, gap smoothing, clock-aware statistics and
  global process-prior scoring. Cluster-clock and the pinned reference recipe
  remain unchanged by default. All experiment builders expose the clock flag.

- Added XJTU-SY raw-acquisition-window PCA/autoencoder clustering with explicit
  acquisition clocks, bounded per-bearing downloads, terminal-inspection
  references and standard unlabeled artifacts. Shared transition diagnostics
  are reused by flutter and bearing experiments.
- Added checksum-pinned flutter motion loading and a raw-window autoencoder
  experiment with shared calibration, unlabeled clustering reports and
  variational transition-stability diagnostics. Shared affine configuration is
  reused by flutter and expression experiments.
- Added zebrafish and Xenopus PCA/autoencoder entry scripts using the shared
  expression workflow, fixed calibrated decoders, and standard artifacts.
- Added dense Xenopus GSE65785 expression experiments with separate clutches,
  unsmoothed all-gene inputs, published sample QC and explicit reference-label
  provenance. Zebrafish and Xenopus share expression configuration/execution.
- Added an all-gene zebrafish temporal expression-state experiment with
  versioned/checksummed count loading, shared experiment arguments/results,
  and paired independent-factor/HMM versus dynamic-factor/HMM configurations.
  See docs/zebrafish_workflow.md.
- Added opt-in left-to-right transition topology for CPU offline split/merge:
  reduced-simplex hierarchical priors/ELBO, ordered messages and joint decoding,
  temporal boundary splits and adjacent merges. See docs/transition_topology.md.
- Added learned affine factor-analysis observations through the native HDPGPC
  state and inference strategies, with independent factors by default, optional
  LDS evolution, and a CAS-Colon example. See docs/mfa_emission.md for the
  mathematical formulation, supported policies and validation status.
- Removed legacy model/GP/warp modules and compatibility exports. Old import and pickle paths intentionally break.
- Native typed state composes cluster lifecycle, HMM messages, objective scoring and output weighting from their owning modules.
- GP prior/prediction, training, and GPyTorch models are owned by emissions; alignment directly exports its native AMTGP class.
- Engine entry points are `OnlineInference.update/update_fast` and `OfflineInference.fit/classify_batch/fit_labels`.
- Removed flat configuration adapters. Alignment policy `gp` replaces the serialized default token `legacy`; scientific defaults are unchanged.
- Kept the numerical fixtures, tolerances, algorithm order, optimizer settings and known failure cases unchanged.
- See docs/documentation_audit.md for current validation and
  docs/migration_0_3/report.md for the initial migration measurements.

# Changelog

## 0.2.0 — release preparation (not yet published)

- Added eight public-API, real-record Colab tutorials, shared plotting/loading
  helpers, an uploadable isolated-install bundle and automatic repeated execution.

- Added a typed, serializable configuration and minimal `HDPGPC.fit/update` API.
- Isolated emissions, dynamics, alignment, HDP calculations and inference schedules
  while retaining legacy imports and characterized update ordering.
- Added deterministic small characterization fixtures, exact inference traces,
  integration tests, public API examples and package-install verification.
- Added Python 3.10–3.12 dependency bounds, optional dataset/notebook dependencies,
  wheel/sdist metadata, CI, citation and reproducibility guidance.
- Extended installed-package gates to invoke the console entry point and all five
  standalone examples in isolated Python processes, without checkout imports.
- P01 (`7d259c4`): scalar warp control lookup prepared once per call; paired CPU
  runtime reduced 5.39–9.04%, with exact captured results. No peak-memory saving.
- P02: reuse temporary nongradient HMM message rows; measured post-startup online
  runtime reduced 4.22–5.53%, with exact states and original autograd/subclass fallback.
- No old API is deprecated. Existing unsupported online multi-output/warping
  paths remain documented; no inference formulas were changed for packaging.

## Earlier research implementation

Historical metadata declared 0.1.0 while the README already advertised 0.2.0.
This preparation makes the version consistent. It does not invent a historical
release date, publication DOI, or claim that an unpublished release was uploaded.
