# 0.3.0 ? native implementation migration

- Removed legacy model/GP/warp modules and compatibility exports. Old import and pickle paths intentionally break.
- Native typed state composes cluster lifecycle, HMM messages, objective scoring and output weighting from their owning modules.
- GP prior/prediction, training, and GPyTorch models are owned by emissions; alignment directly exports its native AMTGP class.
- Engine entry points are `OnlineInference.update/update_fast` and `OfflineInference.fit/classify_batch/fit_labels`.
- Removed flat configuration adapters. Alignment policy `gp` replaces the serialized default token `legacy`; scientific defaults are unchanged.
- Kept the numerical fixtures, tolerances, algorithm order, optimizer settings and known failure cases unchanged.
- See docs/migration_0_3/report.md for validation, compatibility breaks and measurements.

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
