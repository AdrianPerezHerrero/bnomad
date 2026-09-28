> Historical 0.2 evidence. Source paths and compatibility instructions below describe the archived release, not 0.3. Use [the current architecture](architecture.md), [migration guide](migration.md), and [0.3 validation report](migration_0_3/report.md) for this release.

# Release validation record

The original evidence below applies to packaging checkpoint `49d59a3`.
The post-P02 verification at the end of this document validates the current
package, including performance checkpoint `c43a930`. Optimization measurements
remain separate in `performance_changes.md`.

Packaging preparation on 2026-09-26, Windows, CPython 3.11.0, CPU. The fresh
virtual environment has `include-system-site-packages = false`; imports resolve
under its own `Lib/site-packages`. No editable install or checkout PYTHONPATH is
used for artifact verification. Other Python/OS combinations await the CI matrix.

## Dependency investigation

An unconstrained-within-initial-ranges `pip install .` succeeded in the fresh
environment, resolving torch 2.9.1. `pip check`, artifact checks, all 15 original
characterization tests and public offline reference comparisons passed. The
initial full run reported 216 tests, 44 failures, two CUDA skips: 43 exact-trace
failures from dependency differences, plus one incorrect expectation in the new
online tutorial test. That expectation assumed the static growth fixture's labels;
the dynamic tutorial actually returns `[0, 0, 0, 0]` in both environments. Only
the new test/documentation was corrected; existing fixtures were untouched.

The first exact trace divergence was `GPI_model.fit_kernel_params` at an exit
boundary: `cov_f[0][0,0]` was `0.1492108400120193` instead of
`0.14921084001201937`. Public offline result maximum absolute difference was
`5.551115123125783e-17`; labels were identical. Torch 2.9.0 and 2.9.1 Windows
wheels report bundled MKL 2025.2 and 2025.3, respectively. Downgrading only torch
to 2.9.0 restored the first failing exact test. Metadata therefore constrains
`torch>=2.9.0,<2.9.1`; no inference implementation or tolerance was changed.

The development baseline passed 213 tests in 277.105 seconds (two CUDA skips).
This includes the 198 existing repository tests and 15 original characterization
tests, now copied into the repository. New release integration checks add three
tests, exercising all five public workflows twice and checking installed metadata.

## Original packaging verification

Completed on 2026-09-27 in a second fresh venv, `.release-final`, installed with
`python -m pip install .` using the final dependency bounds. The source archive
was unpacked under `.release-verification/final/hdpgpc-0.2.0`, outside the Git
checkout. `tools/verify_installed.py` ran there with no PYTHONPATH and imports
verified under `.release-final/Lib/site-packages`.

- All 216 tests passed (214 passed, two unavailable-CUDA skips), 298.039 seconds.
  The original 15 cases and all exact inference/configuration/public API traces
  are included. No reference tolerance was changed.
- All five public tutorial modes ran twice with exactly equal scientific results.
- Three fresh-process captures of all 15 original cases, including reversed order,
  matched the original baseline bytes exactly: zero differences in every run.
- Offline public reference passed; pip reported no broken requirements.
- Paired scalar warp performance smoke: current median 0.029856 s, preserved
  control median 0.032610 s; ratio 0.915545, exact output/loss equality,
  maximum absolute difference zero. This checks the existing P01 change and is
  not a new optimization or a new general speedup claim.
- Wheel and sdist built successfully, passed `twine check` and explicit artifact
  inventory checks. The wheel contains configuration/reference resources and both
  licenses; large datasets, historical results and bytecode are excluded.
- All 49 pre-existing Python modules match checkpoint `4db9bf1` after normalizing
  checkout line endings; all 16 original fixture JSON files match byte-for-byte.

Evidence: [gate status](release_evidence/status.json),
[full environment](release_evidence/environment.txt),
[import provenance](release_evidence/provenance.txt),
[reference result](release_evidence/reference.json),
[determinism hashes](release_evidence/determinism.json),
[performance samples](release_evidence/performance.json), and
[source/fixture audit](release_evidence/source_audit.json).
The full local test log remains in workspace `release-final-evidence/tests.log`;
CI uploads complete gate logs. Numerical/package deprecation warnings remain
visible, including NumPy matrix and third-party parser warnings.

The Python 3.10/3.11/3.12 Linux/Windows CI matrix is configured but has not been
executed on GitHub from this session. CUDA, remote publication and installation
of a published artifact are not claimed. These remain explicit release gates.

## Current verification after P02

Completed on 2026-09-27 against production checkpoint `c43a930`, including P02.
This follow-up changes verification tooling and documentation only. All 52
package Python files match that checkpoint after normalizing checkout line endings;
all 16 original fixture files are byte-identical to the original workspace copies.
No inference mathematics, defaults, test baselines or tolerances were changed.

A new venv, `.release-current`, was created without system site packages. A newly
built source archive was unpacked outside the checkout at
`.release-verification/current/hdpgpc-0.2.0`. From that directory, the new venv's
interpreter ran `python -m pip install .`, followed by
`python tools/verify_installed.py --output ../../../release-current-evidence`.
Installation resolved the bounded dependencies normally; no editable install,
manual path manipulation or preinstalled scientific stack was used. Imports were
verified under `.release-current/Lib/site-packages`. The independently built wheel's
59 package files also match the installed files byte-for-byte.

- **221 tests: 218 passed, three unavailable-CUDA skips**, in 304.008 seconds.
  This includes unit, original characterization, exact inference/configuration
  traces, public API integration, P02 message-row tests and installed metadata.
- The actual `hdpgpc-reference` console executable and the isolated module command
  both passed the packaged reference and returned identical scientific results.
- All five standalone scripts (online, offline, static, warping, multi-output)
  passed with `python -I` against the installed package. The integration suite
  additionally ran all five modes twice with exact scientific-result equality.
- All 15 original cases matched the reviewed snapshot bytes in three fresh
  interpreters, including reversed case order: zero differences in every run.
- `pip check` reported no broken requirements. Wheel/sdist build, `twine check`
  and artifact-content checks passed.
- The paired scalar warp smoke passed with exact output/objective equality:
  current median **0.030067 s**, preserved control **0.033414 s**, ratio **0.899822**,
  maximum absolute difference **0**. This validates the existing performance gate;
  it is not a new optimization or a cross-machine performance guarantee.

The final archives were rebuilt after the documentation/evidence update. Their
package payload was compared again against the tested installation; only release
documentation and verification tooling changed during this task.

Archived evidence includes [all gate statuses](release_evidence/post_p02/status.json),
[complete test log](release_evidence/post_p02/tests.txt),
[environment](release_evidence/post_p02/environment.txt),
[venv isolation](release_evidence/post_p02/pyvenv.txt),
[import provenance](release_evidence/post_p02/provenance.txt),
[reference results](release_evidence/post_p02/reference.json),
[determinism](release_evidence/post_p02/determinism.json),
[performance samples](release_evidence/post_p02/performance.json), and
[source/fixture audit](release_evidence/post_p02/source_audit.json).
Individual console/example logs are in the same directory. Existing numerical and
third-party deprecation warnings remain visible in the logs.

Packaging preparation is complete locally. The configured Python 3.10/3.11/3.12
Linux/Windows CI matrix still needs a remote run before tagging or publication;
this session validates Windows Python 3.11 CPU only. Publication and CUDA
reproducibility are not claimed. See the [release checklist](release_checklist.md).
