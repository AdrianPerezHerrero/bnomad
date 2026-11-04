# Documentation and CI audit: 0.3.0

This audit concerns the current bnomad source tree, whose Python distribution
and import are `hdpgpc`. Historical 0.2 and initial 0.3 migration reports remain
records of their own revisions. Their archived machine paths and unbundled
benchmark filenames are presented as text rather than broken hyperlinks.

The current guides cover the supported observation models, common dataset
arguments and result folders, cluster-clock versus physical-clock semantics,
and the distinction between retrospective clustering and failure prediction.
Repository metadata uses the canonical bnomad URL. The
[bibliography](references.md) distinguishes software citation from method and
dataset publications; no software DOI or arXiv number is declared.

## Diagnosed hosted CI failure

In [Actions run 41671889587, Linux/Python 3.12](https://github.com/AdrianPerezHerrero/bnomad/actions/runs/41671889587/job/126042147647),
the wheel installed successfully. Provenance, dependency checks, the pinned
reference and all five standalone examples passed. The suite then reported
292 tests, two failures and four skips:

- Float32 warp covariance: an exact-equality assertion failed by 5.96e-8,
  approximately one rounding step.
- Reduced-inducing GP training: optimized log-kernel parameters differed by
  3.71e-7; the other frozen scientific output checks remained in place.

Neither test selects an HDP preset or tests a target cluster count. Sigma,
Gamma and HDP presets were not tuned to conceal these failures. Narrow
precision-aware bounds are described in [reproducibility](reproducibility.md),
and the original fixtures are retained. CI now prints every unittest failure
block before its log tail. Official Node 24 action versions replace deprecated
Node 20 actions; that warning was separate from the numerical failures.

The complete artifacts for the other jobs exposed additional portability
assertions hidden by the same log-tail truncation. Windows/Python 3.11 and 3.12
reported 12 failures; Windows/Python 3.10 reported 13; Linux/Python 3.10 reported
three. Eight Windows failures compared floating-point optimizer objective traces
exactly, two concerned saved scalar/batch warm-start controls, and Python 3.10
also compared interpreter-specific English `UnboundLocalError` wording.
These contracts now distinguish numerical traces from discrete results and
compare the exception type and uninitialized variable across interpreter versions.
See [reproducibility](reproducibility.md) for the measured bounds and guard tests.

The printed zebrafish purity/ARI values come from small workflow fixtures used
to check loading, configuration and saved artifacts. Those tests do not assert
a biological segmentation accuracy threshold. Low printed scores are separate
from the failing numerical assertions; judging real-data recovery requires a
scientific experiment with suitable priors, controls and evaluation.

The inspected hosted run targets the original commit. Re-running that
commit would exercise the original assertions; validation of the changes
requires running CI after the reviewed changes reach a branch.

## Repeatable checks

```sh
python tools/check_documentation.py --output documentation.json
python -m build
python -m twine check dist/*
python tools/check_artifacts.py dist
# Install the wheel noneditably, then run from an unpacked source archive:
python tools/verify_installed.py --output /new/evidence/directory
# With the tutorials extra installed, from the checkout/source archive:
python tools/validate_tutorials.py --output tutorials/.validation/new-run
```

The offline documentation checker validates local paths, Markdown/notebook
heading anchors and release metadata. It inventories external URLs for separate
publisher/creator verification; CI does not depend on remote publishers being
available. The installed-package verifier also runs this check.

## Initial local validation, 2026-11-03

| Check | Result |
| --- | --- |
| Authored documentation/notebook links and metadata | 81 files, 517 links, no missing local paths or headings after continuation |
| Historical hyperlinks | 106 corrected or converted to explicit archival reference text |
| External URLs | 33 checked: 27 HTTP 200, three HTTP 202, three HTTP 403; no HTTP 404 |
| Wheel/source archive | Isolated build, twine and artifact-content checks passed |
| Installed-package provenance and dependencies | Passed; imports came from the audit environment's site-packages |
| Console/pinned reference and standalone examples | Passed, including all five example modes |
| Complete installed regression suite | 296 tests in 254 seconds; four skips, no failures |
| Fresh-process characterization captures | Three runs, including reversed case order; exact repeats and no baseline regressions |
| Paired performance check | Exact outputs; current/control median ratio 0.920, limit 1.25 |
| Real-record notebooks | All eight executed twice, every cell; exact scientific-report equality for all pairs |
| README ECG illustration | Reproduced five clusters, purity 0.96875, ARI 0.87409, NMI 0.83173 |

The local platform was Linux, CPython 3.12.3 and CPU PyTorch 2.9.0+cpu with
single-thread scientific settings. The audit environment installed the built
wheel noneditably and reused the existing pinned scientific dependencies; it
was not a fresh resolution/download of every dependency. Source tests ran from
an unpacked archive outside the checkout. Notebook code cells and scientific
recipes are unchanged. Generated evidence is kept under
`agents/documentation_ci_audit/`, outside release artifacts.

Publisher HTTP 202 challenge pages and HTTP 403 access restrictions leave some
external page contents unverified. Publication titles/years were checked against
publisher, creator or Crossref metadata where accessible. These responses are
not treated as missing DOI records.

The final rebuild retains byte-identical package Python and resource content;
its archive documentation and test/checking tools are checked separately.
These local results do not establish success across the declared six-job
Linux/Windows and Python 3.10–3.12 matrix or on the hosted Google Colab service.

## Continued clean-environment validation

The continuation checked the completed jobs across the original six-job matrix
and added guards for the newly recovered Windows and Python 3.10 assertions.
It also removed the initial dependency-reuse limitation for installation testing:
new Python 3.10.20 and 3.12.3 environments installed CPU PyTorch and the wheel
through normal dependency resolution, without shared site-packages or path files
pointing at the development environment.

| Check | Result |
| --- | --- |
| Fresh Python 3.12 installation | Installed wheel and resolved scientific dependencies independently; pip check passed |
| Scientific import provenance | All ten inspected package/scientific imports came from the new environment |
| Complete installed suite, Python 3.12 | 299 tests in 251.551 seconds; four skips, no failures |
| Focused Python 3.10 portability checks | 80 tests; one CUDA skip, no failures |
| Pinned public reference, Python 3.10 | Passed; labels `[0, 0, 1, 1, 0, 1]` |
| Fresh Python 3.12 release gates | Console, reference, five examples and every verifier stage passed |
| Fresh-process captures, Python 3.12 | Three exact repeats, including reversed order; no baseline regressions |
| Performance, Python 3.12 | Exact outputs; current/control median ratio 0.914, limit 1.25 |
| Frozen fixtures and production implementation | Unchanged |

The original Python 3.10 bootstrap pip rejected normalized dependency names
from the CPU PyTorch index and fell back to a source build. Upgrading pip before
installing CPU PyTorch resolved that setup failure. The release checklist and
tutorial workflow now include this bootstrap explicitly, matching the main
installed-package workflow and installation guide.

Windows assertion guard tests use the reported numerical differences, and
both local interpreters exercise the affected regression cases. A hosted Windows
execution of the updated assertions remains required; local Linux results and
replays of reported values do not establish that every later Windows boundary
passes. The eight paired notebook executions above remain the tutorial evidence.
