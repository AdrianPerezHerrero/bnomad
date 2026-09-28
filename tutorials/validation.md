> Historical tutorial preparation record for 0.2. Current verification is in [the 0.3 report](../docs/migration_0_3/report.md).

# Tutorial execution record

Validated 2026-09-27 on Windows CPU, Python 3.11.0, installed hdpgpc 0.2.0,
NumPy 1.26.4, SciPy 1.14.1, PyTorch 2.9.0 and GPyTorch 1.13. The complete
versions and SHA-256 hashes of the executed notebooks/scientific results are in
[validation.json](validation.json).

All **eight notebooks executed twice**, top to bottom, with no skipped cells or
suppressed inference exceptions. Every pair produced **exactly identical** scientific
reports, including labels, public GP and latent covariances/means, online events,
configuration, input hashes, output-weighting comparisons and pairwise warp results.
There are no private/deprecated HDP-GPC imports or private attribute reads in notebook
code; an AST audit runs before execution. No scientific baseline was regenerated.

| Notebook | First run | Second run | Exceeds 5 minutes? |
| --- | ---: | ---: | --- |
| 00 Quickstart | 8.53 s | 8.46 s | No |
| 01 Static | 8.50 s | 8.45 s | No |
| 02 Dynamic | 8.52 s | 8.59 s | No |
| 03 Online | 7.14 s | 7.39 s | No |
| 04 Offline | 8.91 s | 8.94 s | No |
| 05 Warping | 10.08 s | 9.17 s | No |
| 06 Multi-output, including comparison | 19.93 s | 25.91 s | No |
| 07 Fast online, including comparison | 11.58 s | 9.31 s | No |

Times include fresh host/worker kernel startup, data loading, configuration,
inference, figures, reports and shutdown with dependencies already installed.
They are illustrative local notebook runtimes, not an isolated performance study;
dependency installation for the separate bundle check overlapped some later runs.
They do not measure Colab provisioning or guarantee a Colab runtime. Inference-only
times are recorded separately in the JSON.

Executed notebooks and figures remain under `tutorials/.validation/final/` locally
and are uploaded as artifacts by the new tutorial CI workflow. Representative
assignment/GP and warp plots were visually inspected. Ordinary numerical warnings
remain visible. A Jupyter worker may report that its parent exited when the validator
shuts down a host kernel; this occurs after successful cells, not during inference.

The original notebook's raw-amplitude settings and private-state inspection were
not copied blindly. The two-pass budget and record/annotation length mismatch are
documented in [inventory.md](inventory.md). The pairwise warp plot uses real beats
and can be close to identity; it is not fabricated to make alignment appear stronger.

## Fresh bundle installation

The final isolation path was exercised outside the checkout in a new directory,
with `HDPGPC_TUTORIAL_FORCE_ISOLATED=1` (the same installation branch used on Colab).
uv provisioned **Python 3.11.16**, created an environment without system site
packages, and installed the bundled wheel, its dependencies and ipykernel.
The real record was loaded from that bundle. Quickstart then ran twice with
identical scientific reports. The cold run including installation took **100.73 s**;
the second run took **8.24 s**. Neither exceeded five minutes on this machine.
See [bundle_validation.json](bundle_validation.json). Network-dependent setup on
Colab can take longer; the timing table above deliberately excludes first installation.

To repeat this branch locally, extract `tutorials/bundles/hdpgpc-colab.zip` outside
the checkout, set `HDPGPC_TUTORIAL_FORCE_ISOLATED=1`, and run the validator with an
output directory inside that extracted folder. A prior pilot also verified standard
venv installation; the shipped setup uses uv to avoid relying on Colab ensurepip.

## Scope and remaining external verification

The notebooks use a Colab-compatible upload/setup path and a separate Python worker
to avoid changes to Colab's own scientific libraries. **Hosted Google Colab was not
executed from this session.** The Linux CPU-wheel path and the hosted upload widget
therefore still require a Colab smoke run. Local execution and an isolated Windows
bundle installation do not establish hosted service availability or cross-platform
bitwise equality. The GitHub CI workflow is configured but has not run remotely.

No inference module changed. Core package files in the built wheel match the
previously validated installed package. Packaging changes are limited to the new
optional `tutorials` dependency group and inclusion of notebooks/helpers in source
archives. The full record collection is still excluded from wheel/sdist artifacts;
only the separate Colab bundle explicitly includes record 100.

All **15 original characterization tests passed** again in 31.839 seconds after
the tutorial additions. The [source audit](source_audit.json) checks all 52 package
Python files against the preceding checkpoint and all 59 wheel package resources
against the previously validated installation. The original numerical baselines
and inference code are unchanged.
