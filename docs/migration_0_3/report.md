# 0.3 final implementation migration

Status: native migration completed; additional validation stopped at the user's
explicit instruction on 2026-09-28. No further tests, benchmarks, inference runs
or notebook executions were started after that instruction. The release is
prepared locally; it has not been tagged, pushed or published.

## Implementation and removals

The separate release is `npdc-0.3.0`; original `npdc`, its ZIP and `HDP-GPC` are
unchanged. The new architecture owns the implementations. No new class inherits
from, wraps, instantiates or imports an old implementation class.

| Removed module/symbol | Native owner or disposition |
| --- | --- |
| `GPI_HDP.GPI_HDP` | typed `model._state.ModelState`; public `HDPGPC` and native engines |
| model cluster lifecycle/defaults/copying | `model._clusters` |
| model local responsibilities and HMM recurrences | `inference._messages` |
| model emission/LDS/HDP objective aggregation | `inference._scoring` |
| model SNR/output weighting | `emissions._weights` |
| model representation conversions | `model._conversion` |
| `GPI.IterativeGaussianProcess` | `emissions._prior.GaussianProcessPrior`, fitting in `_training` |
| `GPI_model.GPI_model` and alias shim | concrete `GaussianProcessEmission` in `emissions._gaussian_process` |
| `GPI_models_pytorch` | three active classes in `emissions._torch_models`; inactive warp classes retired |
| `amtgp_warping_system.Warping_system` and shim | concrete `alignment._amtgp.GaussianProcessWarp` |
| inactive `warping_system.Warping_system` | retired; never the supported current AMTGP path |
| `OptimizerRhoOmega` re-export shim | sole optimizer in `hdp._optimizer`, bnpy attribution retained |
| `get_data`, `util_plots` | private `datasets._research`, `diagnostics._plots`; no old import aliases |
| `GPI_test`, `extract_data`, five study scripts, two study notebooks | retired; maintained examples/tutorials replace public workflows |
| historical P02 executables loading old code through Git | retired; their recorded study evidence remains |

[removed_files.json](removed_files.json) is the exact deletion list. The
pre-edit inventory and every symbol/reference site remain in
[inventory.json](inventory.json). Math functions are directly bound from their
owners, not copied into a second orchestration class. The state still holds one
set of scientific arrays and caches. Internal HDP/alignment dispatch is used by
native schedules and contains no duplicated calculations.

Compatibility constructors/codecs (`from_legacy_args`, `from_legacy_kwargs`,
`to_legacy_kwargs`, `from_config`) and old model `include_sample/include_batch`
entry points are removed. Engines expose `update`, `update_fast`, `fit`,
`classify_batch` and `fit_labels`. The emission's internal observation update is
named `condition_observation` to avoid confusion with the retired model API.
Old class/pickle aliases and saved-model alignment fallback are removed.

## Evidence completed before the stop instruction

- Unchanged reference: 221 tests, 217 passed, four expected skips, 320.983 seconds.
- Reference tutorials: all eight ran twice; scientific reports repeated exactly.
  See [baseline_notebooks.json](baseline_notebooks.json).
- Component stage: 99 numerical tests passed; 57 boundary/math tests passed with
  one hardware skip. Numerical fixtures were not edited.
- Native source, with legacy files deleted: 221 tests, 217 passed, four expected
  skips, 303.146 seconds (`native-tests.log`). Skips are unavailable CUDA checks
  and the installed-artifact-only test.
- Four additional architecture tests passed: absent old modules/imports, native
  class/function ownership and byte-identical frozen fixture hashes.
- Exact reference traces cover 14 engine/configuration scenarios (4,323 boundaries)
  and 12 facade scenarios (4,129 boundaries), plus final scientific state.
- Source audit: 127 extracted function definitions have identical ASTs after
  excluding docstrings and applying explicit symbol renames. No numerical
  expressions were simplified. See [source_equivalence.json](source_equivalence.json).
- Wheel/sdist build, twine metadata and artifact-content checks passed.
- A new venv without system site packages successfully installed the unpacked
  source distribution and pinned scientific/tutorial dependencies. See
  [install-report.json](install-report.json). No post-install inference/test run
  was performed: the user stopped further testing immediately after installation.

Early errors were test-invocation/test-migration errors, not numerical differences:
wrong unittest discovery root, a private warp-prior import, an identity assertion,
and eight assertions expecting deliberately removed entry points. Corrected
native engine-contract tests passed; the final full source run is green.
Expected posterior numbers and numerical tolerances were never regenerated or
weakened. Historical pickle compatibility assertions were replaced intentionally
with native pickle/absent-interface checks; old pickle support is a declared break.

## Numerical and performance comparison

All six benchmark scientific-state SHA-256 hashes match exactly, covering cluster
state, responsibilities, sticks/transitions, GP/LDS state, objectives, warps and
caches available in each workload. Uninstrumented timing used a warmup and three
measured repeats; RSS sampling and Python allocation tracing ran separately.

| Workload | Reference (s) | Native (s) | Change | Peak RSS change |
| --- | ---: | ---: | ---: | ---: |
| offline_dynamic | 2.7486 | 2.7459 | -0.10% | +0.32 MiB |
| offline_multi | 5.1772 | 5.2049 | +0.54% | -2.82 MiB |
| offline_static | 2.7177 | 2.7541 | +1.34% | -1.23 MiB |
| online_dynamic | 2.8387 | 2.7984 | -1.42% | +0.59 MiB |
| warp_batch | 0.1932 | 0.1904 | -1.48% | -0.97 MiB |
| warp_scalar | 0.1576 | 0.1634 | +3.66% | -0.58 MiB |

No workload changed runtime by more than 5%. These small differences are
measurements, not an optimization claim or a statistical guarantee. Absolute peak
RSS stayed within 3 MB; Python allocation peak differences were below 9 KB.
Memory includes interpreter/library allocator history; it is not a per-array
live-memory bound. Complete measurements and hashes are in
[benchmark_comparison.json](benchmark_comparison.json) and the two benchmark folders.

## Supported scope and limitations

Static and dynamic LDS, online/fast/offline, alignment on/off, single/multiple
outputs, GP exact/projected/reduced modes, current ML/Bayesian learning policies,
HDP concentration presets and supplied configurations retain their existing
implementations. Existing tests protect shared parameter objects, GP fitting
sharing, output weighting, growth/pruning and estimation limits. This does not
invent fixed/shared-C policies or representation transforms that never existed.
Known online warp, online multi-output and forced-reuse failures remain frozen.
No GPU or additional Python/OS CI matrix execution is claimed.

Not run after the stop instruction: installed-package regression/console/example
verification, additional determinism captures, final 0.3 notebook executions,
new performance checks or artifact validation tests. The successful reference
notebooks ran on 0.2; they are not presented as 0.3 execution evidence. Tutorial
worker setup now requires version 0.3.0 to prevent reusing a stale installation;
that infrastructure change has not received a new notebook execution.

## Minimal package and compatibility breaks

The wheel contains native runtime modules, tiny deterministic example recipes and
the reference result, metadata and license notices. It does not contain tests,
benchmarks, datasets, study scripts, notebooks or validation output. Large old
datasets/results/environment snapshots were removed from this release copy.
Only record 100 and its labels remain outside the installed package for the
optional Colab bundle (about 3.3 MB); other records can be fetched explicitly by
the tutorial loader. The original full data remain in the unchanged 0.2 folder.
No additional abstraction hierarchy or new heavy dependency was introduced.

0.3.0 is the chosen breaking pre-1 minor release; 1.0 stability is not claimed.
Old imports, flat constructors, pickle paths, inactive alternate warping and
path-dependent study programs intentionally break. Replace serialized alignment
policy `legacy` with `gp`; its scientific default is unchanged. See
[the migration guide](../migration.md) for explicit mappings and advanced engines.

[checkpoint.md](checkpoint.md) records completed stages and the deferred commands
for a future, explicitly authorized validation pass. Do not reapply the one-shot
migration scripts. Do not mistake historical 0.2 documents for the new API.

## Source distribution cleanup

Historical benchmark scripts/reports are retained in the baseline Git commit and
the untouched 0.2 folder. This release keeps only the workload and profiler modules
needed to reproduce current measurements, plus the lightweight performance tool.
The wheel excludes all of these development assets.
