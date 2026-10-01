> Historical 0.2 evidence. Source paths and compatibility instructions below describe the archived release, not 0.3. Use [the current architecture](architecture.md), [migration guide](migration.md), and [0.3 validation report](migration_0_3/report.md) for this release.

# Explicit inference engines

This is a software-architecture change. The source-based
[schedule pseudocode](inference_schedule_pseudocode.md) was written before any
code changes. Standard online, fast online, and offline remain distinct
algorithms. No posterior formula, numerical library, optimizer, tolerance,
convergence condition, or cluster acceptance policy was changed.

## Boundary and use

```python
from hdpgpc.inference import OnlineInference, OfflineInference

online = OnlineInference(model)
online.update(x, y, warp=False)
# Alternatively, select the existing fast algorithm explicitly:
online.update_fast(x, y, warp=False)

offline = OfflineInference(other_model)
offline.include_batch(xs, ys, it_limit=3, warp=False)
labels = offline.cluster_new_batch(xs, ys, learning=False)
```

The example illustrates separate choices; it is not a recommendation to switch
algorithms midway through training. Existing `model.include_sample(...)`,
`model.include_sample_fast(...)`, and `model.include_batch(...)` calls still work
with identical signatures, defaults, mutations and return values. In particular,
training methods still implicitly return `None` despite the historical docstrings.

`InferenceEngine` only binds a live owner and exposes its read-only `model`
property. Concrete engines have different entry points and no shared `run()`
algorithm. Constructing an engine does not initialize components, change model
state, copy tensors, populate caches, seed RNGs or import numerical libraries.
No engine is added to the model's serialized fields. Multiple bindings observe
the same state; they do not provide isolation or concurrent-update safety.

Public classes are exported by `hdpgpc.inference`. Private `_online` and `_offline`
functions contain the relocated schedule bodies. Their explicit first argument
is the **actual model**, still named `self` to retain the copied code. There is
no attribute proxy and no duplicate posterior store. Concrete methods pass their
owner to these functions; legacy model methods delegate to a concrete engine.
Internal calls continue through owner methods so existing overrides and instance
instrumentation remain effective. This deliberate compatibility dependency is
not a new independent model protocol.

| Location | Relocated methods |
| --- | --- |
| `inference._online` | `include_sample`, `include_sample_fast`, `_init_resp_cache_cold_start`, `_ensure_fast_cache_online`, `_update_q_lat_tail`, `_append_hard_step` |
| `inference._offline` | `include_batch`, `variational_local_terms_batch`, `estimate_q_first`, `estimate_q_all`, `cluster_new_batch`, `reload_model_from_labels` |
| `GPI_HDP` retained | State, constructors/factories, cluster lifecycle/reordering helpers, HMM recurrences, objective assembly, weighting and conversion helpers, compatibility entry points |

The engine calls existing GP emission, LDS, alignment and HDP operations through
their existing integration points. Emission `full_pass_weighted` retains its
own ascending active-sample loop, filter/smooth/update ordering and static/dynamic
branches. Mathematical operations already embedded in the moved schedule bodies
are relocated unchanged, not copied into a second implementation.

## Order-sensitive behavior retained

- Standard online evaluates the initial local terms and baseline objective twice
  for later samples. Birth/absorption proposals precede the committed weighted
  inclusion. Temporary GP installation, restoration, cluster/output loops,
  reordering and four theta/stick updates remain in their original positions.
- Fast online reuses hard history and patches latent-score tails. Its first
  observation appends history twice, and its HDP update does not refresh
  `trans_A` as standard online does. These differences were not unified.
- Offline refills before its end check; a one-column state performs two global
  updates before that check. Local reallocation/splitting and iterative
  re-estimation precede the outer two global updates. Proposal-local objectives
  and global objectives keep their distinct order and stopping tests.
- Offline `it_limit` and exact `M == max_models` checks precede q/history/cache
  refresh. One sample skips the final objective/history block. A printed `STOP`
  at iteration 14 is still only a print statement.
- Rejected proposals keep the original restores, references and occasional
  side effects. There is no transaction/rollback abstraction. GP/warp list trims,
  representative-index updates, normalized SNR and selected warped-data commits
  retain their original order, including the branch-specific differences.
- `cluster_new_batch` read-only classification, its learning branch, and label
  reload remain separate schedules. The latter retains its original pair-index
  construction. Unsupported online warping/multi-output and forced-reuse failures
  remain observable; they are not silently repaired.

See the pseudocode for each nested loop and the workspace
[implementation specification](../../docs/implementation_specification.md) for
the mathematical and numerical contracts.

## Validation and first-divergence gate

The original 15 characterization cases retain their original golden files and
floating-point tolerances (`rtol=1e-7`, `atol=1e-9`, exact discrete values).
They ran before changes, after introducing online, and after introducing offline.

`tests/inference_cases.py` adds pre-extraction traces of real model and GP
boundaries: entry arguments, exit values, HDP state, responsibilities, objectives,
cache fields, GP parameters and filtered/smoothed trajectories. Wrappers call
each original method once and return its actual result. Final snapshots include
transition probabilities, cluster state and alignment outputs/cache keys.
There are no mocked numerical routines in these comparisons.

The ten initial scenarios record **3,793 entry/exit events**. They cover standard
online dynamic/static, fast online, offline dynamic/static, two outputs with and
without SNR weighting, offline warping, and the two known online failures. The
compressed fixture was recorded before either engine extraction. Four additional
edge traces were recorded from the preserved pre-extraction package; capture
asserts that its model source hash matches the original fixture. These protect
one-sample batch exit, iteration-limit exit, label reload, and forced-reuse failure.
No existing reference was overwritten.

The supplemental scenarios add 530 events: 64 for one-sample batch, 272 for
iteration-limit exit, 84 for label reload and 110 for forced-reuse failure. The
total is **4,323 boundaries across 14 scenarios**. The static online fixture
commits cluster counts `[1, 1, 2, 2]` and assignments `[0, 0, 1, 1]`, protecting
both actual cluster creation and reuse in addition to proposal evaluation.

Each event is compared immediately, with exact numerical equality on the pinned
CPU environment. On a difference, `InferenceDivergence` identifies the first
event index, method, entry/exit phase and nested field path. It inherits directly
from `BaseException` so a legacy broad `except Exception` cannot swallow it.
The process must stop for investigation; regenerating a baseline is not recovery.
Tests also run three representative cases directly through public engines against
the same pre-edit traces, and protect signature/return forwarding, live state,
pickle compatibility and lightweight imports.

An AST audit compares all twelve relocated functions against the preserved source,
excluding only docstring indentation caused by dedenting. Executable statements,
arguments and defaults are identical. Every other statement in the main model
class is unchanged. This supplements, rather than replaces, numerical evidence
for less frequently exercised proposal branches.

Validation is CPU float64 with seed 42, deterministic Torch algorithms and one
numerical thread. It does not establish CUDA behavior, every possible convergence
trajectory, every classification-learning/warping combination or arbitrary
third-party monkeypatching of module globals. Numerical tolerances were not
loosened to accommodate this extraction.

## Reproduction

Run from the workspace with the `hdpgpc_env` Python interpreter:

```text
python -B -m unittest discover -s tests/characterization -t . -q
python -B -m unittest discover -s HDP-GPC/tests -p 'test_*.py' -q
python -B -m tests.characterization.verify_determinism --runs 2 --report NEW_REPORT.json
python -B HDP-GPC/tests/inference_cases.py
python -B HDP-GPC/tests/inference_cases.py --supplemental
```

`benchmarks/inference_schedule.py` times four real schedules without tracing in
the timed region. It warms each case, repeats deterministic runs, and records
final-state SHA-256 values with timings and source/environment metadata.
`--source PACKAGE_PARENT` runs a preserved reference package in a separate process.
Reports refuse overwrites.

## Validation results

| Stage | Result |
| --- | --- |
| Before production changes | 15/15 original characterization tests; 119 repository tests, one CUDA skip |
| Pre-edit trace baseline | 11/11 initial trace/inventory tests |
| Online engine introduced | 15/15 characterization tests; 11/11 exact trace/inventory tests |
| Offline engine introduced | 15/15 characterization tests; all original traces and direct-engine comparisons exact |
| Final repository suite | 141 tests: 140 passed, one CUDA skip; includes 21 inference tests and the new lightweight-import check |
| Fresh-process determinism | Two complete original-characterization captures, forward and reverse case order: byte-for-byte identical to all 15 original snapshots |
| Source audit | All 12 moved executable ASTs/signatures equal; every other model statement equal; 37 other Python files byte-identical |

The trace guard's own initial unit test mistakenly expected model construction to
emit a traced helper event. Construction emits none; the test now explicitly
invokes the factory under observation. No posterior mismatch occurred, and neither
production calculations nor frozen expected values were changed to fix that test.

The [source audit](../benchmarks/inference_source_audit.json) records method hashes
and the unchanged-file inventory. The numerical benchmark used five measured
runs after one warm-up per workload, sequentially in separate old/new processes.
Model construction and final-state serialization were outside the timed region;
the full inference calls, including existing printed output redirected to memory,
were inside it. Inputs are four online observations or six batch observations,
each only five points long, on CPU float64 with one numerical thread.

| Workload | Reference median (s) | Engine median (s) | Change |
| --- | ---: | ---: | ---: |
| Standard online | 1.307379 | 1.287824 | -1.50% |
| Fast online | 1.270744 | 1.263380 | -0.58% |
| Offline | 2.580706 | 2.577733 | -0.12% |
| Offline with warping | 2.707811 | 2.708507 | +0.03% |

All before/after final-state hashes match, and repeated states are deterministic
within each timing process. These small timing differences do not establish a
speedup; no numerical optimization was performed. Engine binding/delegation adds
Python call overhead but no persistent matrices or caches. Peak memory and GPU
runtime were not measured. Raw timings and metadata:
[before](../benchmarks/inference_before.json),
[after](../benchmarks/inference_after.json),
[comparison](../benchmarks/inference_timing_comparison.json).

The [fresh-process determinism report](../benchmarks/inference_characterization_determinism.json)
records zero differences in both complete captures. The implementation and all
original characterization baselines remain unchanged apart from the documented
schedule relocation and compatibility delegates.
