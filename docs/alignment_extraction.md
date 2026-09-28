> Historical 0.2 evidence. Source paths and compatibility instructions below describe the archived release, not 0.3. Use [the current architecture](architecture.md), [migration guide](migration.md), and [0.3 validation report](migration_0_3/report.md) for this release.

# Optional temporal alignment: existing path and extraction

## Before-edit implementation inventory

This task is category D, software architecture. The current main model imports
`amtgp_warping_system.Warping_system`. The older `warping_system.py` implements a
different GP alignment algorithm and remains available at its original path;
this extraction does not select it or mix its formulas with the active engine.

### Initialization and parameterization

GPI_HDP constructs one warper for each output/cluster, including otherwise empty
candidate models. New-cluster factories construct more warpers using stored
defaults. The active warper stores a float64 basis on CPU or available CUDA,
`n_ctrl=max(4,min(n_ctrl,T))` (default 8), learning rate .05, smooth penalty 200,
amplitude penalty .001, recursion flag and a fixed `WarpPriorAMTGP` scorer.
Mode is stored but does not set a training budget or change this optimizer.

Unconstrained controls expand to T locations, then `softplus(u)+1e-6`, cumulative
sum, endpoint normalization with denominator `+1e-12`, and affine scaling produce
the monotone grid g. Returned `x_warp` is **g-x**, not g. Queries are clamped to
the input domain; linear interpolation uses searchsorted and denominator `+1e-12`.
Scalar controls interpolate over stored physical coordinates; batch controls use
`F.interpolate(...,align_corners=True)` in index space. Those are distinct on
irregular grids. Zero controls need not mean identity on an irregular grid.

### Objective, optimizer and constraints

The scalar training loss is `.5*SSE/(n+1e-12) + lambda_s*sum((D2*w)^2) +
lambda_a*sum(w^2)`. It compares the target sampled at g against the template.
Batch uses the same per-sample quantities and a weighted mean divided by
`sum(weights)+1e-12`; negative supplied weights are clamped to zero. Controls are
independent across batch members but share optimizer and weighted objective.

Adam optimizes only controls, with lr .05 and otherwise library defaults. Each
call creates a fresh optimizer. Exactly `int(train_iter)` steps run; there is no
convergence test/early stop. Defaults are 50 steps, while scalar main-model
strategies and the assigned-state batch path request 250. No steps are removed.
Both penalties remain in the optimization objective even when `bayesian=False`;
that flag affects the scalar returned score, not its fitted controls. Traces
record the loss computed before each Adam step; returned warps are recomputed
after the final step under `no_grad` and detached. Inputs are coerced to the
warper's float64/device, and controls alone receive optimizer updates.

Tuple/list `(rho,omega)` or dict theta scales penalty weights by the corresponding
squared scale plus `1e-12`. A scalar theta (as normally supplied by the emission
lengthscale) falls back to base penalties. Scalar `_safe_noise` uses the first
element of a vector and clamps to bounds; batch uses the vector mean and clamps
when noise was supplied. Their None/scalar handling is not unified or repaired.

### Returned scores and fixed GP prior

Scalar final score is Gaussian residual log likelihood, including its dimension/
noise normalization, and (when bayesian) negative smooth/amplitude penalties.
Batch final score is the displacement GP prior log density, independent of the
bayesian flag. GPI_HDP adds a second score from the last warper's prior in both
paths; it is not removed as a suspected duplicate. No shared score abstraction
replaces these distinct quantities.

The prior covariance is RBF with optional coordinate normalization, variance
omega squared, bounded `noise_warp` as variance and jitter `1e-6`. rho/omega clamp
to at least `1e-12`. Cholesky scoring includes quadratic form, log determinant
and dimension constant. Batch always assigns the scorer's theta; scalar assigns
theta only when rebuilding the scorer after a length change. Same-length changed
coordinates do not replace the scalar object's stored basis.

### Sequential memory and caches

Recursive scalar calls clone/detach the previous controls if their count matches.
Batch repeats the previous mean control vector for all members and saves their
new mean after fitting. Nonrecursive calls start at zero. Optimizer moments are
never reused. `update_warp` only stores a detached last displacement when recursive;
`reset` clears sequential memory but not the prior factorization cache.

Prior cache keys contain device, dtype, point count, rho/omega, noise, endpoints
and normalization flag, but not interior coordinates or jitter. Reusing a key
can therefore reuse a stale factorization; that behavior is preserved.

The main cached batch path stores complete all-observation results under
`(output, representative_index)`. Cluster ID, data, theta, noise, grid and training
budget are not part of that key; shape invalidation is commented out. Its
`reset_warp_cache` method clears three older cache fields, not `_warp_cache_full`.
No cache keys, lifetime, storage or invalidation are changed here.

### Assignments and GP emissions

`compute_warp_y` builds one candidate waveform/offset/score per cluster. Templates
come from emission observation predictions; noise is the covariance diagonal,
halved for static models. Forced, standard, greedy and greedy-bound strategies
retain their exact candidate ordering, score propagation and stopping formulas.
Greedy pre-ranking uses output-zero emissions. The empty `liks[:-1]` reduction
that breaks a currently characterized single-cluster online warp remains intact.

Candidate strategies use these exact gates:

- Forced alignment fits only the requested nonempty cluster, using `observe`
  when a time index is supplied and `observe_last` otherwise. Standard alignment
  visits all clusters in list order. Empty candidates retain identity waveforms
  and add `max(liks[:-1])` to their baseline score; the final candidate also gets
  that addition after the loop.
- Greedy and greedy-bound rank `q_C` descending, with `q_C` obtained from
  output-zero emissions through `estimate_new` at the last stored input grid.
  Greedy-bound stops after rank index 3, i.e. at most four visited candidates.
- For a nonempty greedy candidate with a following rank and `i<8`, the test is
  `(q_C[m] + .5*liks[m] - q_C[next]) / (q_C[m] - q_C[next]) >
  .3/log(max(model.N,1)+1)`, or `i==5`. There is no zero-denominator guard.
  On stopping, unvisited candidates inherit the maximum visited score as an
  addition and the last candidate receives the separate final addition.
  If there is no following rank or `i>=8`, that propagation occurs directly.
  Empty greedy candidates take their separate score-addition branch.

These are candidate-search stopping rules, separate from Adam's fixed budget.
The assigned-state scalar helper searches membership lists and defaults to
cluster zero; if multiple lists contain an index, the last match wins. Its batch
counterpart skips empty clusters and uses the first member's grid for the whole
cluster without validating common grids. Candidate batch helpers clamp a missing
cluster/representative index to the last available entry. Output loops and these
fallbacks remain in the copied code.

Uncached batch alignment visits every output/cluster and uses that cluster's
representative waveform, GP lengthscale and mean diagonal Sigma. Cached batch
alignment instead uses `kernel_def` and `sqrt(ini_sigma_def)` and reuses results
by representative. Both chunk observations in batches of 128 by default.
Responsibilities supply the number/order of competing clusters; these paths do
not pass responsibilities as optimizer weights. Argmax responsibility selects a
candidate warped waveform later. Warp scores feed emission scores/objectives;
GP fitting consumes the selected waveform. These schedules remain unchanged.

### Existing disabled behavior

Online `with_warp=False` appends raw waveforms to candidate histories without
appending the enabled path's `y_w`, `x_w` or `liks` history entries. The first enabled
online sample is also not optimized, but does append those histories. Standard
and fast first-sample zero allocations differ in device spelling and stay distinct.

Cached offline `warp=False` expands raw waveforms by cluster, returns an expanded
**absolute input grid** for x_w, and zero scores. It creates the full-cache dict
but stores no entries. This odd absolute-grid contract must remain compatible;
an identity component must not silently replace it with offsets in that API.

## Planned boundary

Add stateless model-level `GPAlignment` and `NoAlignment` policies, selected by
the new trailing `GPI_HDP(...,alignment=...)` argument. Default None keeps GP
alignment and the existing per-call flags; `alignment=NoAlignment()` disables
alignment for the model even when a call requests it. A compatibility lookup
supplies the default for older saved models without this field.

Move the active warper intact to `alignment._amtgp`, keeping old class identities,
names, imports and pickle paths through its legacy shim. Move candidate/batch/
assigned-state logic and online preparation/update gates into private alignment
integration helpers. The model keeps simple delegates and inference schedules.
No optimizer, objective, parameterization or scoring calculation changes.

NoAlignment implements identity results and zero alignment scores without GP
prior construction or Adam calls. Lightweight identity objects retain the
per-cluster list shape needed by the existing cluster lifecycle. The established
disabled online history and cached offline grid/view semantics remain intact.
There is no new optimizer hierarchy or general-purpose registration framework.

Before/after benchmarks cover scalar/batch fitting, recursion, cold and reused
model caches. Any median runtime change above 5% triggers an investigation before
further extraction proceeds; measurements and the investigation will be recorded.

## Implemented boundary

```python
from hdpgpc.model import GPI_HDP
from hdpgpc.alignment import NoAlignment, GPAlignment, GaussianProcessWarp

model = GPI_HDP(x_basis, alignment=NoAlignment())
# Or retain the existing behavior and per-call flags:
model = GPI_HDP(x_basis, alignment=GPAlignment())
# Omitting alignment has exactly the same policy as GPAlignment().
```

Existing constructor options can still be supplied. NoAlignment takes
precedence over `include_sample(...,with_warp=True)`, its fast variant,
`include_batch(...,warp=True)` and classification/reload requests. Set the policy
at construction; switching a trained model's policy in place is not an added API.

The smallest useful boundary has two levels:

- `GPAlignment` / `NoAlignment` are stateless model policies. They choose warper
  creation, online preparation/update, candidate and assigned-state alignment,
  batch alignment and cache-reset delegation. Their private helpers accept the
  existing owner rather than copying its intertwined histories and caches.
- `GaussianProcessWarp` is an exact alias of the existing per-cluster
  `Warping_system`; `compute_warp`, `compute_warp_batch`, `update_warp` and `reset`
  retain their signatures. Controls, sequential memory and prior factors still
  belong to those individual objects. No extra wrapper enters Adam iterations.

`alignment/_amtgp.py` contains the unchanged engine. `_integration.py` contains
the seven moved model methods plus the original online preparation/update
blocks. `_components.py` selects the policy, and `_identity.py` supplies identity
objects, neutral scores and direct helper results. `amtgp_warping_system.py` is
a compatibility shim retaining imported names and the three original class pickle
paths. Existing model methods remain delegates, so experiment imports/calls work.
Saved owners without an `alignment` attribute use the original GP policy.

NoAlignment creates neither a warp GP prior nor an Adam optimizer. Direct identity
calls preserve target tensor storage/gradients and return zero displacement,
zero scores and empty loss traces. Its cached model helper deliberately retains
the old disabled branch's expanded raw waveform and **absolute grid** views.
That distinction is tested and is not normalized into a new return convention.
Existing inference convergence checks on `warp` remain at their original schedule
locations; the effective flag is selected once at the public entry point.

Owner fields (`wp_sys`, `x_w`, `y_w`, `liks`, `liks_w`, `_warp_cache_full` and the
three older caches) retain their ownership and lifetime. Emission/HDP/LDS update
ordering, cluster lifecycle and output weighting are unchanged. This is an
explicit state-coupled integration boundary, not a new general plugin framework.

## Numerical and regression evidence

Validation used Python 3.11, NumPy 1.26.4, SciPy 1.14.1, PyTorch 2.9.0+cpu and
GPyTorch 1.13. Fixtures fix Python/NumPy/Torch seeds to 42, use float64 and one
numerical thread, and enable deterministic Torch algorithms.

- Before extraction: all 15 original characterization cases passed; the repository
  suite had 64 passing tests and one CUDA-only skip.
- After moving the engine, and again after moving integration, all 15 original
  characterization cases passed against unchanged baselines.
- Final repository suite: **86 passed, one CUDA-only skip** (87 tests).
  The 21 alignment tests include twelve pre-extraction snapshot cases plus an
  inventory check, six component-contract tests and two inference tests with
  static/dynamic/fast/multi-output subcases. A separate fresh-process structural
  test verifies that importing policies does not load the active engine/model
  or change Torch's default dtype.
- The twelve snapshots protect scalar/batch offsets, warped waveforms, scores,
  all objective traces, recursive controls, prior theta/factors, irregular grids,
  length-changing grids, negative weights, nonrecursive/non-Bayesian settings,
  stale prior/model cache reuse, and standard/forced cluster candidates.
- Component tests additionally protect monotonicity, iteration counts, detached
  outputs, reset scope, legacy class/instance pickle resolution and identity
  storage/gradient semantics. A constructor patched to raise proves that the
  disabled component does not instantiate an active warper, including factories.
- Constructor-level disabling matches original disabled inference snapshots for
  online dynamic/static/fast and offline dynamic/static/multi-output SNR cases,
  including internal state, inference traces and classification outputs.
- Both fresh full characterization captures (forward/reverse order) matched the
  original snapshots **byte for byte**. All twelve alignment snapshots also
  matched exactly in two fresh processes in opposite orders. Reports:
  [full inference](../benchmarks/alignment_characterization_determinism.json),
  [alignment component](../benchmarks/alignment_component_determinism.json).
- A syntax-tree audit against an isolated pre-task package confirmed the complete
  warp engine is identical after removing the added module docstring. All seven
  moved model-method bodies (excluding docstring indentation), both online
  preparation blocks and selected-warper update gates are also AST-identical.
  The [source audit](../benchmarks/alignment_source_audit.json) also verifies
  that the preserved pre-task sources match the frozen fixture's source hashes.
- An offline wheel built in a temporary source copy included all seven alignment
  modules. Imports and public/legacy class pickle identity passed from the
  unpacked wheel; no build artifacts or dependency installations were added.

Normal alignment regression tests require identical structure/shapes/dtypes and
exact discrete/scalar metadata. Floating tensor snapshots use `rtol=1e-7`,
`atol=1e-9`, matching the existing component comparison policy: this permits small
CPU numerical-library roundoff while detecting changes to the much larger warp
and score values. Independent exact comparisons above establish that this
extraction needed none of that tolerance in the measured environment. Identity
and storage assertions use exact comparison. No expected values were regenerated.

Run from the workspace containing this repository:

```text
python -B -m unittest discover -s HDP-GPC/tests -p test_alignment.py -v
python -B -m unittest discover -s HDP-GPC/tests -p "test_*.py" -q
python -B -m unittest discover -s tests/characterization -t . -q
python -B -m tests.characterization.verify_determinism --runs 2 --report NEW_REPORT.json
```

## Runtime measurements and the 5% investigations

`benchmarks/benchmark_alignment.py` uses real optimization: 64 locations,
16 batch members and 250 Adam steps for scalar/batch, with recursion on/off.
Model-cache tests use 32 locations, eight observations, two clusters, 50 steps
and chunks of four; cache-hit samples average 1,000 calls. Each workload warms
up before measurement. Benchmarks ran sequentially with no concurrent numerical
tests. Original/final primary runs have eleven samples per workload. These small
CPU fixtures do not establish GPU or large-dataset performance.

| Workload | Original median ms | Final median ms | Primary change |
|---|---:|---:|---:|
| Scalar, zero start | 161.799 | 162.442 | +0.40% |
| Batch, zero start | 184.100 | 183.134 | -0.52% |
| Scalar, recursive | 161.415 | 170.206 | **+5.45%** |
| Batch, recursive | 193.253 | 184.116 | -4.73% |
| Model cache miss | 139.217 | 141.815 | +1.87% |
| Model cache hit | 0.034520 | 0.035464 | +2.73% |

Raw primary reports are [before](../benchmarks/alignment_before.json),
[engine relocation](../benchmarks/alignment_engine_after.json) and
[final](../benchmarks/alignment_after.json). Two threshold crossings were
investigated and reported to the user **before proceeding**:

1. Engine-only relocation initially measured a **5.92% speedup** for recursive
   batch fitting. Original samples ranged from 181 to 298 ms, versus 180 to
   191 ms after relocation. Alternating old/new/new/old fresh-process runs, with
   seven samples each, produced pooled changes from -0.41% to +0.88%; recursive
   batch was +0.30%. The preserved original implementation itself no longer
   reproduced the unusually high initial median.
2. Final integration's **5.45% recursive scalar slowdown** triggered the same
   alternating experiment against the preserved original package. Pooling all
   fourteen old and fourteen new samples per workload gave the following result.

| Workload | Paired old median ms | Paired new median ms | Change |
|---|---:|---:|---:|
| Scalar, zero start | 166.188 | 164.756 | -0.86% |
| Batch, zero start | 186.978 | 184.983 | -1.07% |
| Scalar, recursive | 165.994 | 164.116 | -1.13% |
| Batch, recursive | 186.558 | 187.077 | +0.28% |
| Model cache miss | 143.363 | 140.066 | -2.30% |
| Model cache hit | 0.034835 | 0.035151 | +0.91% |

All investigation files are retained as
`benchmarks/alignment_{engine,final}_investigation_{1_old,2_new,3_new,4_old}.json`.
The benchmark supports `HDPGPC_ALIGNMENT_REFERENCE_SOURCE` to select an isolated
original source package while keeping the fixture and timing code identical.
`--compare` reports **either sign** of change over 5%; it does not silently discard
the result or automatically increase tolerances.

Both threshold crossings failed to reproduce in alternating comparisons.
Unchanged engine syntax/iteration counts and overlapping timings support timing
variability as the explanation, rather than a systematic algorithmic change.
The specific OS scheduling, frequency or background-load cause was not measured.
The model cache-hit path adds policy dispatch, so some microsecond-scale overhead
is plausible; the paired +0.91% result alone cannot separate it from noise.
No optimization or iteration reduction was made in response to either result.

## Limits deliberately retained

CUDA was unavailable. No GPU equivalence, large-dataset memory result or broad
historic full-model pickle guarantee is claimed. A warmed active-warper instance
round trip and a missing-owner-field compatibility test pass. The different
legacy `warping_system.py` implementation is unchanged and was not substituted.

The known online-warp empty reduction and online multi-output failures remain
characterized, not fixed. The direct successful candidate and batch fixtures do
not establish every greedy-bound, learning-classification or label-reload branch.
Cached stale values, scalar/batch score differences and warm-start differences
remain part of the executable specification. General module-global monkeypatches
on the legacy shim no longer rebind implementation globals; class identity,
ordinary imports and the tested serialization paths remain supported.
