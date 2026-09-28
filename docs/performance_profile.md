> Historical 0.2 evidence. Source paths and compatibility instructions below describe the archived release, not 0.3. Use [the current architecture](architecture.md), [migration guide](migration.md), and [0.3 validation report](migration_0_3/report.md) for this release.

# Performance evidence and decisions

Structural baseline: `81fbcd2`. Measurements on 2026-09-26, Python 3.11,
PyTorch 2.9.0, CPU float64, seed 42, one numerical thread. Production code was
unchanged when these profiles were recorded. Full baseline validation: 193
repository tests (one CUDA skip), plus all 15 original characterization tests.

## Workloads and measurement limits

The deterministic generated waveforms need no dataset downloads. Online/dynamic,
offline/dynamic and offline/static use 12 observations on 24 points. Offline
multi-output uses eight observations, 16 points and two outputs. Offline runs
allow three outer iterations. Separate real scalar and batch warpers use 64
points, 250 Adam steps, and one or 16 waveforms respectively. Warping benchmarks
are component workloads, not a claim that the unsupported online warp path works.

Each isolated process warms its workload once, then records three uninstrumented
times. Separate reruns collect cProfile, RSS at 5 ms intervals, tracemalloc, and
PyTorch operator allocation/time data. Every final state hash agrees, including
across profiler modes. Construction is timed; result serialization is not.

Full-run PyTorch tracing was stopped when **profiler event storage** reached
approximately 8 GiB. The retained reports instead trace construction through the
first **three Adam steps**. Operator byte counts are allocation traffic in that
window, not whole-run peak memory. cProfile and both memory measurements cover
the whole workload. RSS is measured separately from tracemalloc. Warm allocator
reuse, sampling granularity, Python-only tracing and unmeasured CUDA are explicit
limits; zero RSS growth does not imply zero allocations.

Raw profiles, including per-function call counts, all timings, allocation sites
and environment versions: [performance_baseline_profiles](../benchmarks/performance_baseline_profiles).

| Workload | Median wall seconds | Warm RSS growth MiB | Traced Python peak MiB |
| --- | ---: | ---: | ---: |
| Online dynamic | 2.749 | 0.125 | 0.52 |
| Offline dynamic | 2.633 | 0.000 | 0.53 |
| Offline static | 2.637 | 0.156 | 0.51 |
| Offline multi-output | 5.169 | 0.020 | 0.58 |
| Scalar warp | 0.164 | 0.09 | 0.06 |
| Batch warp | 0.195 | 0.09 | 0.07 |

## Ranked bottlenecks

Times below are **inclusive cProfile seconds**, not the uninstrumented times
above. Nested costs overlap and must not be added. In particular, cluster
re-estimation contains GP fitting, and GP fitting contains factorization.

1. **GP hyperparameter optimization:** online `fit_torch` uses 4.136 of 4.430
   profiled seconds (93.4%); offline dynamic 4.106 seconds, and multi-output
   8.228 seconds. Two fits use 2,002 Adam steps; multi-output uses four fits and
   4,004 steps. Autograd alone uses 0.485 online self seconds. This is the largest
   runtime cost. It is not evidence that any of these steps can be omitted.
2. **Warp optimization within alignment workloads:** scalar 0.193 profiled
   seconds, batch 0.236. Backward propagation takes 0.060/0.072 self seconds.
   Scalar control expansion uses 0.028 seconds (14.5%); 251 calls rebuild the
   same control-grid lookup. All scalar interpolation totals 0.036 seconds.
3. **HDP optimization in online inference:** 40 `find_optimum_rhoOmega` calls
   use 0.062 seconds (1.4%). Offline uses six calls and approximately 0.015 s.
4. **State messages and emission/latent scoring:** online forward/backward
   messages use 0.020/0.014 seconds; coupled coefficients 0.009. Online latent
   error recomputation uses 0.013 seconds. These are measurable but small here.
5. **Cluster copies and Kalman filtering:** 15 online `gpmodel_deepcopy` calls
   use 0.027 seconds; 42 filters use 0.021 seconds. Offline dynamic uses seven
   copies (0.011 seconds) and 30 filters (0.012 seconds).

Memory ranking depends on the metric. Whole-workload traced Python peaks rank
multi-output first (0.58 MiB), followed by the three single-output model runs
(0.51-0.53 MiB), batch warp (0.07 MiB), then scalar warp (0.06 MiB). Retained
allocation sites in the online report include autograd/linear-operator wrappers
and emission tensor conversion. In the three-step tensor window, batch-warp
temporary elementwise products allocate 0.422 MiB, versus 0.087 MiB for online
GP fitting and 0.032 MiB for scalar warp. These are the largest allocating
operators in their respective windows. They do not establish a whole-run native
allocation ranking; the raw per-operator data and sampled RSS are retained rather
than inferring unsupported peak attribution.

## Independent investigation of requested areas

| Area | Measured evidence and source observation | Decision for this increment |
| --- | --- | --- |
| Repeated covariance construction | Online sklearn compound-kernel outer calls: 170, 0.015 s. `dynamics/_kalman.py:filter_posterior` constructs both basis and cross covariance even on equal grids. | Defer; small end-to-end share. Skipping the cross kernel needs tests for custom stateful kernels and GPU behavior. |
| Factorization reuse | GP fitting makes 2,002 Cholesky calls; linear-operator `cholesky` includes 0.526 s, native `cholesky_ex` only 0.025 s. Hyperparameters change at every Adam step. | No cross-step reuse: unchanged coordinates do not imply unchanged covariance. Existing warp-prior factorization cache already works; batch warp builds it once. |
| Unnecessary inverses | Online `inv_r`: two calls, 0.000286 s; offline full smoothing still contains explicit `torch.linalg.inv`, while pair smoothing already solves. | No inverse-to-solve change justified by these measurements. It could change rounding. |
| Allocation/copying | Online 1,516 `numpy.array` calls take 0.0022 s; 1,991 tensor-to-NumPy calls 0.0045 s. Cluster copies take 0.027 s. Warp lookup temporaries are recreated 251 times. | Target only scalar warp lookup temporaries in P01. Do not weaken model/result copy isolation. |
| CPU/GPU transfers | Online 1,303 `.cpu()` calls take 0.000742 s on CPU: these are not GPU transfers. Warp traces also call `.cpu()` each step. | GPU transfer impact unmeasured because CUDA is unavailable. No transfer optimization or GPU speedup claim. |
| Python loops | 2,002 GP optimizer and 250 warp iterations dominate. GPyTorch broadcasting dispatch uses 0.188 online self seconds, but native/autograd costs are intertwined. | Preserve loop/update order; no vectorization. P01 removes loop-invariant preparation only. |
| Repeated GP optimization | Two first-cluster/new-cluster fits, four in multi-output, with unchanged 1,001-step exits. `full_pass_weighted` includes those fits; it is not independent evidence of duplicate optimization. | Keep all fits and stopping criteria. Sharing across outputs/clusters would need scientific equivalence evidence. |
| Warp optimization | Scalar control expansion: 251 calls, 0.028 of 0.193 profiled seconds. Batch uses a different index-space interpolation path. | P01: reuse per-call scalar interpolation indices/weights only. Batch path unchanged. |
| Message passing | 47 online forward/backward calls, 0.020/0.014 s; offline nine, 0.0046/0.0032 s. Kalman filter, full smoother and pair smoother have different formulas. | No rewrite, order change or alternate Kalman library. |
| Recomputation across clusters | Online copies 15; offline seven, multi-output 14. Offline `estimate_q_all` takes 0.021 s; `estimate_q_first` 4.214 s includes fitting. | Do not share mutable GP trajectories or proposals; low measured return for invalidation/aliasing risk. |

## P01: scalar warp control interpolation preparation

This proposal passed the acceptance gate below. Its final measurements and limits
are in [performance_changes.md](performance_changes.md); no other candidate in
the investigation table was optimized in this increment.

**Current bottleneck:** `_expand_ctrl_to_T` recreates `linspace`, clamp/search
indices and interpolation weights for the unchanged stored basis on every
optimizer iteration and final read. Profile evidence is in the scalar-warp report.

**Change:** prepare indices and weights once inside each `compute_warp` call,
then keep the original gather/multiply/add operations for the changing controls.
Keep a fallback for coordinate tensors requiring gradients. Do not retain a
cross-observation cache or modify target interpolation, priors, loss, Adam or steps.

**Operation order/precision:** constant preparation moves before the optimization
loop; arithmetic involving controls must retain its order. Dtype/device remain
unchanged. No inverse/solve substitution, precision change or vectorization.

**Acceptance gate:** focused values/gradient tests, exact existing warp fixtures,
full repository and original characterization suites, frozen inference boundaries,
and alternating old/new component benchmarks. Reject on posterior/warp divergence.
Record actual numerical differences and memory/runtime results before committing.

See [performance_continuation.md](performance_continuation.md) for current status
and [performance_changes.md](performance_changes.md) for accepted changes.
