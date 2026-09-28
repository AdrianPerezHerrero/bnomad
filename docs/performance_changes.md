> Historical 0.2 evidence. Source paths and compatibility instructions below describe the archived release, not 0.3. Use [the current architecture](architecture.md), [migration guide](migration.md), and [0.3 validation report](migration_0_3/report.md) for this release.

# Performance change ledger

See [profiling report](performance_profile.md) for ranked measurements and
[continuation log](performance_continuation.md) for commands and recovery status.
Performance changes start after structural baseline `81fbcd2` and evidence commit
`8d3acc1`. Each accepted change has a separate commit.

## P02: reuse temporary forward/backward message rows

Status: accepted in dedicated commit `7bda4f5`, after packaging checkpoint `49d59a3`.
Recovery context is in [performance_continuation.md](performance_continuation.md).
Full study: [performance_stream_study.md](performance_stream_study.md).

**Evidence:** new post-startup online workloads show forward/backward messages
at 25%/33% of cProfile time for 64/128 total observations, excluding twelve fitted
startup observations. There are no further GP fits in those tails. Existing P01
studies were reused, not rerun.

**Change:** hold a local view of each fresh message row for assignment, summation
and normalization. Original matmul/multiply/sum/division ordering, transition
floors, precision/device, loops and schedules remain intact. No persistent cache
or model field is added. The existing cached-forward branch is untouched.
Gradient-bearing inputs and tensor subclasses use the original indexed code,
preserving its autograd behavior; native nongradient buffer version-counter
increments differ because redundant indexed write-back is removed.

**Validation:** all 221 tests ran successfully (217 passed, three CUDA skips and
one installed-only gate skipped). Existing 4,323-boundary inference traces and
the public/configuration checks remain exact. No fixture/tolerance changed.
Five focused tests cover float32/64, short/long recurrences, cluster counts,
noncontiguous/nonfinite inputs and the original gradient/subclass fallback.
Every paired/profiler/memory final state matches pre-change hashes exactly.

**Performance:** seven alternating pairs after warmup: 64-observation tail
1.315223 -> 1.259739 seconds (-4.22%); 128-observation tail 3.853518 -> 3.640474
seconds (-5.53%). The fixed operator probe removes 765 selects and 255 copy calls
while preserving all matmul/sum/division counts. Numerical difference: zero.

**Memory/limits:** Python peak changes +16,999 bytes for 64 observations and
-4,917 for 128, with no consistent peak-memory reduction. No native/GPU memory
or GPU speedup claim. This accelerates post-startup messages, not initial GP
optimization. Measurements, complete validation and limits are in the study.

## P01: prepare scalar control interpolation once per warp call

Status: accepted in separate commit `7d259c4`
(`perf(alignment): prepare scalar control interpolation once per warp call`).
The [continuation log](performance_continuation.md) records the recovery checkpoint.

**Current bottleneck and evidence:** scalar `_expand_ctrl_to_T` rebuilt identical
coordinate lookups 251 times in a 250-step warp. cProfile measured 0.0280 s of
0.1929 s in `compute_warp` (14.5%). Raw baseline:
`benchmarks/performance_baseline_profiles/warp_scalar.json`.

**Change:** copy coordinate-only preparation from the existing interpolation
helper into `alignment/_control_interpolation.py`, prepare it once per scalar
`compute_warp`, and retain the original control gather/multiply/add expressions.
The lookup is local to the call. There is no persistent cache, no new model state,
no cross-cluster reuse, and no batch-warper change. Original private helper methods
remain available. Coordinate-gradient or customized-helper paths use the original
implementation. Zero training iterations use the original final interpolation.

**Operation order and precision:** fixed-coordinate computations move before the
loop. Operations depending on optimized controls, target interpolation, loss,
backpropagation, Adam configuration, 250 benchmark steps, warm starts, final
likelihoods and dtype/device remain unchanged. No algebraic replacement of
inverses, precision reduction or vectorization is introduced.

**Focused checks:** `tests/test_control_interpolation.py` compares old/new values
and control gradients exactly for float32/float64 on uniform/irregular grids;
checks one preparation per call including changed grids; covers coordinate
gradients, zero steps and custom helper fallback. CUDA comparison is skipped on
this machine. The production warper continues using its original float64 policy.

**Numerical validation:** 198 repository tests ran (196 passed, two CUDA skips),
plus all 15 original characterization tests. Existing 4,323-boundary inference
traces and the public API subset remain exact. All 12 alignment cases match the
frozen JSON values exactly. All paired benchmark outputs, loss traces and
warm-start states match exactly: maximum absolute difference **0**. All six
representative workload state hashes agree with the pre-change reports. No
fixture or numerical tolerance changed. Full results:
[validation](../benchmarks/performance_p01_validation.json),
[paired timings](../benchmarks/performance_p01_paired.json),
[comparison](../benchmarks/performance_p01_comparison.json).

An initially strict in-memory fixture comparison reported four tuple/list type
differences in `model_cached` cache keys. Loading the old implementation reproduced
the same differences, with no optimization delta. JSON stores those tuples as
lists. Comparing the fixture's JSON representation resolves the mismatch without
rounding any numbers or changing tolerances. The diagnostic evidence remains in
[alignment audit](../benchmarks/performance_p01_alignment_exact.json).

**Paired performance:** one warmup and 11 measured repetitions per implementation,
alternating order in the same process, using baseline source from `81fbcd2`.

| Workload | Before median (s) | After median (s) | Runtime change |
| --- | ---: | ---: | ---: |
| Scalar, 64 points, 250 steps | 0.164394 | 0.149537 | -9.04% |
| Scalar, 64 points, two recursive 250-step calls | 0.340764 | 0.322389 | -5.39% |
| Scalar, 512 points, 250 steps | 0.173363 | 0.158796 | -8.40% |
| Unchanged batch control, 16 x 64 points | 0.184363 | 0.184563 | +0.11% |

The greater-than-5% scalar changes are explained by the measured work removal:
whole-call `searchsorted` calls fall from 502 to 252, and `linspace` calls from
252 to two (one input grid, one control grid). Control preparation/expansion
profile time falls from 0.028022 s to 0.010751 s including the one preparation.
Adam still takes 250 steps. The independent unpaired scalar measurements show
-3.02%, demonstrating timing drift; paired comparisons are the preferred effect
estimate, not a universal speed guarantee.

**Memory:** the three-step tensor trace records indexing allocation traffic
falling from 12,288 to 10,240 bytes, subtraction from 30,208 to 25,600, division
from 19,696 to 18,672, and search indices from 3,072 to 2,048. Whole-call traced
Python peak increases slightly from 64,995 to 65,587 bytes (+592); sampled warm
RSS growth is unchanged at 98,304 bytes. This reduces repeated allocation work,
not measured peak memory. The per-call lookup has no lifetime beyond the call.
Raw after profile: [warp_scalar.json](../benchmarks/performance_p01_profiles/warp_scalar.json).

**Other workloads:** sequential after/before medians change by -0.70% online,
+1.05% offline dynamic, -1.34% static, and -0.48% multi-output. These paths do not
call the modified scalar warper, so their small differences are not attributed to
P01. The older unpaired batch median appears 5.53% faster; investigation is the
already-completed alternating unchanged-batch control above (+0.11%). Both old
and new implementations run at approximately 0.184 s in that paired study,
versus 0.195 s in the earlier process: temporal/process variation, not a batch
optimization. No additional study was needed or run for that discrepancy.

**Limits:** CUDA equivalence/performance remains untested; the added CUDA test is
skipped here. This change does not accelerate GP training, batch warping, or an
offline workload that exclusively uses batch alignment. Revert P01 alone to
restore the previous scalar control preparation.
