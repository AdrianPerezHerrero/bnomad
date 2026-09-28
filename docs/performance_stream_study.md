> Historical 0.2 evidence. Source paths and compatibility instructions below describe the archived release, not 0.3. Use [the current architecture](architecture.md), [migration guide](migration.md), and [0.3 validation report](migration_0_3/report.md) for this release.

# Post-startup online study and P02 optimization

This is a new workload study after the completed P01 investigation, not a rerun
of its scalar-warp benchmarks. Packaging checkpoint: `49d59a3`. The packaging
documents were read first; its clean-install validation passes all 216 tests
(two CUDA skips) and three exact original-characterization captures. Torch 2.9.0
is retained because the 2.9.1 Windows wheel changed last-bit reference results.

## Measured post-startup bottlenecks

The original twelve-sample, 24-point dynamic online workload initializes each run.
Timing/profiling then starts at observation 13. The continuation uses the same
deterministic waveform generator with 64 or 128 total observations. No fitting,
inference iteration or update is disabled; initial fitting is merely outside
the measured window. Both runs end with two allocated clusters and have no
additional `fit_torch` call in their measured tails.

One warmup and three uninstrumented repetitions precede separate cProfile and
Python-memory runs. Complete model snapshots match exactly across all six runs
for each size. CPU float64, seed 42, one thread, reference scientific versions.

| Total observations | Measured tail | Median seconds | Python traced peak bytes |
| --- | --- | ---: | ---: |
| 64 | 52 observations | 1.329458 | 264,662 |
| 128 | 116 observations | 3.877203 | 465,814 |

Memory covers allocations during the tail, excluding the existing fitted prefix;
it is not whole-process/native/GPU peak memory. Timing and profiling runs are
separate, so inclusive profiler times below must not be compared as wall timings
or added when nested.

| Function | 64: calls / cumulative seconds | 128: calls / cumulative seconds |
| --- | ---: | ---: |
| Entire tail | 1 / 1.59337 | 1 / 4.55507 |
| Forward messages | 208 / 0.23310 | 464 / 0.85700 |
| Backward messages | 208 / 0.17066 | 464 / 0.64115 |
| Latent-score recomputation | 156 / 0.21142 | 348 / 0.83968 |
| Individual latent score | 2,002 / 0.19600 | 8,178 / 0.78455 |
| HDP optimization | 208 / 0.29298 | 464 / 0.62841 |
| Emission stabilized Cholesky helper | 2,210 / 0.09534 | 8,642 / 0.35715 |

Forward/backward messages account for about 25%/33% of the respective profiled
tails. Their self time includes Python loops and tensor indexing/assignment.
Latent history scoring grows faster than observation count. Its changing smoothed
states and per-time LDS covariances prevent assuming that old scores/factors can
be reused. HDP optimization keeps all original tolerances and restarts. No
inverse replacement, cross-cluster sharing or GP step reduction is proposed.

Raw reusable evidence:
`benchmarks/performance_stream_tail_64.json`,
`benchmarks/performance_stream_tail_128.json`.
Reproduction: `python -B benchmarks/profile_stream_tail.py --samples 128 --output NEW.json`.
Do not rerun these completed studies just to resume work.

## P02 proposal: temporary message-row views

`GPI_HDP.forward` and `backward` repeatedly index the same freshly allocated row
to assign, sum and normalize it. The candidate holds that row view locally,
copies the original expression into it, then applies the original sum/division.
The already-cached forward-prefix branch is unchanged. No persistent cache or
new model field is introduced.

Arithmetic operations, operand order, transition floors, excluded final backward
normalization column, iteration order, dtype/device and update schedules remain
unchanged. Indexed assignment becomes `row.copy_`; redundant indexed write-back
after in-place division disappears. This changes internal buffer version-counter
increments, so gradient-bearing inputs retain the original indexed path. Tensor
subclasses also retain that path. Buffer version counters are not model state or
used by the implementation's caches. CUDA equivalence remains unmeasured here.

The initial isolated prototype ran one warmup and seven alternating pairs on
the 128-sample workload. Median tail time: 3.865500 s before, 3.649918 s candidate
(-5.577%). All final state hashes matched the original 128-sample study exactly.
The last pair showed timing drift, retained in the raw samples. This prototype
alone does not establish acceptance; its unguarded source is only a benchmark
probe. `benchmarks/performance_message_views_probe.json` records that limitation.

The actual guarded implementation has five focused tests: exact float32/float64
recurrences at multiple sequence lengths and cluster counts, noncontiguous and
nonfinite scores, preservation of gradient/subclass dispatch, and a CUDA test
that skips without hardware. Full characterization and guarded before/after
benchmarks must pass before acceptance. See the continuation record for status.

## Accepted guarded implementation

All acceptance gates passed. The complete suite ran 221 tests in 302.875 seconds:
217 passed, three CUDA skips and the installed-artifact-only gate skipped in this
source-checkout run. Original characterization fixtures, exact inference traces,
serialized configurations and public-facade regressions are unchanged. The
packaging checkpoint separately passed the installed-artifact gate.

The final benchmark loads original methods directly from Git `49d59a3`, then
alternates original/current methods on independently reconstructed models. Each
size has one warmup and seven measured pairs. Initial twelve-observation fitting
is outside these timings. Every final state, including separate profiler/memory
runs, has the exact hash from the corresponding pre-change study; maximum
absolute numerical difference is zero.

| Total observations | Before tail median (s) | After tail median (s) | Change |
| --- | ---: | ---: | ---: |
| 64 | 1.315223 | 1.259739 | -4.22% |
| 128 | 3.853518 | 3.640474 | -5.53% |

The greater-than-5% result has a measured explanation. A separate fixed
128-row/two-cluster forward+backward operator probe counts `aten::select` falling
from 1,925 to 1,160 and `aten::copy_` from 638 to 383. Matmul remains 254 calls,
sum 255 and in-place division 255. Probe outputs match exactly. On the actual
128-observation tail, forward cProfile cumulative time falls from 0.855339 to
0.774057 seconds and backward from 0.639045 to 0.546284 seconds. Both still have
464 calls. This removes repeated tensor indexing and redundant indexed write-back,
not mathematical operations, history revisits or variational updates.

Python traced peaks are 236,920 -> 253,919 bytes for the 64-observation case
(+16,999), and 439,692 -> 434,775 bytes for 128 (-4,917). These single diagnostics
include Python allocator/cache history and do not establish a consistent peak
memory improvement. Tensor views themselves report zero payload allocation in
the operator trace. No native/GPU peak-memory reduction is claimed; all persistent
captured model values/shapes are identical.

Evidence: [paired samples and diagnostics](../benchmarks/performance_p02_paired.json),
[validation summary](../benchmarks/performance_p02_validation.json), and
[complete test log](../benchmarks/performance_p02_tests.log).
`benchmark_message_rows.py --output NEW.json` reproduces the comparison; `--resume`
reuses complete cases only when source/environment metadata match. This P02
study is complete and should not be repeated merely to resume work. GP startup
fitting, LDS formulas and alignment were not optimized here.
