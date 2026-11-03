# 0.3 characterization coverage

All existing numerical fixtures remain byte-identical. The native engines replay
14 historical scenarios with 4,323 exact boundaries; the public API separately
replays 12 scenarios and 4,129 boundaries. Recorded class names in trace IDs are
historical identifiers, not imports or compatibility classes. Typed recipes now
construct native state directly. Component tests protect GP fitting, posterior
state, covariance shortcuts, HDP bounds, cluster growth/pruning, LDS recurrences,
warps, caches, dtypes and known failures. Public examples cover the supported modes.

New tests in test_native_architecture.py enforce absent legacy modules, direct
component ownership, no retired runtime imports and immutable fixture hashes.
Old pickle/alias tests were replaced by native serialization and absent-interface
checks because those compatibility contracts intentionally end in 0.3. Numerical
comparisons and tolerances were not weakened. See migration_0_3/report.md.

The remainder documents historical fixture capture and the earlier extractions.
Old source paths/API advice below apply only to 0.2, not current usage.

---

# HDP-GPC characterization and regression tests

The minimal public facade has public-only end-to-end tests in
[`test_public_api.py`](../tests/test_public_api.py). A separate private
integration audit in
[`test_public_api_regression.py`](../tests/test_public_api_regression.py)
compares 12 scenarios and 4,129 frozen inference boundaries through `fit`/`update`,
including updates that request predictive and latent snapshots. These reads must
not alter subsequent posterior state. Advanced paths without a new facade
equivalent retain their original compatibility tests and are not deprecated.
See the [public API contract](../docs/public_api.md).

Typed-configuration coverage now lives in
[`HDP-GPC/tests/test_config.py`](../tests/test_config.py). It checks all
38 constructor defaults, data-only dictionary/YAML round trips, shared parameters,
validation and the six runnable example recipes. All 14 frozen inference scenarios
also construct their models through `legacy arguments -> config -> YAML -> config
-> model`, comparing all 4,323 boundary events and final states exactly. No old
golden file or tolerance was changed. See the
[configuration contract](../docs/configuration.md).

The inference-engine extraction adds exact step-level regression gates in
[`HDP-GPC/tests/test_inference.py`](../tests/test_inference.py), alongside
the unchanged original suite described below. Ten pre-edit schedules capture
3,793 real model/GP entry and exit events, including arguments, responsibilities,
HDP parameters, objectives, GP/LDS posterior state and final model snapshots.
Supplemental preserved-source cases cover single-sample and iteration-limit batch
exits, label reload and the existing forced-online-reuse failure. Their 530 events
bring the total to 4,323 boundaries across 14 scenarios. Fixture recording refuses
overwrites; supplemental capture verifies the original model source hash.

These new tests require exact numerical equality in the characterized CPU
environment and stop at the first differing event/method/field. A dedicated
`BaseException` bypasses existing broad fallback catches. Direct engine entry
points, signatures, forwarding, live owner state, model pickle fields and lazy
imports also have focused tests. The existing floating-point tolerances and all
original golden files remain unchanged. See the
[inference extraction record](../docs/inference_extraction.md) and
[pre-edit call-order pseudocode](../docs/inference_schedule_pseudocode.md).

## Scope and reference

These tests treat the current implementation as an executable specification.
They protect observed behavior, including selected existing failures; they do not
assert that the implementation is a conventional HDP, GP, or LDS algorithm.
Production code has not been changed or patched to make the fixtures run.

The original reference checkout was `HDP-GPC/`, at commit
`615109a9e474d37626abaff6c51f1c159b26124a`. Packaging brought the original tests and
unchanged fixtures into the repository's
[`tests/characterization/`](../tests/characterization/); they are also included in
the source distribution. By default test helpers select the local source tree.
Set `HDPGPC_TEST_INSTALLED=1` to test a noneditable installed package instead;
the release verifier additionally checks its `site-packages` provenance.
The two implementation maps remain the broader static inventory:

- [Current implementation map](current_implementation_map.md)
- [Inference call graph](inference_call_graph.md)

Baseline source hashes, package versions, Python version, and execution settings
are recorded in [baselines/manifest.json](../tests/characterization/baselines/manifest.json).
The source hashes establish provenance; they are not compared to the original
hashes during regression tests, since a behavior-preserving refactor must be
allowed to change source. Tests do check that their own execution does not change
production source files.

## Running the suite

After installing the package, run from the repository or unpacked source archive:

```text
python -B -m unittest discover -s tests/characterization -t . -v
```

The tests use standard-library `unittest`; no installation of pytest is required.
They can also be collected by pytest when it is installed in the same numerical
environment. Do not use the research scripts under `hdpgpc/tests` as the
regression test command: those load bundled datasets, fit models at import time,
and have known API mismatches.

The baseline environment is Python 3.11.0, NumPy 1.26.4, SciPy 1.14.1,
PyTorch 2.9.0 CPU, GPyTorch 1.13, Pyro 1.8.6, scikit-learn 1.5.2, and
TorchMetrics 1.6.0. Other versions are recorded in the manifest. A second local
environment contains PyTorch 2.14.0; it was inspected but was not used to record
or certify these baselines. Cross-version and GPU equivalence are not claimed.

## Fixtures and coverage

The main synthetic dataset is six five-point waveforms. The first waveform is
`[0, 0.4, 1, 0.4, 0]`; the contrasting waveform is
`[0, -0.6, -1.2, -0.6, 0]`. Small fixed offsets provide nearby repetitions.
Multi-output cases add a scaled, reversed, offset second output. Online cases
consume the first four waveforms. No fixture loads MIT-BIH or ocean arrays.

Main model options are explicit: CPU float64, Bayesian parameter updates,
annealing disabled, initial observation noise 0.04, process noise 0.03,
five basis points, two exploratory split candidates, and at most four models.
The offline outer loop has an explicit three-iteration limit. Initial parameter
reestimation is disabled. These are fixture configuration choices, not changes to
production defaults. SNR weighting is disabled except in the dedicated SNR case.

Real GP fitting runs with the implementation's actual optimization loops and
stopping rules. There are no optimizer stubs, shortened training-loop patches,
fake emission scores, forced cluster assignments, or replacement linear algebra.

| Test case | Fixture / path | What it protects |
|---|---|---|
| `online_dynamic` | Four single-output dynamic observations, warp off | Existing-cluster reuse, observation-by-observation labels, historical scores, hypothetical birth/absorption comparisons, filtering and Bayesian updates. This fixture retains one cluster. |
| `online_fast` | Same observations through `include_sample_fast` | Separate fast approximation, retained historical responsibilities/pairs, tail latent updates, cold-start bookkeeping, and final caches. It is not compared for equality to standard online inference. |
| `online_static` | Four single-output static observations, warp off | Actual unforced cluster birth and reuse: model counts `[1, 1, 2, 2]`, selected states `[0, 0, 1, 1]`; GP fitting, hard assignments, reordering, and HDP updates around birth. |
| `offline_dynamic` | Six single-output dynamic observations, warp off | Representative-driven split, two occupied clusters, rejected subsequent proposals, repeated assignments and stopping, GP/LDS histories, objective traces, and post-fit batch classification. |
| `offline_multi` | Six two-output dynamic observations, warp off, SNR off | Shared assignments with separate per-output GP/LDS models, output score combination, two occupied clusters per output, and post-fit classification. |
| `offline_multi_snr` | Same two-output fixture with SNR on | Data-dependent output weights, SNR normalization, weighted proposals/objectives, and shared-state classification. |
| `offline_warp` | Six single-output dynamic observations, warp on | Actual AMTGP batch alignment inside splitting/refitting, per-cluster warped data and displacements, warp scores/cache keys, two clusters, and final objective/state. |
| `offline_static` | Six single-output static observations, warp off | Static full-pass behavior, zero process dynamics/latent scores, split decisions, and static offline objective/state. |
| `emission_dynamic` | Direct dynamic GP/LDS, two observations | Actual kernel fitting; posterior filtering; pair and full smoothing; Bayesian parameter distribution updates; same-grid and interpolated observation/latent predictions; emission and latent scores. |
| `emission_static` | Direct static GP/LDS, two observations | Zero process covariance, fixed observation mapping, static Bayesian noise updates, GP predictions, and static scoring. |
| `warper_scalar` | Seven-point shifted pulse, two recursive calls, 12 Adam iterations each | Nonidentity monotone displacements, aligned signal, scalar likelihood/penalty score, entire loss trace, and recursive warm start. |
| `warper_batch` | Two shifted/scaled seven-point pulses, two recursive calls, 12 iterations each | Batch interpolation, independent controls, prior-based returned scores, cached prior determinant, and mean-control recursive warm start. |
| `hdp_messages` | Two-state, three-step fixed emission scores | Forward/backward-derived hard responsibilities and pair tables; tie breaking; pair dtype; remaining stick mass; transition quantities; actual rho/omega optimization. |
| `known_failure_online_warp` | Default one-cluster model; attempt two warped observations | Current exception on the second observation, exception text/type, and partial model state. Protects an unsupported integration path without an xfail or silent fallback. |
| `known_failure_online_multi` | Default model with two outputs; attempt online inclusion | Current first-observation indexing exception and partial model state caused by shared reordered output lists. |

Every case has a separate readable JSON baseline. The suite asserts the observed
cluster-birth sequence and occupied offline cluster counts independently of the
generic snapshot comparison. Standalone warp cases must contain displacement
greater than `1e-3`, preventing a degenerate identity-only fixture from satisfying
the intended alignment coverage.

The standalone warper's 12-iteration budget is an explicit public argument for a
small component fixture. The integrated offline warper still uses its production
50-iteration batch default; the known online failure reaches the production
250-iteration scalar alignment call before failing.

## Recorded quantities

`_support.py` snapshots all entries of the small matrices, rather than selecting
only a few entries. Snapshots copy values immediately so later in-place mutation
or list appends cannot rewrite an earlier observation's expected state.

For each online observation and the final offline model, recording includes:

- `M`, `T`, output count, selected state, occupied cluster counts and GP memberships;
- assignment histories and emission score histories;
- `rho`, `omega`, `transTheta`, `startTheta`, `trans_A`, and `compute_Pi()`;
- representative indices and SNR weights;
- stored raw coordinates, chosen/warped observations, displacements, likelihoods;
- offline/fast caches where present: `q_last`, `q_lat_last`, `resp_last`,
  `respPair_last`, counts, `snr_last`, and `elbo_last`;
- each GP's filtered and smoothed means/covariances, full `A/Gamma/C/Sigma`
  histories, initial/default matrices, kernel parameters and matrix;
- posterior parameter-container means, scales, concentrations, and fixed matrices;
- fitting state and configuration values that affect subsequent updates.

Test-only observers additionally record actual returns from
`variational_local_terms`, `compute_q_elbo`, `elbo_Linears`, and `full_LDS_elbo`.
This preserves standard online responsibilities that the model does not retain,
as well as candidate objective comparisons. The observer calls the original bound
method exactly once, serializes its result, and returns the original object. It
does not replace numerical calculations or run extra optimization. Instance
observers are removed when the fixture finishes.

The objective trace records emission and latent inputs, the emission/auxiliary
objective pair, and HDP/LDS helper returns. It avoids relying on rounded printed
ELBO summaries. Offline state includes the outer-loop `train_elbo` history.

This is a final **numerical/inferential state** snapshot, not pickle compatibility
testing or an exhaustive serialization of every incidental attribute.

## Comparison policy

- All dictionary keys, list lengths, tensor shapes, dtypes, integer values,
  booleans, labels, memberships, model counts, cache keys, and exception text are
  compared exactly.
- Hard `responsibilities`, `pair_responsibilities`, `resp_last`, and
  `respPair_last` are compared exactly even though production stores them in
  floating-point arrays.
- Finite numerical quantities use `rtol=1e-7`, `atol=1e-9`, applied to each scalar
  with an error message identifying its complete snapshot path.
- NaN and positive/negative infinity are serialized as explicit strings and their
  locations must match exactly. Initial impossible-state scores and log-zero pair
  tables legitimately contain negative infinity; they are not replaced by zeros.

The float tolerance allows tiny numerical roundoff while remaining substantially
tighter than the `1e-5` objective stopping tolerance and `1e-4` GP loss-change
threshold. The absolute tolerance supplies a bound near zero. It is a regression
comparison policy, not evidence that alternate BLAS libraries, library versions,
or GPU execution are equivalent. Discrete decision changes always fail, even when
an associated numerical difference would fall within the float tolerance.

The determinism audit uses **no tolerance at all**: it compares exact serialized
snapshot bytes from fresh processes to the baseline.

## Seeds, execution settings, and state isolation

Before every case, the harness sets Python `random`, NumPy, PyTorch, and Pyro RNGs
to 42; clears Pyro's parameter store; sets float64 and deterministic PyTorch
algorithms; and sets PyTorch CPU threads to one. Before numerical imports it sets
OMP, MKL, OpenBLAS, and NumExpr thread environment variables to one, and also
applies `threadpoolctl` limits to loaded BLAS libraries. There is no stochastic
fixture-data generation.

These settings live only in the test process. `GPI_HDP` itself seeds NumPy and
PyTorch at import time, but import-time seeding alone does not isolate individual
models from earlier RNG consumption. The harness reseeds every fixture for that
reason. It does not reseed within an inference run or clear warm-start state
between the two calls of a recursive warp fixture.

## Repeated execution and determinism

The reproducible audit command is:

```powershell
& C:/Users/Adrian/miniconda3/envs/hdpgpc_env/python.exe -B -m tests.characterization.verify_determinism --runs 3
```

To save an additional report, pass `--report` with a new path. The script refuses
to overwrite an existing report. Each run launches a fresh interpreter; every
second run reverses fixture order. It compares all case snapshots byte-for-byte
to the recorded baseline and exits unsuccessfully if any differ. Temporary
capture directories are removed afterwards.

The saved [determinism report](../tests/characterization/determinism_report.json)
records per-case hashes and the result of the repeated audit. Environment and
baseline provenance are in `baselines/manifest.json`. A numerical change is not
handled by automatically widening tolerances or replacing the baseline.

Validation on 2026-09-24: all 15 regression tests passed in both full runs
(32.588 and 30.434 seconds). Three additional fresh-process captures, including one in
reverse fixture order, each had **zero byte-level differences** across all 15
cases. No nondeterminism was observed within this fixed CPU environment and
fixture set. The two known-error cases failed reproducibly in production and
passed their explicit failure-characterization assertions.

## Unsupported behavior and variability boundaries

Two supported test cases deliberately exercise failures of the current code:

1. **One-cluster online warping:** on observation index 1,
   `GPI_HDP.compute_warp_y` executes `torch.max(liks[:-1])` with an empty tensor.
   The expected exception is `RuntimeError`. The first observation has already
   been fitted, and the second call has changed some main-model state before the
   failure. Both the exception and partial state are recorded.
2. **Multi-output online inference:** on observation index 0, `reorder` builds
   shared output lists using list multiplication. `full_LDS_elbo` subsequently
   visits more models than there are entries in `sum_resp`, producing `IndexError`.
   The partially updated GP/model state is recorded as well.

These are reproducible implementation errors, not nondeterminism. They use normal
passing regression assertions rather than skips or expected-failure markers. If
a separately authorized fix makes them succeed, the failure characterization must
be explicitly replaced with a successful-path regression test.

The audit does not establish determinism for GPU paths, the legacy GP warper,
ML LDS learning, reduced/inducing grids, large data, incremental batch learning,
forced labels, or persistence. No large-data or all-option support is implied.

Specific variability/state-dependence risks outside the tested contract are:

- Legacy GP fitting with `random_state=None` consumes the global NumPy RNG for
  restarts; MDS also lacks its own explicit seed.
- Legacy warping samples a LogNormal initialization using Pyro/PyTorch RNG state.
- AMTGP recursive initialization depends on previous calls and, for a batch, the
  previous mean control vector. Changing batch order or composition can change
  answers even with a fixed seed. Tests intentionally preserve and check that
  memory rather than resetting it mid-sequence.
- The main batch warp cache is keyed only by `(output, representative index)`;
  reuse after changing data or parameters can produce stale answers. This is
  deterministic state dependence, not RNG variability.
- Import order and earlier model construction can consume/reset global RNGs.
- Parallel reductions, device changes, and numerical-library changes are not
  covered by the CPU/single-thread determinism claim.

The suite reports exceptions and retains warnings. Only verbose progress text is
captured. It does not catch arbitrary exceptions and label them expected.

## Baseline review and maintenance

Normal tests only read baselines. They never regenerate expected values. To record
a proposed baseline in a **new** directory:

```text
python -B -m tests.characterization.record_baseline --output agents/characterization/proposed_baselines --reference-commit REVIEWED_REVISION
```

The recorder refuses an existing output directory. Review numerical changes,
discrete changes, source/environment provenance, and determinism before promoting
any candidate files. Do not regenerate baselines merely to obtain a passing suite.

`_cases.py` contains explicit fixture construction; `_support.py` contains
snapshots and comparison policy; `test_regression.py` registers one unittest per
case. `record_baseline.py` and `verify_determinism.py` are separate opt-in tools.

The most urgent future additions are successful online warping/multi-output tests
after approved fixes, ML and inducing-point paths, estimation-limit boundaries,
cluster removal, cache invalidation/reuse across datasets, and persistence.
Existing tests must continue to pass before any structural refactor begins.

## GP component extraction follow-up

The original fifteen cases and their golden files are unchanged. The repository
now also contains nine focused tests in
[`tests/test_emissions.py`](../tests/test_emissions.py), with four tiny
pre-extraction GP snapshots. They add direct coverage of exact, different-grid
projected and reduced-point fitting, optimizer groups/step counts, prior/kernel
differences, prediction/covariance branches, update gates, cache identity, and an
actual legacy instance pickle. This closes small parts of the inducing-point and
persistence gaps above; it does not cover every inducing cleanup or persistence
scenario. See the [extraction record](../docs/gp_emission_extraction.md) for
test contracts, commands, tolerances and benchmark results.

After relocation, both fresh full characterization captures (forward and reverse
order) matched the original snapshots byte for byte. All four new emission
snapshots also matched their pre-extraction values exactly in reverse order.

## Dynamics extraction follow-up

The unchanged fifteen-case suite now accompanies 33 dynamics tests in
[`test_dynamics.py`](../tests/test_dynamics.py). Twenty-six pre-extraction
snapshots cover static/linear initialization, filtering (prior, ordinary, warped,
gradient), both smoother variants, ML matrices, Bayesian trajectories, gates and
resets. Twenty-two cases succeed; four preserve explicit failures in optional
static full-smoothing/full-data and reset-likelihood paths. New unit assertions
also protect real static posterior updates, list/storage identity, dtype and
gradient behavior. The original supported static inference fixtures still pass.

After each of the two bounded extraction steps all fifteen original cases passed.
Final fresh-process captures (forward and reverse order) matched the original
snapshots byte for byte; all new dynamics snapshots also match exactly. See the
[dynamics extraction record](../docs/dynamics_extraction.md) for the exact
implicit/public interfaces, failure coverage and before/after benchmarks.

## Optional alignment extraction follow-up

The original fifteen cases and baselines remain unchanged. There are now 21
alignment tests in [`test_alignment.py`](../tests/test_alignment.py):
twelve frozen pre-extraction snapshots, their inventory check, six component
tests and two inference tests containing static/dynamic/fast/multi-output subcases.
Fixtures use tiny synthetic waveforms, seeded Python/NumPy/Torch, float64,
deterministic Torch algorithms and one numerical thread.

The snapshots cover scalar/batch waveforms, offsets, scores and every objective
trace component; warm-start controls; prior theta/factors; irregular/changed
grids; negative weights; nonrecursive/non-Bayesian settings; standard/forced
cluster candidates; and intentionally stale prior/model cache reuse. Unit tests
protect old class/pickle identity, monotonicity, iteration count, detached
outputs, reset scope and identity storage/gradient/view semantics. Constructor
disabling is compared to original no-warp online dynamic/static/fast and offline
dynamic/static/multi-output SNR results, even when call-level flags are enabled.
A patched active constructor confirms that NoAlignment never creates that engine.

All fifteen original cases passed after each logical extraction. The final full
repository run had 86 passes and one CUDA-only skip. Two independent full captures
(forward/reverse order) matched original JSON snapshots byte for byte. All twelve
new alignment fixtures also matched exactly in two fresh processes in opposite
orders. Normal component comparisons retain `rtol=1e-7`, `atol=1e-9` for floating
tensors; shape, dtype and discrete metadata remain exact. Exact comparisons
establish no observed numerical drift within that tolerance in this environment.

Reports are [full inference determinism](../benchmarks/alignment_characterization_determinism.json)
and [alignment determinism](../benchmarks/alignment_component_determinism.json).
The [alignment extraction record](../docs/alignment_extraction.md) describes
the complete pre-edit path and sequential runtime measurements, including both
threshold crossings over 5% and their alternating old/new investigations.
Known online warp/multi-output failures remain explicit. GPU behavior, every
greedy-bound/classification branch and arbitrary historic model pickles are not
established by these tests.

## HDP component extraction follow-up

The original fifteen characterization baselines remain unchanged. There are now
31 HDP tests in [`test_hdp.py`](../tests/test_hdp.py), including nineteen
frozen pre-edit snapshots, their inventory and eleven focused component tests.
They cover sticks/remainders, initialization, full/truncated/posterior theta,
distinct expected-log paths, gradients/objectives, actual optimizer results,
temporary growth, empty-state refilling, growth/pruning array contracts, state
aliasing, concentration profiles and legacy stateless calling conventions.

Additional tests force EPS clipping, in-place omega modification, factr retries,
recursive initialization fallback, SciPy warning statuses and model failure
recovery. Thus those previously documented optimizer coverage gaps now have
direct tests. No GP/LDS/warp model is constructed by the HDP-only fixtures.
The import-boundary test verifies the concrete component loads no model/emission/
alignment implementation and leaves Torch's default dtype unchanged.

All fifteen original cases passed after both logical extraction steps. The final
repository suite passed 118 tests with one CUDA-only skip. A full-suite failure
initially caught a changed owner-free entropy calling convention; its delegates
were corrected without changing tests or expected values. Final forward/reverse
full captures again matched the original snapshots byte for byte. Nineteen HDP
snapshots also matched exactly in independent forward/reverse processes, including
direct calls to the extracted expected-log helpers.

Normal floating array comparisons use `rtol=1e-7`, `atol=1e-9`; discrete values,
shapes, dtypes and scalar diagnostics are exact. These tolerances were not needed
to obtain the measured exact-match result. The optimizer source/comments and
twenty moved method bodies were checked for text/AST equality; inlining the
extracted calls also reproduces all other original model-method ASTs, including
cluster lifecycle and update ordering.

See the [HDP extraction record](../docs/hdp_extraction.md),
[final full determinism report](../benchmarks/hdp_characterization_determinism_final.json)
and [component report](../benchmarks/hdp_component_determinism.json).
The record reports reproducible Python dispatch overhead in tiny helpers and
unchanged numerical workloads, rather than claiming zero runtime change.
Every data-dependent offline pruning branch and GPU execution remain outside the
small-fixture coverage claim.


Development reference version 2 (2026-10-27) updates the historical snapshots for
intentional inference corrections. See [reference history](reproducibility.md#reference-version-2).
Repeated captures compare exactly with one another; comparison with a reviewed
baseline uses the established scientific floating-point tolerance across platforms.
