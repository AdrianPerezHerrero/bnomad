> Historical 0.2 evidence. Source paths and compatibility instructions below describe the archived release, not 0.3. Use [the current architecture](architecture.md), [migration guide](migration.md), and [0.3 validation report](migration_0_3/report.md) for this release.

# GP emission component extraction

## Responsibility inventory and decision (before editing production code)

This is category D, software architecture. The pre-change characterization run
passed all 15 cases in 29.651 seconds. Existing calculation extractions in the
working tree are separate work and are retained.

| Responsibility | Existing owner and behavior to retain |
|---|---|
| GP prior | `GPI_model.__init__`, `initial_conditions`, `compute_mean`; `GPI.IterativeGaussianProcess.__init__`. Zero prior mean, sklearn covariance on basis coordinates. The emission's two-argument `K` excludes WhiteKernel diagonal noise, whereas the backend's initial one-argument `K_X_X` includes it. |
| Kernel selection | `GPI_HDP.__init__`, `set_default_options`, `create_gp_default` supply sklearn kernels; normal model construction uses ConstantKernel * RBF + WhiteKernel. `GPI.fit_torch` chooses ExactGPModel or ProjectedGPModel with GPyTorch ScaleKernel/RBF. No new arbitrary-kernel promise. |
| Hyperparameters | `fit_kernel_params` and `GPI.fit_torch`: existing sklearn theta/bounds, GPyTorch defaults/constraints, log-parameter transfer. Exact composite-kernel fitting writes lengthscale 1.2; trained constant mean is not transferred. Observation Sigma is reset from the incoming alpha, not the fitted kernel noise. |
| Basis/inducing points | `x_basis` and `inducing_points` select exact versus projected fitting. Different observation/basis grids also select projected fitting. The reduced-points branch trains inducing locations, sorts the local basis and removes points using spacing < log(lengthscale). No replacement sparse-GP algorithm. |
| Prediction | `observe`, `observe_last`, `step_forward_last`, `resample_latent_mean` delegate to `GPI.pred_dist` / `pred_latent_dist`. Historical filtered and smoothed states are distinct. Observation predictions omit latent uncertainty in several paths. Exact-grid prediction returns the supplied covariance itself; equal diagonal Sigma selects isotropic covariance even with nonzero off-diagonals. |
| Scores/likelihoods | `log_sq_error`, `compute_sq_err_all`, `_gaussian_score_shared_cov` implement the emission score without a log determinant. `log_lik_sample` delegates to a separate backend likelihood. These must not become one normalized density. |
| Posterior/state | `include_sample`, `include_weighted_sample`, `posterior_weighted` call `GPI.posterior`, storing filtered/smoothed tensors with existing aliasing. Exact h == 1 gates real weighted updates; SNR <= .5 records an index without incrementing N. LDS smoothing, Bayesian/ML parameter updates and resets are coupled to this state. |
| Optimization | `GPI.fit_torch`: Adam, exact cap 4000 at .1; different-grid/nonreduced cap 2000 at .05; reduced cap 5000 with explicit .1 groups. Stop only after >1000 losses and last-ten loss-difference sum close to zero at atol 1e-4. Legacy SciPy `fit`/optimizer paths remain available and unchanged. |
| Stabilization/precision | `_chol_spd`: symmetrize, diagonal-scaled 1e-8 jitter. Backend posterior/projection, observation and latent prediction have separate 1e-4 policies; general observation covariance adds 1e-6. Keep conversions, detach/clone and CPU sklearn evaluation, float64 import defaults, and CUDA branches as written. |
| Caches | `_eye_cache` keys include size/device/dtype; grouped batch scores reuse a Cholesky per group within a call. `K`, `K_X_X`, `K_inv`, initial theta, fitted flags, prior defaults, trajectories and learned LDS parameters remain on their existing objects. GPyTorch training models are temporary. No new cache/invalidation policy. |
| Multi-output | `GPI_HDP.gpmodels[output][cluster]` owns one emission per output/cluster. Shared assignments, SNR weights, aggregation, copy/reorder and cluster lifecycle remain in GPI_HDP. No multi-task/cross-output covariance is introduced, and existing online multi-output failure remains characterized. |

## Minimal interface and extraction plan

Expose `EmissionModel`, a structural typing protocol, and
`GaussianProcessEmission`, the actual existing stateful implementation. The
protocol describes only existing initialization (`initial_conditions`), kernel
fitting (`fit_kernel_params`), last-state observation prediction (`observe_last`),
emission scoring (`log_sq_error`), and responsibility-gated inclusion
(`include_weighted_sample`). Kernel/basis, static/dynamic prior matrices and
inducing-point choice remain existing scientifically meaningful configuration.
There is no factory registry, inheritance framework, optimizer strategy, new
normalization, or new model injection option.

Move the complete `GPI_model.py` implementation to the private
`emissions._gaussian_process` module as one unit. Its LDS history and two custom
parameter-distribution classes stay alongside it because separating that state
would be a second refactor. Numerical GP/LDS backend `GPI.py` and GPyTorch models
remain untouched. The component boundary therefore isolates an existing
cluster/output emission trajectory, not a claim of GP/LDS independence.

The concrete class keeps its runtime name `GPI_model` and original pickle module
path. `GaussianProcessEmission` is its canonical descriptive public alias. The
old module re-exports the same class and parameter distributions, retaining
experiment imports, constructor signatures, mutable state and object identity.
Bare namespace imports stay lightweight. The protocol imposes no runtime checks
and does not claim that the complete GPI_HDP driver can accept any five-method
implementation: its legacy state/LDS access remains deliberately explicit.

Tests and measurements are recorded below after extraction. No characterization
goldens are regenerated.

## Implementation and preservation evidence

The move is complete. `emissions._gaussian_process` holds the original three
classes, and `GPI_model.py` supplies compatibility imports and their original
pickle module names. Public concrete access passes through that shim, so class
identity is the same in either import order. The protocol is typing-only and does
not add a method call, adapter object, state copy, validation, or conversion.

An AST comparison against the pre-extraction `GPI_model.py` found all three class
bodies identical, including method docstrings, expressions and statement order.
`GPI.py` and `GPI_models_pytorch.py` are unchanged. GPI_HDP has no changes from this
extraction; its previously extracted entropy/normalization callers remain intact.
No source experiment was edited.

### Unit contracts

The nine tests in `tests/test_emissions.py` protect:

| Test suffix | Contract |
|---|---|
| `real_training_predictions_states_and_optimizer_match_original` | Four actual fits: same-grid exact, different-grid projected, reduced/inducing points, and static dynamics. Compare priors, basis, kernel theta, backend matrices, filtered/smoothed histories, learned LDS distributions, observation/latent predictions, scores, and Adam groups/step counts. All four fixtures currently stop at 1001 Adam steps. |
| `public_identity_and_structural_interface` | New/old concrete names are identical; the protocol recognizes the component; class pickles retain legacy resolution for all three classes. |
| `pre_extraction_instance_pickle_loads_with_identical_predictions` | An actual populated instance serialized before moving code loads through the legacy path, with identical state and predictions, and can be serialized again. |
| `prior_white_noise_and_static_dynamic_initialization` | One-argument/two-argument kernel difference, zero mean, initial shared covariance storage, empty inclusion state and different static/dynamic Bayesian parameter classes. |
| `responsibility_and_snr_gates_preserve_history_aliases` | h=.999 is ignored; h=1/SNR=.5 records only an index; SNR=.51 updates; filtered/smoothed posterior objects share storage; deepcopy preserves internal sharing but separates the copy. |
| `exact_grid_returns_covariance_by_identity_and_dtype` | Exact-grid observation covariance returns Sigma itself; latent predictions return the stored mean/covariance; the CPU float64 contract remains. |
| `score_omits_log_determinant` | Zero-residual score is independent of observation variance and retains only the dimension constant. |
| `prediction_equal_diagonal_shortcut_ignores_off_diagonals` | Off-diagonal Sigma does not prevent the current isotropic shortcut on a new grid. |
| `identity_cache_reuse_dtype_separation_and_batch_scores` | Identity reuse, dtype-separated cache keys, and agreement between batched and scalar scoring. |

The ~164 KiB `tests/fixtures/emission_reference.json` was recorded **before** the
production move, with source hash, versions and seed metadata. It contains four
numerical snapshots and a base64-encoded trusted legacy pickle, not dataset data.
The recorder refuses to overwrite an existing reference. Instrumentation observes
real Adam construction/steps while calling the original optimizer unchanged.
Numerical arrays use the existing characterization tolerances (rtol=1e-7,
atol=1e-9); discrete quantities, shapes, dtypes and optimizer configuration/counts
are exact. A separate reversed-order run compared all four snapshots exactly,
without tolerances, and found no difference. Pickle bytes themselves are not a
determinism contract; their loaded numerical state is compared.

### Validation results

- Pre-change: all 15 characterization tests passed (29.651 s); all 22 existing
  repository tests passed except the one existing CUDA skip (9.185 s).
- Post-change: all 15 characterization tests passed (30.029 s), with untouched
  golden files. This includes online/offline assignments and responsibilities,
  cluster birth/reuse, static/dynamic state, multi-output aggregation, warping,
  and the two intentionally recorded unsupported online paths.
- All 31 repository tests passed with one CUDA skip, twice (16.205 s and
  17.283 s). These comprise nine emission tests, twenty calculation tests and two
  import-boundary tests. The latter now check the new public names and that
  importing the protocol does not initialize numerical libraries.
- Two fresh full characterization captures, forward and reverse order, matched
  every original snapshot **byte for byte**. See
  [`emission_characterization_determinism.json`](../benchmarks/emission_characterization_determinism.json).
- The four component snapshots also matched exactly in a separate reverse-order
  execution. No numerical difference was observed, including below the regression
  tolerances on these fixtures.
- An offline wheel build in a temporary directory included both new modules and
  the legacy shim; imports and pickle class resolution from the unpacked wheel
  passed. No build artifacts were left in the checkout.

Run from the surrounding workspace with the pinned environment:

```powershell
& C:/Users/Adrian/miniconda3/envs/hdpgpc_env/python.exe -B -m unittest discover -s HDP-GPC/tests -p 'test_*.py' -v
& C:/Users/Adrian/miniconda3/envs/hdpgpc_env/python.exe -B -m unittest discover -s tests/characterization -t . -q
& C:/Users/Adrian/miniconda3/envs/hdpgpc_env/python.exe -B -m tests.characterization.verify_determinism --runs 2 --report NEW_REPORT.json
```

### Performance measurements

No dedicated emission benchmark existed, so `benchmarks/benchmark_emissions.py`
was added and run against both versions. Same Python 3.11 environment,
PyTorch 2.9.0+cpu, float64, seed 42 and one numerical thread; no concurrent test
job during either benchmark. Five observation points, three basis points for
projected/reduced fits, seven prediction points, six-waveform batch scoring.
Three repetitions, 200 warm prediction/scoring calls per repetition. Fitting
includes two posterior updates, pair smoothing and Bayesian parameter updates.

| Path | Work | Before median | After median | Change |
|---|---|---:|---:|---:|
| Exact | Fit + two updates | 1.244657 s | 1.252577 s | +0.64% |
| Exact | Query prediction | 0.770443 ms | 0.789614 ms | +2.49% |
| Exact | Batch score | 0.585923 ms | 0.591665 ms | +0.98% |
| Projected | Fit + two updates | 2.578801 s | 2.567819 s | -0.43% |
| Projected | Query prediction | 0.784793 ms | 0.781015 ms | -0.48% |
| Projected | Batch score | 3.003994 ms | 3.014944 ms | +0.36% |
| Reduced | Fit + two updates | 2.537307 s | 2.534445 s | -0.11% |
| Reduced | Query prediction | 0.773952 ms | 0.778258 ms | +0.56% |
| Reduced | Batch score | 3.005509 ms | 2.968777 ms | -1.22% |

Raw samples: [`emissions_before.json`](../benchmarks/emissions_before.json) and
[`emissions_after.json`](../benchmarks/emissions_after.json). These are small
wall-time measurements, not evidence of an optimization or statistically proven
equivalence. The first exact fit includes lazy optimizer startup; medians reduce
that effect. No memory benchmark was performed. Arithmetic, state allocation and
cache ownership are unchanged; importing one additional shim/module has not been
separately profiled.

```powershell
& C:/Users/Adrian/miniconda3/envs/hdpgpc_env/python.exe -B HDP-GPC/benchmarks/benchmark_emissions.py --output NEW_TIMINGS.json
```

### Limits and deliberately retained coupling

CUDA is unavailable in this environment. CPU fixtures do not establish GPU
equivalence, all possible kernels, every optimizer-cap exhaustion path, collapsed
inducing-point cleanup, ML LDS learning, or large-data runtime/memory behavior.
The unchanged source provides preservation evidence for those paths, but they
are not newly claimed as tested. GPI_HDP still owns multi-output weights and
cluster lifecycle, and accesses concrete emission history/parameter attributes.
Separating the LDS state or supporting an interchangeable non-GP emission would
require another characterized change.

Source-inspection locations and rebinding globals on the old compatibility
module differ after relocation; class patching and ordinary public calls retain
the same object. The supported high-level APIs, legacy imports and tested pickle
resolution remain intact. This extraction introduces no mathematical (A),
numerical-policy (B), or performance-algorithm (C) changes.
