> Historical 0.2 evidence. Source paths and compatibility instructions below describe the archived release, not 0.3. Use [the current architecture](architecture.md), [migration guide](migration.md), and [0.3 validation report](migration_0_3/report.md) for this release.

# Dynamics boundary: current contract and extraction

## Implicit interface, documented before production edits

Category D only: separate responsibilities without changing the current scientific
or numerical implementation. Earlier GP/calculation extractions in the working
tree are retained. The initial 15-case characterization run passes (30.696 s).

The existing call chain is `GPI_HDP -> GPI_model` (now the emission component)
`-> IterativeGaussianProcess`. GPI_HDP chooses `model_type[m]`, constructs the
per-output/per-cluster object and orders operations. It directly reads and copies
the object's matrices, histories, defaults and Bayesian parameter objects.

| Boundary | Current signature / state contract |
|---|---|
| Initial matrices | Emission `GPR_static(ini_Sigma)` / `GPR_dynamic(gamma, sigma)` return `(A, Gamma, C, Sigma)` on a basis of length d. Both A/C identity; static Gamma zero; dynamic Gamma defaults .01 I; Sigma defaults .25 I. Allocations follow the global Torch dtype/device behavior already written. |
| Initial trajectory | `initial_conditions(ini_mean, ini_cov, ini_A, ini_Gamma, ini_C, ini_Sigma)` appends basis-shaped means `(d,1)` and covariance `(d,d)` to filtered/smoothed lists, sets defaults, creates MNIW for transition parameters and IW for static observation parameters (MNIW otherwise). Mean/covariance come from the GP; all-None matrices select dynamic defaults. |
| Filtering | Backend `posterior(mean_prior, cov_prior, y_train, A, Gamma, C, Sigma, warped_mean=None, x_train=None, x_warped=None, embedding=True, h=1.0)` returns two tensors. The GP backend supplies kernel, coordinates, conversion/device helpers and `pred_dist`. Some compatibility arguments are unused. |
| Full smoothing | Backend `backward(A_prior, Gamma_prior, means, covars)` mutates and returns the supplied lists. It indexes A/Gamma as histories with last-entry fallback, uses explicit inverse, and has no new jitter. Emission `backwards(h)` builds lists from filtered states excluding the initial prior, gates on h==1, and writes smoothed entries. Dynamic passes matrix histories; static passes matrices directly, an unsupported multi-state shape path rather than a no-op. |
| Pair smoothing | Backend `backward_notrange(A_prior, Gamma_prior, means, covars)` accepts individual matrices and uses solve rather than inverse. Emission `backwards_pair(h,snr)` gates on >1 indices/h==1 and optionally snr>.5. Without snr it uses filtered states; with accepted snr it uses already-smoothed states. |
| ML parameters | Backend `new_params_LDS(A_prior,Gamma_prior,C_prior,Sigma_prior,y_samples,means,covars,model='dynamic')` returns four matrices. Static branch sets A/C identity and Gamma zero but estimates Sigma from moments. Emission `new_params` owns the existing likelihood/convergence loop; `new_params_weighted` applies the inclusion gate and sample/batch limits. |
| Bayesian parameters | Emission `bayesian_new_params(h, model_type='dynamic', full_data=False, q=None, force=False, snr=1.)` mutates distributions/matrix histories. Zero current Gamma forces static regardless of the requested label. h==1, N, estimation_limit, full_data, force and snr control updates. Preserve computed-then-zeroed covariance statistics, annealing and exception fallback. |
| Resets / objective | `reinit_LDS(save_last,save_last_diag,return_likelihood)` restores defaults or saved matrices, optionally triples covariance diagonals, and constructs MNIW for both distributions even after static IW initialization. `return_LDS_param_likelihood(first)` constructs prior objects and rescales score by 100/d. |
| History ownership | Emission owns `f_star`, `cov_f`, `f_star_sm`, `cov_f_sm`, `A`, `Gamma`, `C`, `Sigma`, defaults, `N`, `indexes`, `var`, `y_var`, distributions and `_eye_cache`. No separate dynamics state container exists. N counts posterior inclusions; low-SNR index-only inclusion can grow histories/indexes without growing N. |

### Actual prediction/filtering equations and special cases

`posterior` first computes `x_basis_mean = A @ mean_prior`. Kernel projection K_cov
is identity on an exact grid, otherwise a transposed solve of kernel covariance
plus `1e-4 I`. It then computes `P_t = A @ cov_prior @ A.T + Gamma`.
When `torch.equal(cov_prior, K_X_X)` holds, it instead uses the incoming covariance
without A/Gamma propagation, a zero observation mean, and the one-argument minus
two-argument kernel difference divided by h as observation covariance. Otherwise
it calls the existing GP observation prediction with `C @ x_basis_mean` and Sigma.
The mean propagation is retained even in the special first-covariance branch.

Gain uses the existing transposed solve, and covariance uses the Joseph form.
Static filtering follows these same equations with its supplied matrices; it is
not a shortcut that leaves the posterior untouched or discards observation noise.
`include_sample` scales Sigma by h before filtering; hypothetical
`posterior_weighted` also scales Gamma and passes h. Those call sites remain intact.

Emission `step_forward_last` is an observation projection of the last smoothed
mean by C with Sigma; it currently does not apply A or add Gamma/latent covariance.
It stays distinct from the prediction inside the filter. No new generic predict
method will silently unify them.

### Ordering and numerical dependencies

GPI_HDP's standard/fast online drivers keep their own proposal and commit order.
Proposals may filter, pair-smooth, then learn parameters on a temporary emission.
Committed inclusion and parameter updates retain their existing ordering.
Offline `full_pass_weighted` selects resp>.99 but actual inclusion still requires
h==1; for dynamic trajectories it filters, pair-smooths, learns parameters per
observation, then full-smooths and scores. Static offline trajectories skip that
dynamic smoothing/learning block. All schedules stay at their current call sites.

Keep full-smoother inverse versus pair-smoother solve, transpose order, in-place
list element replacement, and existing exceptions. ML code has determinant-based
`1e-8` stabilization and symmetrization. Bayesian code uses `_chol_spd`'s scaled
`1e-8` policy and the custom MNIW/IW routines' distinct jitter/scale equations.
No shared stabilization policy, detach, clone, dtype or device conversion is added.

## Smallest useful interface and implementation plan

`DynamicsModel` is a typing protocol with five operations: `initial_parameters`,
`filter`, `smooth`, `smooth_pair`, and `estimate_parameters`. `LinearDynamics`
implements the current recurrences. `StaticDynamics` shares the recurrences but
supplies the exact static initial matrices and static ML parameter branch.
Methods take existing arrays/lists explicitly. Filtering also receives the
existing GP projection backend because the actual filter depends on kernel/grid
behavior; replacing that dependency is out of scope.

The objects are stateless. No copied matrices, stored model reference, registry,
automatic device conversion or mode cached from a possibly stale Gamma is added.
The legacy backend/emission methods remain callable compatibility delegates.
Original Bayesian/ML trajectory mutation and parameter distributions move to
private dynamics modules; their explicit state argument is the existing emission
object. The core protocol does not pretend that the full high-level driver can
accept an arbitrary new dynamics implementation without those state contracts.

Two bounded steps, each followed by characterization:

1. Move initial-matrix construction and numerical filtering/smoothing/ML routines
   behind the public dynamics operations; preserve backend signatures.
2. Move trajectory-level dynamics learning, smoothing gates, reset/objective
   routines and parameter distributions; preserve emission delegates and pickle
   identities. Keep GP fitting, scoring and sample/cluster lifecycle outside.

Validation, recorded pre-change regression cases and benchmark results follow
below when complete. Original characterization goldens will not be overwritten.

## Implemented boundary

Both steps are complete. Public names are available from `hdpgpc.dynamics`.
`_models.py` defines the stateless policies; `_kalman.py` holds the original
posterior and two smoothers; `_learning.py` holds the original ML matrix step.
`_trajectory.py` contains exactly seven relocated routines: `reinit_LDS`,
`return_LDS_param_likelihood`, `backwards`, `backwards_pair`, `new_params`,
`new_params_weighted`, and `bayesian_new_params`. `_distributions.py` contains
the existing MNIW/IW classes. Legacy methods delegate and retain their signatures.

The following stay outside this boundary: GP prior mean/kernel fitting,
observation-grid prediction, scoring, sample acceptance/storage, GP resets,
high-level proposal/cluster lifecycle, output weighting and all driver schedules.
The emission still owns the tensors/lists/distributions; no serialized state field
or dynamics back-reference was added. Private trajectory routines receive that
same emission as their state argument, including its GP projection, conversion
and factorization helpers. No extra copies or new cache invalidation occur.

Static initialization/ML choose the existing static formulas. Both policies use
the actual filter with explicitly supplied A/Gamma/C/Sigma rather than replacing
caller matrices on each call. The legacy parameter-learning entry point retains
its `model` argument, including existing handling of unsupported labels. Gamma-zero
detection in trajectory methods remains in its original location and order.

The ML loop retains six trials for N<101, four otherwise, the original finite/non-
decreasing likelihood conditions and `torch.isclose(..., .01)`, including existing
verbose-dependent control flow. Bayesian routines retain estimation-limit/force
gates, annealing, low-SNR branches and their original covariance statistics.
These are moved statements, not newly designed learning policies.

Moved classes retain their old names and `hdpgpc.GPI_model` pickle paths through
the existing compatibility shim. Source-inspection locations change. As with the
prior GP extraction, rebinding imported globals on a legacy module does not rebind
the new implementation module's globals. Ordinary calls, instance/class overrides
and legacy serialization continue to use the same state/interface contracts.

## Unit and regression coverage

`tests/fixtures/dynamics_reference.json` was captured before production edits and
contains both original source hashes. It records 26 tiny cases: 22 successful
results and four explicit existing failures. Data are five-point waveforms,
three latent states, or seven-point warped observations; no external dataset is
loaded. Its approximately 1.2 MiB size comes from indented trajectory snapshots,
not a large training fixture.

`tests/test_dynamics.py` adds 33 tests:

- Sixteen numerical regression cases: static and linear initialization,
  first-prior filtering, ordinary filtering, nonidentity warped/different-grid
  filtering, filtering gradients, full smoothing, fixed-matrix pair smoothing,
  and ML parameter estimation.
- Ten trajectory regression cases: both modes' diagonal-preserving resets,
  responsibility/SNR gates, full smoothing, full-data Bayesian learning and the
  optional reset-likelihood call. Twenty-two total cases succeed. The following
  four retain exact exception types/messages, rather than being skipped:
  static full smoothing (`linalg.inv` receives a 1D tensor), static full-data
  learning (uninitialized `n_f`), and reset-likelihood in both modes (missing n0).
- One inventory test requires precisely those four failure cases.
- Six component tests compare all numerical cases via the new public interface,
  prove static filtering changes the posterior, check static ML A/C/Gamma,
  preserve default dtype behavior, verify smoother list/storage identity, and
  exercise float32 pair-smoothing gradients.

Floating arrays use the established rtol=1e-7, atol=1e-9; shape/dtype, discrete
state, object-identity assertions, exception types and messages are exact. Those
tolerances preserve the CPU reference while accommodating small library-level
roundoff; they were not widened. Separate forward/reverse executions compared
all 26 snapshot dictionaries exactly without tolerances and found no difference.

Existing nine emission tests continue to check real GP fitting, static/dynamic
posteriors and loading a populated instance serialized before the earlier GP
extraction. The import tests now also verify that DynamicsModel stays lightweight
and importing concrete dynamics does not import the GP/emission/main-model class
or reset Torch's default dtype.

## Numerical and integration validation

| Stage | Result |
|---|---|
| Before edits | 15/15 characterization cases passed, 30.696 s; repository suite 30 passed / one existing CUDA skip, 16.070 s. |
| After numerical-operation extraction | 15/15 characterization cases passed, 29.969 s; all 33 new dynamics tests passed. |
| After trajectory/distribution extraction | 15/15 characterization cases passed, 31.196 s; repository suite 63 passed / one CUDA skip, 16.953 s. |
| Final import-contract addition | Repository suite 64 passed / one CUDA skip, 18.505 s. |
| Exact repeat checks | All original characterization snapshots matched byte for byte in two fresh processes, one reverse order. All 26 new dynamics snapshots matched exactly in forward/reverse order. |
| Packaging | Offline temporary wheel contained all six private dynamics modules; new imports, static initialization and old parameter-class pickle resolution passed from the unpacked wheel. |

The original characterization goldens and GP-emission reference were not changed.
The complete inference comparisons include assignments, cluster birth/reuse,
responsibilities, GP/LDS posterior state, multi-output weights and alignment.
See [`dynamics_characterization_determinism.json`](../benchmarks/dynamics_characterization_determinism.json).

An executable-AST audit compared all four numerical function bodies and seven
trajectory bodies to the original methods, normalizing only the numerical
receiver name and excluding docstring indentation. They match exactly, including
ordered arithmetic and branches. Both distribution class ASTs are identical;
initializer arithmetic is identical after parameter/receiver-name normalization.
No Kalman/message-passing library was substituted. GPI_HDP received no changes in
this task, and its earlier independent calculation extractions are retained.

Run from the surrounding workspace:

```powershell
& C:/Users/Adrian/miniconda3/envs/hdpgpc_env/python.exe -B -m unittest discover -s HDP-GPC/tests -p 'test_*.py' -q
& C:/Users/Adrian/miniconda3/envs/hdpgpc_env/python.exe -B -m unittest discover -s tests/characterization -t . -q
& C:/Users/Adrian/miniconda3/envs/hdpgpc_env/python.exe -B -m tests.characterization.verify_determinism --runs 2 --report NEW_REPORT.json
```

## Before/after benchmarks

Same Python 3.11 / Torch 2.9.0+cpu environment, float64, seed 42 and one numerical
thread. No concurrent numerical test jobs during benchmarks. The new
`benchmark_dynamics.py` measures the original legacy calls on five-dimensional
states and three-state histories, five repetitions of 500 calls each. It creates
fresh smoothing lists per call to avoid measuring progressively mutated inputs.
Static full smoothing here uses the backend's supported history-list contract;
it does not exercise the failing emission-level matrix-as-history call.

| Dynamics | Operation | Before median | After median | Change |
|---|---|---:|---:|---:|
| Static | Filter | 268.863 us | 270.140 us | +0.47% |
| Static | Full smoother | 113.412 us | 115.145 us | +1.53% |
| Static | Pair smoother | 84.862 us | 87.758 us | +3.41% |
| Static | ML matrix step | 258.268 us | 259.839 us | +0.61% |
| Linear | Filter | 265.425 us | 266.401 us | +0.37% |
| Linear | Full smoother | 112.196 us | 113.729 us | +1.37% |
| Linear | Pair smoother | 84.105 us | 84.711 us | +0.72% |
| Linear | ML matrix step | 593.843 us | 591.781 us | -0.35% |

Raw samples: [`dynamics_before.json`](../benchmarks/dynamics_before.json) and
[`dynamics_after.json`](../benchmarks/dynamics_after.json). The largest increase
is approximately 2.90 microseconds per static pair-smoothing call. Thin delegates
add a Python call; these measurements also include timing noise. No optimization
or statistically proven equivalence is claimed.

The existing emission benchmark was also rerun before/after, including real GP
fitting plus two filtering/pair-smoothing/Bayesian updates. Three repetitions,
200 warm prediction/scoring calls per repetition, same original fixtures:

| Emission path | Fit + updates before/after | Prediction before/after | Batch score before/after |
|---|---|---|---|
| Exact | 1.257852 / 1.249385 s (-0.67%) | .782827 / .789037 ms (+0.79%) | .591649 / .583512 ms (-1.38%) |
| Projected | 2.585422 / 2.585192 s (-0.01%) | .781584 / .775205 ms (-0.82%) | 2.982134 / 2.977566 ms (-0.15%) |
| Reduced | 2.551096 / 2.562821 s (+0.46%) | .777917 / .777755 ms (-0.02%) | 2.968626 / 2.953590 ms (-0.51%) |

Raw samples: [`dynamics_emissions_before.json`](../benchmarks/dynamics_emissions_before.json)
and [`dynamics_emissions_after.json`](../benchmarks/dynamics_emissions_after.json).
There is no clear regression in these small combined workloads. The first exact
fit includes optimizer startup; medians reduce that effect. No memory benchmark,
GPU benchmark, or large-data performance claim is made.

```powershell
& C:/Users/Adrian/miniconda3/envs/hdpgpc_env/python.exe -B HDP-GPC/benchmarks/benchmark_dynamics.py --output NEW_DYNAMICS_TIMINGS.json
& C:/Users/Adrian/miniconda3/envs/hdpgpc_env/python.exe -B HDP-GPC/benchmarks/benchmark_emissions.py --output NEW_EMISSION_TIMINGS.json
```

The tests protect supported static inference, not hypothetical fixes for its
optional failing branches. CUDA, all ill-conditioned matrices, every ML outer-loop
convergence path and arbitrary historical pickles remain outside the measured
contract. This extraction deliberately leaves those calculations and failures as
written rather than introducing new numerical policy.
