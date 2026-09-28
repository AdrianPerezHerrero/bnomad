> Historical 0.2 evidence. Source paths and compatibility instructions below describe the archived release, not 0.3. Use [the current architecture](architecture.md), [migration guide](migration.md), and [0.3 validation report](migration_0_3/report.md) for this release.

The implementation combines an HDP/HMM switching model with a separate GP/LDS emission model for each cluster and output. Its operational behavior includes hard assignments, heuristic objective scaling, different online and offline update schedules, and two substantially different warping implementations.

The code—not a conventional HDP, GP, or LDS formulation—is the behavioral reference.

Paths below are relative to the repository root. Package sources reside in `hdpgpc/hdpgpc/`; experiment scripts reside in `hdpgpc/tests/`.

## A. Model semantics

### A1. Principal model classes

| Class | Location | Responsibility |
|---|---|---|
| `GPI_HDP` | [GPI_HDP.py:30](/C:/Users/Adrian/Projects/Thesis/gpexplore/HDP-GPC/hdpgpc/hdpgpc/GPI_HDP.py:30) | Owns clusters, output-specific models, HDP parameters, inference schedules, proposals, warping, histories, and persistence. |
| `GPI_model` | [GPI_model.py:16](/C:/Users/Adrian/Projects/Thesis/gpexplore/HDP-GPC/hdpgpc/hdpgpc/GPI_model.py:16) | One cluster/output GP/LDS trajectory, observation membership, parameter learning, emission scoring, filtering, and smoothing orchestration. |
| `IterativeGaussianProcess` | [GPI.py:26](/C:/Users/Adrian/Projects/Thesis/gpexplore/HDP-GPC/hdpgpc/hdpgpc/GPI.py:26) | Kernel evaluation, GP fitting/interpolation, Kalman calculations, smoothing recurrences, ML LDS updates, likelihoods, and sampling. |
| `matrix_normal_inv_wishart` | `GPI_model.py:1280` | Custom distribution-like container and updates for `(A, Gamma)` and `(C, Sigma)`. Its implemented updates and score are specialized. |
| `inv_wishart` | `GPI_model.py:1417` | Covariance updates with a fixed observation matrix, used during static-model initialization. |
| AMTGP `Warping_system` | [amtgp_warping_system.py:266](/C:/Users/Adrian/Projects/Thesis/gpexplore/HDP-GPC/hdpgpc/hdpgpc/amtgp_warping_system.py:266) | **Currently selected warper.** Optimizes monotone interpolation grids with a penalized data-fitting objective. |
| `WarpPriorAMTGP` | `amtgp_warping_system.py:106` | Fixed RBF GP prior score for displacement vectors, including quadratic, log-determinant, and normalization terms. |
| Legacy `Warping_system` | [warping_system.py:20](/C:/Users/Adrian/Projects/Thesis/gpexplore/HDP-GPC/hdpgpc/hdpgpc/warping_system.py:20) | Older GP-based alignment, with a learnable GP/LDS warp model and Pyro initialization. The main model’s import is commented out. |

`GPI_models_pytorch.py` additionally defines:

| Class | Role |
|---|---|
| `ExactGPModel` | Exact GP with constant mean and scaled RBF kernel; used for initial kernel fitting. |
| `ProjectedGPModel` | Exact GP using `InducingPointKernel`; used for differing/reduced basis grids. |
| `VarProjectedGPModel` | Approximate GP with Cholesky variational distribution and learnable inducing locations; defined but not selected by the current `fit_torch` construction branches. |
| `LinearExactGPModel` | GP with negated linear mean; used by the legacy warper. |
| `AlignmentGPModel` | Experimental alignment GP with a monotone input transformation. |
| `AlignGPModel` | Experimental composition of a warp GP and signal GP. |
| `GPMean` | Uses another GP’s predictions as a mean function. |

`_LossTrace` in the AMTGP module is a dataclass for optimization traces, not an inferential model.

### A2. Mathematical quantities and code names

There are **two different time indices**:

- Global sample index: sequence position among waveforms.
- Cluster-local index: position among observations assigned to a particular GP/LDS.

There is also the within-waveform coordinate, `x_train`. Confusing these three axes would materially change inference.

Use \(N\) for waveform count, \(L\) for waveform length, \(D\) for output count, and \(B\) for GP basis size.

| Code | Mathematical interpretation / qualification |
|---|---|
| `self.T` | Number of observations processed by the switching model. |
| `self.M` | Current model count; local methods also use `M` for augmented dimensions. Its interpretation must be checked at each call. |
| `n_outputs`, `ld` | Number of outputs and output index. |
| `gpmodels[ld][m]` | Emission trajectory for output \(d\), cluster \(k\). |
| `wp_sys[ld][m]` | Corresponding alignment object. |
| `x_basis` | GP latent support coordinates, typically \(B\times1\). |
| `x_train`, `x_trains` | Observation coordinates, typically \(L\times1\) or \(N\times L\times1\). |
| `y`, `y_trains` | Observations, typically \(L\times D\) or \(N\times L\times D\). |
| `resp[t,k]` | Implemented hard state assignment \(r_{tk}\), obtained from message-derived scores. |
| `respPair[t,j,k]` | Hard pair assignment obtained by a separate maximization over the flattened pair table. |
| `resplog`, `resp_log` | Maximum-shifted log state scores; not necessarily normalized log probabilities. |
| `q[t,k,d]` | Emission score, sometimes augmented with warp scores. It is not a variational probability distribution. |
| `q_lat[t,k,d]` | Latent-dynamics score; generally nonzero only at observations indexed by that GP. |
| `h` | Assignment/update weight; most actual inclusion and Bayesian update paths require exactly `1.0`. |
| `rho[k]` | Mean parameter of a variational Beta stick. |
| `omega[k]` | Total concentration of that Beta distribution. |
| `eta1`, `g1` | \(\rho_k\omega_k\). |
| `eta0`, `g0` | \((1-\rho_k)\omega_k\). |
| `Ebeta` | Expected stick-breaking cluster weights, optionally including remaining mass. |
| HDP `gamma` | Top-level stick concentration. Distinct from LDS `Gamma`. |
| `transAlpha`, `startAlpha` | Transition-row and initial-state concentration parameters. |
| `kappa` | Sticky self-transition contribution; all supplied presets initialize it to zero. |
| `transTheta`, `startTheta` | Dirichlet parameter arrays for transition rows and starting state. |
| `trans_A`, `transPi`, `startPi` | Log transition/start quantities; normalization differs by caller. |
| HMM `alpha`, `beta` | Forward and backward messages; unrelated to `Ebeta` or GP `alpha_ini`. |
| `margPrObs`, `margprob` | Forward normalization factors. |
| `startStateCount` | Usually `resp[0]`. |
| `transStateCount` | Usually `respPair.sum(axis=0)`. |
| GP `N` | Number of posterior inclusions; can differ from `len(indexes)` when indices are recorded without filtering. |
| GP `indexes` | Mapping from cluster-local observations to global sample indices. |
| `f_star`, `cov_f` | Filtered latent means and covariances, with an initial prior entry. |
| `f_star_sm`, `cov_f_sm` | Smoothed latent histories. |
| `A`, `Gamma` | LDS latent transition matrix and process covariance. |
| `C`, `Sigma` | LDS observation matrix and observation covariance. |
| `K`, `K_X_X` | Kernel matrices over basis coordinates, owned at different layers. |
| `P_t`, `K_t`, `J_t` | Predicted covariance, Kalman gain, smoothing gain. |
| `m_mean`, `m_r_cov`, `scale`, `n0` | Custom matrix-normal/inverse-Wishart parameterization. Names should not be assumed to imply a standard library parameter convention. |
| `snr_norm` | Per-observation output weights. |
| `f_ind_old[k]` | Global index of the representative waveform used for cluster alignment. |
| AMTGP `u_ctrl` | Unconstrained warp-control parameters. |
| AMTGP `g` | Monotone aligned coordinate grid. |
| `x_warp`, usually `x_w` | Displacement \(g(x)-x\); the no-warp batch branch instead returns expanded input coordinates. |
| `y_w`, `y_trains_w` | Warped signals; offline candidate storage can be \(N\times L\times D\times K\). |
| `liks`, `liks_w` | Alignment scores, with different definitions across scalar/batch paths. |

### A3. HDP stick-breaking and transition calculations

Implemented in `GPI_HDP.py`:

- `init_global_params`, `reinit_global_params`, `temp_reinit_global_params`: initialization and dimension changes.
- `create_initrho`: initializes nearly uniform weights with remaining mass `min(0.1, 1/M²)`.
- `rho2beta`: computes
  \[
  E[\beta_k]=\rho_k\prod_{j<k}(1-\rho_j),
  \]
  with optional remaining mass \(\prod_j(1-\rho_j)\).
- `beta2rho`: inverse stick mapping, using a denominator floor of `1e-100`.
- `_calcThetaFull`: constructs transition/start Dirichlet arrays.
- `_calcThetaPost`: proposal-specific transition construction.
- `find_optimum_rhoOmega`: prepares optimization inputs and invokes the adapted bnpy optimizer.
- `L_top`, `c_Dir`, `calcELBO_LinearTerms`, `elbo_Linears`: HDP objective terms.
- `compute_Pi`, `compute_trans_A`, `compute_trans_pi`: transition/start representations.

Important implementation details:

1. `_calcThetaFull` adds counts only to the leading `M-1` block/vector entries. The final position receives prior mass. This makes the passed dimension and reserved-state convention significant.
2. `_calcThetaPost` mixes the existing transition parameters with new counts using **0.8/0.2** coefficients. Starting-state parameters use a different accumulation rule.
3. Some callers use
   \[
   \psi(\theta_{jk})-\psi(\sum_l\theta_{jl}),
   \]
   while others use
   \[
   \psi(\theta_{jk})-\log\sum_l\exp\psi(\theta_{jl}).
   \]
4. `forward`, `backward`, and `coupled_state_coef` recompute transitions through `compute_trans_A`, replacing their supplied transition argument.
5. `hmm_switch=False` changes counts and objective composition, but does not globally bypass message passing.

These are separate behaviors to preserve, not interchangeable formulations.

### A4. State inference and responsibilities

The message-passing core is [GPI_HDP.py:3546](/C:/Users/Adrian/Projects/Thesis/gpexplore/HDP-GPC/hdpgpc/hdpgpc/GPI_HDP.py:3546).

- `LogLik` subtracts a maximum; it does not implement log-sum-exp normalization.
- `forward` exponentiates shifted emissions and transitions, applies probability floors, and normalizes each forward row.
- `backward` normalizes by the sum **excluding the final state**.
- `coupled_state_coef` constructs adjacent-state tables and returns their logarithms.
- `_safe_exp` does **not exponentiate**. It returns one-hot argmax assignments.
- For pair tables, `_safe_exp` maximizes over all \(K^2\) entries, independently of the state marginal argmax.

Consequently, pair assignments need not equal the transitions implied by consecutive hard state assignments.

The first pair table produced by `coupled_state_coef` is zero before taking logs. Subsequent hard maximization selects the first flattened entry. The fast path and label-reload path initialize/index pair histories differently.

### A5. Cluster creation, removal, and reordering

| Operation | Location | Behavior |
|---|---|---|
| Initial creation | `GPI_HDP.__init__` | Creates one GP and warper per initial cluster/output. Default `M` is **1**. |
| Model factory | `create_gp_default:496` | Creates/reset models and mutates main-model configuration lists. |
| Warper factory | `create_wp_sys_default:579` | Creates AMTGP warpers. An earlier identical definition is shadowed. |
| Standard online birth | `include_sample:1906` | Evaluates a provisional extra cluster; selected birth commits a fresh default model and warper per output. |
| Fast online birth | `include_sample_fast:2312` | Compares forced hard birth/absorption alternatives using retained history. |
| Offline birth proposal | `estimate_q_first:1243` | Tries reallocation, then representative-driven split proposals. |
| Tensor augmentation | `new_group:1112` | Adds responsibility, pair, emission, latent-score, and SNR columns. Does not itself create model objects. |
| Offline removal | `estimate_q_first`, `estimate_q_all:2844` | Emergency reallocation branches truncate the last model and associated tensors. |
| Tensor removal | `remove_last_group:1133` | Slices the last cluster out of score/responsibility tensors. |
| Empty-state handling | `refill`, `refill_resp` | Swaps an occupied last state into an empty position, or stops the iteration. |
| Ordering | `reorder:1091` and inline offline permutations | Sorts clusters by assigned population. Cluster identifiers are not stable. |
| Label reconstruction | `reload_model_from_labels:3952` | Reconstructs models from supplied labels and refits trajectories. |

There is no separately encapsulated general merge/death algorithm.

Offline birth proposals rank samples using normalized representative-fit, accumulated emission, and latent scores. They group approximately equal candidate scores using `rtol=0.01`, explore up to `n_explore_steps`, and accept according to the implemented objective.

Online absorption candidates are ordered by emission fit. The provisional birth copies the **worst-ranked** existing model as its template. The first absorption candidate beating the birth comparison is accepted; the algorithm does not necessarily find the best absorption candidate.

Removal is distributed across branches. `self.M`, warpers, representatives, configuration lists, and tensors are not visibly updated through one common invariant-preserving operation.

### A6. GP emissions

Primary implementation: [GPI_model.py:250](/C:/Users/Adrian/Projects/Thesis/gpexplore/HDP-GPC/hdpgpc/hdpgpc/GPI_model.py:250).

`log_sq_error` computes, after covariance stabilization,
\[
-\tfrac12(y-\mu)^\top V^{-1}(y-\mu)
-\tfrac L2\log(2\pi).
\]

**It omits \(-\frac12\log|V|\).** `_gaussian_score_shared_cov` implements the corresponding batched score.

Emission distribution construction:

- `observe`: selects a historical filtered latent mean and LDS parameters.
- `observe_last`: uses the last smoothed latent mean.
- `step_forward_last`: despite its name, currently projects the last mean through `C`; it does not apply a full `A/Gamma` prediction.
- `IterativeGaussianProcess.pred_dist`: maps the mean/covariance to query coordinates.
- On exactly matching grids, `pred_dist` returns the supplied mean and `Sigma`.
- On different grids with approximately equal diagonal entries of `Sigma`, it returns isotropic covariance.
- Otherwise it uses a GP interpolation covariance expression.

The normal emission-scoring path therefore does not simply use \(CPC^\top+\Sigma\).

Kernel assumptions:

- Standard construction is `ConstantKernel * RBF + WhiteKernel`.
- Multiple accesses hard-code sklearn kernel-tree keys such as `k1__k2__length_scale`.
- GPyTorch trains initial kernel parameters on the first included waveform.
- In the exact-GP branch for the standard composite kernel, the transferred lengthscale is **hard-coded to `1.2`**.
- `fit_kernel_params` reads learned kernel noise and immediately overwrites it with `alpha_ini[0][0]` for LDS observation noise.
- The iterative GP uses a zero prior mean; the fitted GPyTorch constant mean is not transferred.

### A7. LDS filtering, smoothing, and learning

The intended state equations are
\[
f_r=A_rf_{r-1}+\epsilon_r,\qquad
y_r=C_rf_r+\eta_r,
\]
on the **cluster-local observation sequence**.

Filtering occurs in `IterativeGaussianProcess.posterior:72`:

1. Predict mean through `A`.
2. Form predicted covariance \(APA^\top+\Gamma\).
3. Construct a kernel projection between basis and observation coordinates.
4. Compute the Kalman gain using a linear solve.
5. Update covariance using Joseph form.

A special first-step branch is selected by exact equality between `cov_prior` and a newly computed kernel matrix. It substitutes the prior covariance and a WhiteKernel-derived observation covariance.

Smoothing:

- `GPI_model.backwards` → `IterativeGaussianProcess.backward`: full trajectory.
- `backwards_pair` → `backward_notrange`: last two latent states.
- The full smoother uses an explicit matrix inverse; the pair smoother uses a solve.
- `smoother_weighted` performs hypothetical posterior inclusion and returns temporary histories. Its name does not imply a full backward pass.

Bayesian parameter learning occurs in `bayesian_new_params:966`:

- Usually requires `h == 1.0`, `snr > 0.5`, and `1 < N < estimation_limit`.
- The one-step dynamic branch computes covariance/cross-covariance quantities and then explicitly replaces them with zero matrices.
- Observation-update covariance terms are also zeroed in the one-step branch.
- Annealing adds initial noise matrices divided by `N²`.
- `estimation_limit` limits parameter learning, not all filtering, history growth, or scoring.

The custom `matrix_normal_inv_wishart.posterior` uses a specialized sequential update. For `n_k == 1`, its covariance residual is `y1p - y2p`, rather than the residual under the newly estimated transition matrix.

Its `log_likelihood_MNIW` includes quadratic/trace contributions but omits the normalizing and determinant terms of a full MNIW density. `return_LDS_param_likelihood` further scales the result by `100 / dimension`.

ML learning is a separate path:

`new_params_weighted` → `backwards` → `new_params` → `new_params_LDS`.

### A8. Objective composition

`compute_q_elbo:1796` combines:

- Emission scores at positions where `resp == 1.0`.
- Latent scores at those positions.
- HDP linear terms.
- Weighted LDS parameter scores.

Relevant scaling:

- `static_factor` and `dynamic_factor` initialize as a quantity divided by itself, ordinarily yielding `1`.
- Offline HDP terms are multiplied by basis-point count.
- Online comparisons use `one_sample=True`, making that multiplier `1`.
- Output weights and basis-point count scale LDS terms.
- Offline LDS aggregation additionally divides by occupied model count.
- With `hmm_switch=False`, the auxiliary returned objective contains only latent terms.
- Entropy is added separately in batch drivers; it is generally zero for hard assignments.

Names such as “ELBO” should therefore be documented as the **implemented comparison objective**, without asserting equivalence to an unmodified textbook bound.

### A9. Temporal alignment

Current selection: `GPI_HDP.py:10` imports AMTGP `Warping_system`.

AMTGP optimization:

1. Interpolate coarse controls, normally eight.
2. Apply `softplus + 1e-6` to produce positive increments.
3. Integrate and normalize to the coordinate endpoints.
4. Interpolate the observed signal on the aligned grid.
5. Minimize data error plus squared second-difference and displacement penalties.

Default penalties are `lambda_smooth=200`, `lambda_amp=1e-3`; Adam learning rate is `0.05`.

Important differences:

| Scalar alignment | Batch alignment |
|---|---|
| `compute_warp` | `compute_warp_batch` |
| Returns Gaussian data log-likelihood, optionally minus regularization penalties. | Returns the GP displacement-prior score. |
| Uses the first supplied noise entry through `_safe_noise`. | Uses mean supplied noise. |
| Control interpolation uses stored coordinates. | Uses `F.interpolate`, assuming approximately uniform coordinates. |
| Updates prior `theta` when rebuilding for a changed grid length. | Sets prior `theta` every call. |
| Recursive memory retains one fitted control vector. | Recursive memory retains the mean control vector across the batch. |

The HDP caller adds a baseline warp-prior score in both paths. In batch execution this combines the returned prior score with another baseline prior score.

Typical HDP callers pass a scalar signal-kernel lengthscale as `theta`. AMTGP’s penalty/prior parsers recognize tuple/list or dictionary inputs, so a scalar generally leaves their default parameters in effect.

Offline cached alignment uses representative **raw waveforms**, default kernel lengthscale, and `sqrt(ini_sigma_def)`. The uncached alternative uses current model parameters. They are not interchangeable implementations.

`warp_updating` calls `update_warp`, but AMTGP’s method only stores diagnostic displacement when recursive mode is enabled. It does not learn the legacy warp GP/LDS.

### A10. Single-output versus multi-output execution

| Aspect | Single output | Multiple outputs |
|---|---|---|
| Storage | `gpmodels[0][k]` | Separate GP/LDS per output and cluster. |
| Switching assignments | One sequence | One shared sequence across outputs. |
| Emission scores | \(N\times K\times1\) | \(N\times K\times D\), reduced over outputs. |
| Offline reduction | Scalar weight | SNR-weighted score aggregation. |
| Online reduction | Weight of one | `snr_norm` is initialized to ones, so scores are summed across outputs. |
| Warping | Per cluster | Separate alignment per output/cluster; not necessarily one shared multivariate time warp. |
| Priors | Estimated from output zero | Initial estimators still use output zero. |
| Cost | One emission family | Usually scales with output count. |

`share_gp=True` reuses/copies fitted models in selected offline initialization/proposal branches. It does not establish a general multi-task GP covariance model.

`compute_joint_xy_q` implements an experimental two-output block covariance with `tanh(rho_xy[m])` cross-correlation. No initialization of `rho_xy` or call from the main inference paths was found.

Multi-output limitations visible in the code include:

- The online warped-sample assembly contains an explicit unfinished multi-output TODO.
- Greedy warp ranking always examines `gpmodels[0]`.
- `reorder` creates output lists using list multiplication, sharing their underlying list.
- The standard online birth gate uses flattened `argmax(q_prev[-1])`, unlike the fast path’s output-reduced gate.
- Online selected-observation storage appends within output/model loops.

Multi-output online behavior should not be inferred from the offline multi-output implementation.

## B. Numerical implementation

### B1. Numerical libraries

| Library | Actual use |
|---|---|
| NumPy | Data loading, indexing/ranking, stick-breaking, optimizer arrays, random state, kernel interfaces, CPU conversions. |
| SciPy | `digamma`, `gammaln`, `polygamma`; Cholesky/`cho_solve`; L-BFGS-B and optional CG optimization. |
| PyTorch | Main state tensors, dense algebra, filtering/smoothing, emission scores, custom parameter updates, alignment, autograd, optimizers, optional CUDA. |
| GPyTorch | Exact/projected kernel fitting and legacy GP alignment. |
| scikit-learn | Kernel objects, scaling, random-state helpers, MDS visualization. |
| Pyro | Legacy warp initialization and plotting distributions/GP helpers. No Pyro SVI-based main HDP inference was found. |
| bnpy-derived code | Adapted source in `OptimizerRhoOmega.py`; no runtime `bnpy` dependency/import. |
| TorchMetrics | `SignalNoiseRatio` for output weighting. |
| pandas | Rolling-SNR helpers. |
| Matplotlib / Plotly | Visualization and experiment outputs. |
| tqdm | Progress reporting in expensive loops. |
| pickle | Whole-object persistence. |

Many pinned dependencies in `pyproject.toml` and `docs/requirements.txt` are environment/transitive dependencies rather than directly imported numerical components.

### B2. Dtype, device, and randomness

- Importing `GPI_HDP` sets:
  - `np.random.seed(42)`
  - `torch.manual_seed(42)`
  - `torch.use_deterministic_algorithms(True)`
  - default PyTorch dtype to `float64`.
- `GPI`, `GPI_model`, `util_plots`, and the legacy warper also set default dtype.
- AMTGP explicitly uses `float64`.
- `_safe_exp` creates three-dimensional pair assignments with explicit **float32**.
- `GPI_models_pytorch.py` declares `dtype = torch.float32`, but does not thereby set the global default.
- Sampling helpers default to `random_state=0`.
- Legacy SciPy GP fitting defaults to `random_state=None`.
- MDS does not set an explicit `random_state`.
- `take_standard_labels` forcibly sets `permutation=False`, overriding the supplied argument.
- No explicit Python `random` seed, thread-count configuration, or dedicated per-model RNG ownership was found.

Reproducibility depends on import order, global RNG consumption, dtype/device conversions, and library versions.

CUDA support is partial: SciPy/sklearn operations remain CPU-based, numerous tensors are created without explicit device placement, and some conversion methods expect APIs absent from `WarpPriorAMTGP`.

### B3. Stabilization constants

| Location | Stabilization / special handling |
|---|---|
| `GPI_model._chol_spd` | Symmetrizes; adds `1e-8 × mean(abs(diagonal))`, with machine-epsilon floor. |
| `log_sq_error(first=True)` | Adds `0.01 × mean(diag(initial Sigma))` to the covariance diagonal. |
| GP posterior/projection | Uses `1e-4` kernel jitter. |
| `pred_dist` | Noise-scaled `1e-4` kernel jitter; general covariance branch adds `1e-6 I`. |
| `pred_latent_dist` | Uses `1e-4 I`. |
| MNIW posterior | Scale-dependent `1e-2` jitter and `1e-8 I` in a subsequent factorization. |
| MNIW score | Symmetrization plus `1e-8 I`. |
| ML LDS estimation | Determinant/is-close checks and `1e-8 I`. |
| HDP expected-log calculations | Several callers add `1e-5` before logarithmic normalization. |
| HMM forward | Transition threshold `1e-6`, increment `1e-4`; initial probability threshold `1e-10`, increment `1e-4`. |
| HMM backward | Transition threshold `1e-5`, increment `1e-4`. |
| Pair normalization | Replaces zero denominator with `1e-10`. |
| Entropy | `1e-30` log/division protection. |
| Stick optimizer | `EPS=1e-8`. |
| AMTGP | Positive increment floor `1e-6`; interpolation/normalization denominators `1e-12`; prior jitter `1e-6`. |

Replacing these with one generic “stable solve” helper would require behavioral characterization first.

### B4. Optimization and stopping criteria

| Routine | Optimizer / iteration policy | Stopping behavior |
|---|---|---|
| `find_optimum_multiple_tries` | SciPy L-BFGS-B, analytic gradients by default | Retries `factr = [1e5, 1e7, 1e9, 1e10, 1e11]`; then retries default initialization. |
| Stick parameterization | Sigmoid for `rho`, exponential for `omega` | Boundary clipping; overflow/NaN handling; selected failures return current/prior parameters. |
| `fit_torch` | Adam | Budgets 4,000 exact, 2,000 differing-grid non-reduced, 5,000 inducing-point iterations. |
| `fit_torch` convergence | Ten-step accumulated loss change | After more than 1,000 losses, `isclose(..., 0, atol=1e-4)`. |
| Legacy `fit` | L-BFGS-B with analytic gradient | Up to 50,000 iterations; optional log-uniform restarts; optional CG/custom optimizer. |
| AMTGP scalar | Adam | Fixed requested iteration count; HDP scalar callers request 250. |
| AMTGP cached batch | Adam | Normally 50 iterations; batches of 128. No convergence-based early stop. |
| Legacy signal GP alignment fit | Rprop | Up to 550 iterations with supplied noise; consecutive-loss closeness `rtol=1e-5` after minimum iterations. |
| Legacy joint alignment | Adam | Mode-specific limits: rough 700, balanced 1,500, fine 5,000; respective loss-change tolerances `5e-2`, `1e-2`, `1e-5`. |
| ML LDS `new_params` | Alternating parameter estimation/smoothing | Up to six trials for `N<101`, otherwise four; convergence requires nondecreasing finite likelihood and `rtol=0.01`. |
| `variational_local_terms_batch` | Repeated model refits/reassignment | Objective closeness `rtol=1e-5`, or counter `i==10`. |
| Offline reallocation refinement | Repeated `estimate_q_all` | Closeness or counter `20`; split refinement uses counter `10`. |
| `include_batch` outer loop | Alternating local/global updates | User iteration limit, exact `M == max_models`, repeated assignments, multiple empty groups, `refill` termination, or single-sample termination. |

The offline outer loop is **not primarily an ELBO-tolerance loop**.

Global HDP updates generally run two alternating steps offline and four online when the applicable model-count condition is satisfied.

Some ML failure/trial handling is nested under `verbose`, so logging settings can affect control flow.

### B5. Assumptions and suspected defects requiring characterization

Observed behavior should be preserved until separately reviewed:

| Finding | Consequence |
|---|---|
| Hard assignments and exact `h == 1.0` checks | Softening responsibilities changes both assignment and learning. |
| No emission log-determinant | A normalized Gaussian likelihood would alter covariance preferences. |
| Specialized MNIW updates/scores | Replacing with a distribution library is not a structural-only change. |
| Exact-grid shortcuts | Approximately equal grids can follow a different mathematical path. |
| Fixed transferred lengthscale `1.2` | Refactoring parameter transfer can change all subsequent GP predictions. |
| First-step covariance equality test | Initialization depends on numerical equality, not just an explicit state flag. |
| Sorted `indexes` assumed | `bisect_right` and `searchsorted` rely on chronological membership. |
| Historical emission indexing | Exact membership uses local position plus one; nonmembers use a separately clamped preceding position. |
| Parameter estimators use output zero | Multi-output prior estimation is asymmetric. |
| `redefine_default` uses first ten coordinates | Offline prior reestimation has a coordinate-selection assumption. |
| `std`/`std_dif` names | `compute_estimators_LDS` computes variance-like quantities, not standard deviations. |
| `gpmodel_deepcopy` | Copies lists but shares contained tensors and parameter objects; also omits several constructor settings. |
| `create_gp_default` | Proposal creation mutates persistent configuration lists, including repeated appends. |
| Rejected proposals | Cannot be assumed completely side-effect-free. |

Static integration defects or incomplete paths:

- `test_offline.py` and `test_offline_multi_output.py` pass `with_warp=`, which `include_batch` does not accept.
- `test_step.ipynb` unpacks two results from `compute_estimators_LDS`, which returns four.
- `reorder` shares lists between outputs through list multiplication.
- Forced-model online branches can reach use of `reorder` without assigning it.
- Fast `classify=True` lacks the standard method’s early return and can proceed into model updates.
- `cluster_new_batch(learning=True)` references `warp_computed` before assignment in its stopping branch, and another branch unpacks three values from a four-value warp helper.
- That incremental-learning path also relies on state such as `self.warp` initialized elsewhere.
- `compute_joint_xy_q` depends on uninitialized `rho_xy`.
- `reset_warp_cache` does not clear the cache actually used by cached batch warping.
- Main-model device/conversion methods call `warp_gp.model_to_*`, absent from `WarpPriorAMTGP`.
- Certain static-model and legacy helper branches have interface/indexing inconsistencies.
- Legacy `log_marginal_likelihood` passes `L.dot(L.T)` as the factor argument to one `cho_solve` call, while another uses `L`.
- `selected_gpmodels` returns a contiguous prefix based on occupied count, assuming occupied models are already packed first.

These findings are source-based, not a claim that every normal experiment reaches every problematic branch.

## C. Performance optimizations

### C1. Existing caches and reused matrices

| Storage / mechanism | Purpose | Behavioral significance |
|---|---|---|
| `q` history | Reuses historical online emission scores | Past scores are not uniformly recomputed under current parameters. |
| `q_last`, `q_lat_last` | Fast online and offline continuation state | Different paths populate them at different points. |
| `resp_last`, `respPair_last` | Fast online hard-history reuse | Freezes past assignments in the fast approximation. |
| `snr_last`, count caches, `elbo_last` | Batch continuation state | Early-stop branches can occur before refreshing all fields. |
| `_warp_cache_full[(ld, ref)]` | All-sample warps/scores for one output and representative | Does not key by dataset, grid, noise, hyperparameters, model identity, or optimization settings. |
| `_warp_cache_state`, `_warp_cache_ref`, `_y_trains_w_cache` | Initialized/reset cache fields | Not the cache used by the active cached batch implementation. |
| `WarpPriorAMTGP._cache_L`, `_cache_logdet` | Reuses GP-prior Cholesky factor and determinant | Key contains device, dtype, length, endpoints, and parameters, but not all interior coordinates. |
| `_u_ctrl_prev` | Alignment warm start | Makes recursive alignment depend on call order and batch composition. |
| `GPI_model._eye_cache` | Reuses identity matrices | Keyed by size/device/dtype. |
| Shared-grid emission grouping | One covariance factorization for a group of observations | Groups by selected historical state and first-sample flag. |
| `K`, `K_X_X`, `K_inv` | Stored kernel quantities | Important state, but prediction/filtering often recomputes kernels. `K_inv` has no substantive downstream numerical read in inspected source. |
| Optimizer module dictionaries | Caches descending vectors and triangular/diagonal indices by `K` | Avoid repeated index construction. |
| Legacy warper `L` | Cached kernel Cholesky | Used for monotone-grid interpolation. |

`fmsg` and `margPrObs` support a forward-message reuse branch, but inspected code only initializes those attributes to `None`; normal calls return messages without storing them there. This is not an established active cache.

### C2. Fast online inference is an approximation

`include_sample_fast` changes computation, not merely execution speed:

- Retains previous hard responsibilities and pair assignments.
- Appends a forced last-state assignment.
- Selects a default cluster by emission score without full forward/backward inference.
- Patches latent scores only near the newest global indices.
- Uses a special birth latent-score adjustment: `h_ini=0.5`, multiplied by `5`.
- Does not add warp likelihoods to the initial last-row scores in the same way as standard online inference.
- Silently ignores exceptions in `_update_q_lat_tail`.

It must remain a separately characterized algorithm.

### C3. Performance-critical loops

| Hot region | Why expensive |
|---|---|
| Offline proposal search in `estimate_q_first` | Nested representative candidates, outputs, clusters, trajectory refits, and objective comparisons. |
| `estimate_q_all` | Repeated filtering, smoothing, and rescoring after changed assignments. |
| `full_pass_weighted` | Sequential cluster-local filtering and parameter updates; order-dependent. |
| Standard online inclusion | Repeated full-history latent scoring, HMM passes, and provisional model comparisons. |
| Dense GP/LDS algebra | Matrix solves/factorizations scale cubically in basis/waveform dimensions. |
| Irregular-grid `compute_sq_err_all` | Falls back to per-observation prediction and factorization. |
| Batch alignment | Output × representative × batch × optimization-iteration loops. |
| SNR computation | Loops over observations and invokes predictions where required. |
| HDP optimizer derivatives | Repeated \(K\times(K+1)\) derivative matrices; source explicitly marks this a hot spot. |
| MDS plotting | Pairwise GP divergence computation and \(N\times N\) distance storage. |
| CPU/GPU conversions | NumPy/SciPy/sklearn boundaries and scalar extraction can synchronize devices. |

Memory costs include:

- Current pair responsibilities: \(O(NK^2)\).
- Candidate warps: \(O(NLDK)\).
- Cached warps: grows with distinct representatives retained.
- Filtered/smoothed covariance histories: approximately \(O(NDB^2)\) across assigned models.
- Online snapshots of growing score and assignment arrays: potentially quadratic growth in observation count.

These are structural estimates; no profiling measurements were taken.

## D. Software architecture

### D1. Every Python module

| Module | Purpose |
|---|---|
| `hdpgpc/hdpgpc/GPI_HDP.py` | Main switching model; all inference modes, HDP calculations, cluster proposals, warping orchestration, state management, persistence. |
| `hdpgpc/hdpgpc/GPI_model.py` | Stateful per-cluster/output model and custom Bayesian parameter containers. |
| `hdpgpc/hdpgpc/GPI.py` | Numerical GP/LDS operations, kernel fitting, likelihoods, sampling, and basic plotting. |
| `hdpgpc/hdpgpc/GPI_models_pytorch.py` | GPyTorch model definitions and experimental alignment components. |
| `hdpgpc/hdpgpc/OptimizerRhoOmega.py` | Adapted bnpy stick-parameter optimizer and cached index helpers. |
| `hdpgpc/hdpgpc/amtgp_warping_system.py` | Current monotone interpolation-based scalar/batch alignment and GP displacement scoring. |
| `hdpgpc/hdpgpc/warping_system.py` | Legacy GP-based alignment implementation. |
| `hdpgpc/hdpgpc/get_data.py` | WFDB/UCR loading, waveform segmentation, normalization, label handling, SNR, and LDS prior estimators. |
| `hdpgpc/hdpgpc/extract_data.py` | Top-level script extracting 48 MIT-BIH records into `.npy` files; changes working directory and uses machine-specific paths. |
| `hdpgpc/hdpgpc/util_plots.py` | GP/ECG/cluster/warp visualization, label-majority summaries, purity/error reporting, and MDS. Some functions named `*_plotly` actually use Matplotlib. |
| `hdpgpc/hdpgpc/GPI_test.py` | Older top-level online research experiment with local imports and machine-specific working-directory behavior. |
| `hdpgpc/tests/test_online.py` | Single-output sequential inference without warping. |
| `hdpgpc/tests/test_online_warp.py` | Single-output sequential inference with warping. |
| `hdpgpc/tests/test_offline.py` | Intended single-output batch experiment; contains keyword mismatch. |
| `hdpgpc/tests/test_offline_multi_output.py` | Intended multi-output batch experiment; contains keyword mismatch. |
| `hdpgpc/tests/test_offline_multi_output_load.py` | Label-based reconstruction, batch classification, then incremental batch learning. |

No `__init__.py` was present in the inspected package tree.

The two notebooks demonstrate MIT-BIH and ocean data workflows; they are executable research examples, not independent package modules.

### D2. Experiment entry points

| Caller | Model entry point |
|---|---|
| `test_online.py:83` | `include_sample(..., with_warp=False)` |
| `test_online_warp.py:82` | `include_sample(..., with_warp=True)` |
| `test_offline.py:79` | Intended `include_batch`; current call uses unsupported `with_warp`. |
| `test_offline_multi_output.py:76` | Same batch keyword mismatch. |
| `test_offline_multi_output_load.py:81` | `reload_model_from_labels` |
| Same script, lines 83/85 | `cluster_new_batch`, then `cluster_new_batch(learning=True)` |
| `GPI_test.py:114` | `include_sample` |
| Both notebooks | `include_sample`, `include_batch`, plotting and posterior inspection. |

No supplied experiment calls `include_sample_fast`.

The scripts commonly declare `M=2` but do not pass it to `GPI_HDP`; the constructor consequently uses its default of one initial model.

### D3. Mutable state of the main model

| Category | Fields |
|---|---|
| Dimensions and progress | `M`, `T`, `n_outputs`, `actual_state` |
| Model ownership | `gpmodels`, `wp_sys` |
| Input/support state | `x_basis`, `x_basis_ini`, `x_basis_warp`, `x_train`, `y`, `y_train` |
| Alignment state | `y_w`, `x_w`, `liks`, lazily created `liks_w`, `warp`, `f_ind_old` |
| HDP state | `rho`, `omega`, `theta`, `transTheta`, `startTheta`, `trans_A`, `OptimizerInfo` |
| Local/history state | `h`, `q`, `resp_assigned`, `train_elbo`, `snr_norm` |
| Continuation state | `q_last`, `q_lat_last`, `snr_last`, `resp_last`, `respPair_last`, `startStateCount_last`, `transStateCount_last`, `elbo_last` |
| Message placeholders | `fmsg`, `margPrObs` |
| Alignment caches | `_warp_cache_full` plus initialized/reset legacy cache fields |
| Per-cluster options | Noise bounds, `annealing`, `model_type`, `recursive_warp`, `warp_updating`, `inducing_points`, `estimation_limit` |
| Factory defaults | `kernel_def`, initial covariance/scale defaults, bounds, mode/update defaults |
| Global settings | HDP concentrations, device flags, SNR/output-reduction settings, exploration/reestimation limits, objective scaling, verbosity |

Several fields change representation across execution modes: lists during online processing, tensors during batch processing. The object is not a mode-independent immutable model plus a separate inference workspace.

Per-GP state includes observations, membership indices, four latent histories, four LDS parameter histories, noise summaries, initial/default matrices, kernel state, fitted flags, posterior parameter objects, and identity caches.

### D4. Dependency structure

```text
Experiment scripts / notebooks
    ├── get_data
    ├── GPI_HDP
    │   ├── GPI_model
    │   │   └── GPI
    │   │       ├── GPI_models_pytorch
    │   │       └── util_plots
    │   ├── OptimizerRhoOmega
    │   └── amtgp_warping_system
    └── util_plots

Legacy warping_system
    ├── GPI_model
    ├── GPI_models_pytorch
    └── util_plots
```

Numerical code imports visualization code, which brings plotting configuration and Pyro dependencies into ordinary model imports.

### D5. Persistence, packaging, and data

- `save_swgp` first invokes `keep_last_all`, mutating/truncating model histories, then performs CPU conversion and pickles the object.
- Pickles depend on original module/class names and internal object layout.
- `gpmodel_deepcopy` is a custom partial copy, not a true deep copy.
- `pyproject.toml` declares version `0.1.0`; README declares `v0.2.0`.
- Pytest configuration points to root `tests`, whereas scripts reside in `hdpgpc/tests`.
- The `test_*.py` files are top-level experiments, with data loading, model fitting, and output generation; they do not provide an assertion-based regression suite.
- No CI configuration or reference numerical regression outputs were found.
- Package configuration uses `package-dir = {"": "hdpgpc"}` and explicitly names `hdpgpc`.
- Data paths depend on working directory; legacy loaders additionally use machine-specific external paths.

Bundled artifacts:

- 48 MIT-BIH signal arrays and 48 label arrays.
- Signals are float64 with shape `(number_of_beats, 90, 2)`.
- Records **100, 113, 115, 116, 210, 212, 213, 233** have one more label than signal segment.
- Ocean data has shape `(14376, 171, 1)`; frequency support has shape `(171,)`.
- Five result PNGs.
- No bundled mathematical specification sufficient to establish the rationale for every implementation choice.