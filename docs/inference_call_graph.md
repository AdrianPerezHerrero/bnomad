> Historical 0.2 evidence. Source paths and compatibility instructions below describe the archived release, not 0.3. Use [the current architecture](architecture.md), [migration guide](migration.md), and [0.3 validation report](migration_0_3/report.md) for this release.

The inference schedules now live behind explicit engines. Legacy
`GPI_HDP.include_sample`/`include_sample_fast` delegate to `OnlineInference`, then
`inference._online`; `include_batch`, local/proposal, classification and label-
reload methods delegate to `OfflineInference`, then `inference._offline`.
The functions receive the actual model owner, so the component calls and order
shown below remain unchanged. Original line references are historical. See the
[pre-edit step-by-step pseudocode](../docs/inference_schedule_pseudocode.md)
and [extraction/validation record](../docs/inference_extraction.md).

These graphs describe source-level control flow. They do not certify that every branch is executable without the integration issues listed above.

`H` denotes `GPI_HDP`, `G` denotes `GPI_model`, and `I` denotes `IterativeGaussianProcess`.

## Online inference

Entry: [GPI_HDP.include_sample:1906](/C:/Users/Adrian/Projects/Thesis/gpexplore/HDP-GPC/hdpgpc/hdpgpc/GPI_HDP.py:1906).

```text
Experiment loop
└── H.include_sample(x_train, y, warp, force_model, ...)
    ├── increment T; initialize output weights; append input history
    ├── optional H.compute_warp_y
    │   ├── optional H.estimate_new for greedy candidate ordering
    │   ├── G.observe_last / G.observe
    │   │   └── I.pred_dist
    │   ├── AMTGP.Warping_system.compute_warp
    │   └── WarpPriorAMTGP.log_sq_error for baseline score
    │
    ├── construct q_aux from stored q plus new row
    ├── for each output and model
    │   ├── G.compute_q_lat_all
    │   │   └── G.log_lat_error
    │   └── G.log_sq_error + warp score
    │       └── G.observe → I.pred_dist
    │
    ├── H.variational_local_terms
    │   ├── H.weight_mean
    │   ├── H.LogLik
    │   ├── H.forward
    │   ├── H.backward
    │   ├── H.coupled_state_coef
    │   └── H._safe_exp → hard assignments
    ├── H.compute_q_elbo for previous-history baseline
    │   [the source repeats this local-inference/baseline block]
    │
    ├── provisional birth comparison
    │   ├── rank existing models by emission score
    │   ├── H.gpmodel_deepcopy of worst-ranked model
    │   ├── G.reinit_GP; G.reinit_LDS
    │   ├── H.estimate_new
    │   │   ├── G.smoother_weighted
    │   │   │   └── G.posterior_weighted → I.posterior
    │   │   └── G.log_sq_error
    │   ├── G.include_weighted_sample
    │   ├── temporarily append candidate
    │   ├── G.compute_q_lat_all
    │   ├── H.variational_local_terms
    │   ├── H.compute_q_elbo(post=True, one_sample=True)
    │   └── remove provisional candidate
    │
    ├── if birth passes emission gate: test absorption candidates
    │   ├── copy existing model
    │   ├── H.estimate_new
    │   ├── G.include_weighted_sample
    │   ├── G.backwards_pair
    │   ├── G.bayesian_new_params
    │   ├── recompute candidate latent scores
    │   ├── local inference + objective comparison
    │   └── restore saved model; stop at first improving absorption
    │
    ├── classify branch: early return in standard method
    ├── apply tie handling / force_model / max_models
    ├── selected birth
    │   ├── increase M
    │   ├── H.create_gp_default for each output
    │   └── H.create_wp_sys_default for each output
    ├── H.reorder
    ├── derive start/transition counts
    ├── conditional H.reinit_global_params
    ├── four global iterations where applicable
    │   ├── H._calcThetaFull
    │   └── H.find_optimum_rhoOmega
    │
    ├── commit observation for each output/model
    │   ├── G.include_weighted_sample
    │   │   ├── first inclusion: G.fit_kernel_params → I.fit_torch
    │   │   └── G.include_sample → I.posterior
    │   └── Bayesian: G.bayesian_new_params
    │       or ML: G.new_params_weighted
    ├── append objective/assignment/score bookkeeping
    └── optional warper.update_warp
```

Ordering constraints:

- Candidate models are used for comparison, then restored/discarded.
- The selected real model is updated afterwards.
- Standard online hypothetical absorption includes pair smoothing before parameter learning.
- The ordinary committed Bayesian update does not visibly insert that same pair-smoothing call.
- Sorting can rename clusters on every observation.
- Standard online recomputes latent-history scores and messages while retaining historical emission scores.

## Fast online inference

Entry: `GPI_HDP.include_sample_fast:2312`.

```text
H.include_sample_fast
├── initialize/recover cached history
│   ├── _init_resp_cache_cold_start
│   └── _ensure_fast_cache_online
├── optional scalar warping
├── reuse q_last/q_lat_last; score new observation
├── choose default state by reduced emission score
├── _append_hard_step
├── construct birth candidate
│   ├── copy/reset model
│   ├── estimate_new
│   └── _update_q_lat_tail
├── if birth wins emission gate
│   ├── force last assignment to birth; score objective
│   └── ordered absorption comparisons
│       ├── copy/include/smooth/update candidate
│       ├── patch tail latent scores
│       └── force last assignment; compare objective
├── commit birth if selected
├── reorder
├── update HDP parameters
├── update real emission models
└── save q/latent/responsibility caches
```

No normal full forward/backward pass over past observations occurs in this method. Historical labels are retained.

## Offline inference

Entry: [GPI_HDP.include_batch:805](/C:/Users/Adrian/Projects/Thesis/gpexplore/HDP-GPC/hdpgpc/hdpgpc/GPI_HDP.py:805).

```text
H.include_batch(x_trains, y_trains, it_limit, warp)
├── optional H.reduce_num_outputs
├── H.compute_snr_ini
├── initialize resp, respPair, q, q_lat
├── optional H.redefine_default
│   ├── estimate default Sigma/Gamma
│   └── replace GP models through create_gp_default
└── outer iteration
    ├── H.refill
    │   └── optional H.refill_resp
    ├── single-state initialization/global updates
    ├── H.variational_local_terms_batch
    │   ├── prepare transition/start scores
    │   ├── conditional H.estimate_q_first
    │   │   ├── H.warp_batch_by_resp_amtgp_cached
    │   │   ├── initial G.full_pass_weighted if scores uninitialized
    │   │   ├── representative-based simplified model scoring
    │   │   ├── try reallocation into existing clusters
    │   │   │   ├── messages → hard assignments → reorder
    │   │   │   ├── refit changed model memberships
    │   │   │   ├── repeated H.estimate_q_all
    │   │   │   └── accept / emergency removal / reject
    │   │   └── try representative-driven birth proposals
    │   │       ├── rank/group candidate samples
    │   │       ├── H.new_group
    │   │       ├── compute/reuse candidate warps
    │   │       ├── construct/refit candidate GP models
    │   │       ├── repeated H.estimate_q_all
    │   │       └── objective comparison; commit birth or reject
    │   └── conditional repeated H.estimate_q_all
    │
    ├── update model count when augmented responsibilities grow
    ├── compute HDP sufficient counts
    ├── H.reinit_global_params
    ├── two global iterations
    │   ├── H._calcThetaFull
    │   └── H.find_optimum_rhoOmega
    ├── compute transition representation
    ├── entropy + H.compute_q_elbo
    ├── store histories and continuation state
    └── evaluate stopping conditions
```

`estimate_q_all`:

```text
H.estimate_q_all
├── weighted emissions → forward/backward/pair messages
├── hard assignments
├── population-based cluster permutation
├── cached warps for candidate representatives
├── for each output and cluster
│   ├── compare proposed membership with gp.indexes
│   ├── unchanged membership: reuse model and scores
│   └── changed membership
│       ├── copy/create model
│       ├── reset LDS/GP state
│       ├── G.full_pass_weighted
│       └── recompute SNR
├── compare old/new objectives
└── accept / emergency last-cluster removal / reject
```

Offline trajectory fitting:

```text
G.full_pass_weighted
├── find active observations with resp > 0.99
├── for active observations in order
│   ├── G.include_weighted_sample
│   │   ├── first fit: G.fit_kernel_params → I.fit_torch
│   │   └── G.include_sample → I.posterior
│   └── dynamic model
│       ├── G.backwards_pair → I.backward_notrange
│       └── G.bayesian_new_params
├── dynamic model: G.backwards → I.backward
├── G.compute_sq_err_all
│   ├── shared-grid grouped covariance scoring
│   └── irregular-grid fallback → G.log_sq_error
└── G.compute_q_lat_all → G.log_lat_error
```

The current `full_pass_weighted` invokes Bayesian updates for dynamic models directly. It does not select the ML alternative according to the main model’s `bayesian_params` flag.

Its `snr` argument is not passed into the actual sample-inclusion calls in this implementation.

## Batch classification and continuation

```text
H.reload_model_from_labels
├── build hard labels/pairs
├── reset/refit every output/cluster via G.full_pass_weighted
├── optional warp-and-refit pass
├── HDP updates
├── select representatives
└── save continuation scores/counts/responsibilities/objective

H.cluster_new_batch(learning=False)
├── score all new samples under existing GPs
├── compute SNR weights
├── H.forward / backward / coupled_state_coef
└── return hard labels

H.cluster_new_batch(learning=True)
├── classify incoming batch
├── concatenate old/new data and responsibility state
├── reorder/refit models
└── enter batch-like local/global update loop
```

Non-learning classification does not visibly apply its `warp` argument. Message computations can still affect model message-related state if that machinery is externally initialized.

## Shared HDP optimization subgraph

```text
H.find_optimum_rhoOmega
└── OptimizerRhoOmega.find_optimum_multiple_tries
    └── find_optimum
        ├── rhoomega2c
        └── scipy.optimize.fmin_l_bfgs_b
            └── objFunc_unconstrained
                ├── c2rhoomega
                └── objFunc_constrained
                    ├── Beta expected-log terms
                    ├── stick-breaking expectations
                    ├── calc_dEbeta_drho / calc_Psi
                    └── analytic gradients
```

## Shared offline alignment subgraph

```text
H.warp_batch_by_resp_amtgp_cached
├── warp disabled: expand original observations; zero scores
└── for each output and representative
    ├── lookup _warp_cache_full[(output, representative)]
    └── cache miss
        ├── batches of observations
        ├── AMTGP.compute_warp_batch
        │   ├── optimize independent monotone grids
        │   ├── interpolate observed signals
        │   └── WarpPriorAMTGP.log_sq_error_batch
        ├── add baseline prior score
        └── cache all-sample displacements/signals/scores
```

---

# Proposed future module structure

This is a placement proposal only. Initial extraction should retain current formulas, update schedules, side effects, branch behavior, and compatibility names.

```text
src/hdpgpc/
├── model.py                     # Public GPI_HDP-compatible facade
├── state.py                     # Explicit model/history state containers
├── config.py                    # Existing options/defaults
│
├── inference/
│   ├── online.py                # Standard online schedule
│   ├── online_fast.py           # Separate approximate schedule
│   ├── offline.py               # Batch schedule
│   ├── batch_prediction.py      # Classification and continuation
│   ├── proposals.py             # Birth/reallocation/removal decisions
│   └── objective.py             # Existing objective composition/scaling
│
├── hdp/
│   ├── sticks.py                # rho/beta transforms
│   ├── transitions.py           # Existing transition constructions
│   ├── messages.py              # Forward/backward/pair calculations
│   ├── objective.py             # HDP terms and entropy
│   └── optimizer.py             # Adapted bnpy optimization and caches
│
├── emissions/
│   ├── model.py                 # GPI_model-compatible stateful wrapper
│   ├── scoring.py               # Emission/latent scores and batch grouping
│   ├── gp.py                    # Kernel projection/prediction
│   ├── fitting.py               # GPyTorch and legacy SciPy fitting
│   └── gpytorch_models.py       # Existing model definitions
│
├── lds/
│   ├── filtering.py
│   ├── smoothing.py
│   ├── bayesian.py              # Existing custom parameter updates
│   ├── maximum_likelihood.py
│   └── distributions.py         # Existing MNIW/IW containers
│
├── alignment/
│   ├── orchestration.py         # HDP candidate/representative alignment
│   ├── amtgp.py                 # Current scalar and batch implementations
│   ├── prior.py                 # WarpPriorAMTGP
│   ├── cache.py                 # Existing key/invalidation behavior
│   └── legacy_gp.py
│
├── data/
│   ├── loaders.py
│   ├── preprocessing.py
│   └── prior_estimators.py
│
├── diagnostics/
│   ├── metrics.py
│   └── plots.py
│
├── persistence.py
└── compatibility/               # Original import paths / pickle support

experiments/
tests/
├── characterization/
├── integration/
└── performance/
docs/
```

Boundaries should follow the four requested categories:

- **A — Semantics:** assignment rules, objectives, proposal acceptance, update schedules.
- **B — Numerics:** factorization, jitter, dtype/device, parameter conventions.
- **C — Performance:** cache keys, shared computations, retained histories, approximations.
- **D — Architecture:** module placement, interfaces, state ownership, compatibility.

For example, moving `log_sq_error` belongs to D; adding its missing log-determinant belongs to A/B and must be a separately authorized change.

# Ten areas with the highest risk of behavior change

| Rank | Area | What must be characterized before refactoring |
|---:|---|---|
| **1** | **Cluster lifecycle and permutations** | Birth/death timing, representative selection, first-improving acceptance, reserved-state dimensions, cluster renaming, and all parallel state arrays. |
| **2** | **Hard responsibilities and HMM messages** | One-hot conversion, independent pair maximization, first-pair handling, asymmetric normalization, probability floors, and transition-argument replacement. |
| **3** | **Emission scores and objective scaling** | Omitted log-determinants, observation covariance shortcuts, warp-score additions, basis-count scaling, and LDS score normalization. |
| **4** | **Custom Bayesian LDS updates** | Zeroed uncertainty terms, sequential residual formulas, `n0` scaling, annealing, exact weight thresholds, and estimation-limit boundaries. |
| **5** | **Online/offline update ordering** | Filter/smooth/update order, hypothetical versus committed updates, full versus pair smoothing, and offline Bayesian-only fitting behavior. |
| **6** | **Warping backend and scalar/batch differences** | Alignment direction, score definition, scalar `theta` handling, noise reduction, representative templates, warm starts, and baseline-prior addition. |
| **7** | **Cache validity and fast-online approximation** | Representative-only keys, stale-data reuse, reset behavior, fixed historical assignments, tail patches, and exception swallowing. |
| **8** | **Copying, aliasing, and persistence** | Partial deep copies, shared parameter objects/tensors, shared output lists, proposal side effects, and save-time mutation/history truncation. |
| **9** | **GP fitting and numerical reproducibility** | Hard-coded lengthscale, learned-noise override, kernel-tree assumptions, convergence tolerances, jitter, dtype exceptions, device placement, and global RNG effects. |
| **10** | **Multi-output and experiment compatibility** | Shared labels versus separate trajectories, output-zero prior estimation, online shape assumptions, SNR aggregation, stale APIs, data/label mismatches, and original imports/pickles. |
