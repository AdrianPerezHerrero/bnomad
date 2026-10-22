# Variational DP-MFA (independent assignments)

For the completed typed facade, recipe serialization, artifact checks and
comparison against finite maximum-likelihood MFA, see [MFA validation](mfa_validation.md).

`HDPMFA` fits a Bayesian mixture of factor analyzers to an ordinary matrix
`X` shaped `(observations, features)`. Feature dimensions are not time points.
Each component learns a centroid and a local subspace, with Gaussian parameter
uncertainty. Factors are independent across observations; this page covers independent
discrete assignments (`hmm_switch=False`). For sequence boundaries, Markov
assignments and adaptive lifecycle, see [HDP-HMM-MFA](hmm_mfa.md).

```python
from hdpgpc import HDPMFA, HDPMFAConfig
from hdpgpc.config import HDPConfig, NumericalConfig

config = HDPMFAConfig(
    n_factors=2,
    hdp=HDPConfig(M=8, hmm_switch=False, hdp_hyp='more'),
    numerics=NumericalConfig(seed=42, num_threads=1),
    loading_prior_precision=0.1,
    noise_initial=0.4, noise_floor=1e-8, learn_noise=True,
    max_iter=200, tol=1e-6, verbose='progress',
)
model = HDPMFA(config).fit(X)
labels = model.labels_
responsibilities = model.responsibilities_  # soft (N,K)
loadings, centroids = model.loadings_, model.means_
factors = model.transform_components(X)     # means (N,K,r), covariances (K,r,r)
coordinates = model.transform(X)           # MAP-selected coordinates (N,r)
```

`M` is the fixed **variational support size**, not the factor rank or a promise
that every component will be MAP occupied. Excess components can receive
negligible responsibility. `n_components_` counts support slots, `n_clusters_`
counts MAP-occupied slots, and `cluster_ids_` reports their IDs. All loading,
centroid and responsibility arrays keep the original K columns/slots.
This is the default `component_strategy='fixed'`. To allow objective-gated
births and pruning, set `component_strategy='birth_prune'` (also supported
with independent assignments); `M` then sets initial support and
`hdp.max_models` caps births. See [lifecycle controls](hmm_mfa.md#births-pruning-and-stopping). Choose `M <= N` and `n_factors <= d`.

## Existing HDP machinery and retained tail

The model reuses `HDPVariational`'s IID stick posterior, expected log weights,
optimized linear ELBO and mean stick-mass calculation. The native independent
assignment core was extracted unchanged from `_messages` into
`inference._independent_assignments`; the GP/HMM code imports the same functions.
MFA never creates fake kernels, trajectories or transition matrices.

The existing IID convention has **K random Beta sticks**, with a remaining
unassigned tail. For component occupancies `n_k = sum_i r_ik`,

\[
q(v_k)=\operatorname{Beta}(a_k,b_k),\quad
a_k=1+n_k,\quad b_k=\gamma+\sum_{l>k} n_l,
\]
\[
E\log\beta_k=\psi(a_k)-\psi(a_k+b_k)
+\sum_{l<k}[\psi(b_l)-\psi(a_l+b_l)].
\]

The last stick is **not** forced to one: `b_K=gamma`, and its KL is retained.
The variational assignment distribution has support only on the first K atoms;
unrepresented atoms/sticks retain their priors and contribute no additional KL.
This is a variational support restriction in a DP, rather than replacing the
model with a finite Dirichlet mixture. `tail_mass_` reports posterior mean mass
beyond the represented atoms. `weights_` normalizes mean stick masses over
represented atoms for the explicitly conditional prediction described below.

Reuse `HDPConfig.hdp_hyp` for concentration selection. In this independent case,
`gamma` is 0.001 / 0.1 / 1 / 100 for `less` / `balanced` / `more` / `max`.
Sticky presets have the same IID gamma as their base preset; transition/start
concentrations and kappa have no effect without an HMM. There is no separate
MFA preset system or second stick-breaking implementation.

## Coordinate ascent and objective

The [standalone FA component](factor_analysis_emission.md) supplies optimized
local evidence `ell_ik`, conditional factor moments, conjugate centroid/loading
updates and analytic Gaussian parameter KLs. Noise is one model-level diagonal
variance vector shared across components.

Each outer iteration performs:

1. Soft `q(s)`: `r_ik = softmax_k(ell_ik + E log beta_k)`, then native conjugate
   `q(v)` updates from these fractional counts.
2. Gaussian `q(B_k)` updates using each component's current conditional factor
   moments and responsibilities, always starting from the original prior.
3. Recompute conditional factors for the updated loadings. Pool their expected
   residual sums across **all** components and update shared noise once,
   dividing by total occupancy. Apply the positive variance floor.
4. Optimize the conditional factors again under the new noise and evaluate the
   complete ELBO. No factor entropy is added again after local evidence.

The full objective is

\[
\mathcal L=\sum_{ik}r_{ik}\ell_{ik}
+\sum_{ik}r_{ik}E\log\beta_k
-\sum_k KL(q(v_k)\Vert\operatorname{Beta}(1,\gamma))
-\sum_{ik}r_{ik}\log r_{ik}
-\sum_k KL(q(B_k)\Vert p(B_k)).
\]

The native `iid_linear_elbo` supplies the expected allocation term minus the
stick KL, with sticks optimized for the current counts. Assignment entropy is
added exactly once. The parameter KL includes centroid/loading covariance and
is counted once per component, without occupancy/dimension normalization. There
are no GP/LDS score multipliers, transition entropies or surrogate birth scores.
Noise is point-estimated; there is no noise posterior KL or noise entropy.

`elbo_components_` reports `local_evidence`, `allocation`, `assignment_entropy`,
positive `parameter_kl`, and `total`. `elbo_history_` includes initialization and
every completed sweep. `history_` additionally records each coordinate phase's
objective and delta. A material decrease or nonfinite objective raises a
`FloatingPointError`; numerical tolerance is `1e-8 + 1e-10 * objective_scale`.
No decreasing steps are silently accepted or hidden.

Convergence requires both relative ELBO change and maximum responsibility
change to be small (`tol`); otherwise fitting ends at `max_iter` with
`converged_=False`. The stored training responsibilities are from the final
coordinate sweep. A fresh query responsibility update can differ if fitting
stopped before convergence.

Initialization uses seeded k-means and low-rank randomized PCA **only to choose
proper initial Gaussian posterior parameters**. Conjugate updates and scoring
remain exact for the declared variational family. This initialization never
redefines the configured prior. The existing numerical seed is used (local
initialization seed zero when none is supplied). Changing row order may change
the search initialization; inference and prediction contain no sequential
conditioning in this mode.

## Configuration and outputs

`HDPMFAConfig` reuses `HDPConfig`, `NumericalConfig`, verbosity normalization and
the existing data-only value codec. It does not require a GP configuration.
Loading priors follow the shapes documented for `FactorAnalysisEmission`.
Noise values are variances, not standard deviations. `learn_noise=False` holds
the common diagonal at `noise_initial` (subject to the positive floor).
`chunk_size` bounds local factor/evidence work. Full `(N,K,r,r)` storage and
`(N,K,K)` pair responsibility arrays are unnecessary here.

| Result/method | Meaning |
| --- | --- |
| `labels_`, `responsibilities_`, `occupancies_` | Training MAP labels, soft probabilities and fractional counts |
| `means_`, `loadings_`, `parameter_covariances_` | All K centroid/loading posterior means and joint row covariances |
| `noise_`, `tail_mass_`, `stick_parameters_` | Shared diagonal variance, unused mean mass, Beta a/b |
| `components_` | Independent FA snapshots with the current shared noise |
| `predict_proba(X)` | Fresh soft variational assignments using `ell + Elog beta` |
| `predict(X)` | MAP of these fresh query assignments |
| `transform_components(X)` | K separate local coordinate systems, with their covariances |
| `transform(X, assignment='map')` | Selects the predicted component's coordinates for each row |
| `score_samples(X)` | Normalized **plug-in** Gaussian mixture log density, conditioned on represented support |
| `variational_score_samples(X)` | Local `logsumexp(ell + Elog beta)`, excluding global parameter/stick KLs |

`score_samples` uses mean loading parameters and mean stick masses normalized
over the represented atoms. It is neither the training ELBO nor exact Bayesian
posterior predictive density. `variational_score_samples` does not renormalize
away tail mass and is a local evidence diagnostic, not a normalized density or
the complete training objective. Parameter-integrated predictive moments are
available from `components_`; the corresponding density is generally not
Gaussian. Component coordinates are not aligned; averaging them is unsupported.

Training uses soft assignments exclusively. MAP operations only report labels
and select coordinate systems. `sequence_lengths` may be supplied to `fit` as
positive lengths summing to N; they are validated/saved but do not change the
independent likelihood. `hmm_switch=True` enables the boundary-aware
[HDP-HMM assignment path](hmm_mfa.md).

## Persistence and example

```python
model.save('dp_mfa.json')
restored = HDPMFA.load('dp_mfa.json')
# Also: model.state_dict(), HDPMFA.from_state_dict(...)
# Recipe only: config.to_dict(), HDPMFAConfig.from_dict(...)
```

The fitted state is data-only and stores learned shared noise once, not in every
component. It preserves training results, Gaussian parameters, Beta sticks,
history and configuration. `fit` always starts a fresh optimization; loading a
model is not a partial-fit/resume API. A failed fit leaves any earlier fitted
state intact. Public result arrays are read-only snapshots.

Run the synthetic example with a deliberately overcomplete support:

```sh
python examples/synthetic_dp_mfa.py --components 5 --iterations 150
```

It generates three low-rank Gaussian groups, fits DP-MFA, reports ELBO terms,
occupancies and label-permutation-invariant metrics, extracts local coordinates,
and writes data-only results under `agents/experiments/mfa/dp_example` by default.
The example measures clustering on easy separated groups; it does not establish
recovery for overlapping/real-world activation distributions. Tests and working
evidence are kept in ignored `agents/tests/mfa/` under this workspace's policy.

The supported validation environment is CPU float64. HMM sequence inference
and adaptive birth/pruning are covered by [milestone 3](hmm_mfa.md). Streaming,
continuous LDS dynamics, temporal warping and fully Bayesian noise are not
implemented by this MFA extension.
