# Bayesian factor-analysis emission (milestone 1)

`FactorAnalysisEmission` implements **one component**, independently of HDP
assignments. It provides conjugate Gaussian updates, local-factor inference,
normalized variational evidence and parameter KLs. It does not yet implement
component discovery, HMM inference or a training schedule. The
[DP-MFA model](dp_mfa.md) now supplies independent mixture coordinate ascent
using this component and the existing HDP weights.
The existing `HDPGPC` models and fixed affine decoders are unchanged.

```python
import torch
from hdpgpc.emissions import FactorAnalysisEmission

# X: (N,d), r: number of factors; noise values are variances, not SDs.
emission = FactorAnalysisEmission(
    observation_dim=X.shape[1], n_factors=r,
    loading_prior_mean=0., loading_prior_precision=1.,
    noise=0.1, noise_floor=1e-8,
)

# Supply a proper, nonzero loading initialization (e.g. a seeded PCA fit).
# B_initial has shape (d,r+1): centroid first, followed by loading columns.
emission.set_posterior(B_initial, torch.eye(r + 1, dtype=torch.float64))
weights = torch.ones(len(X), dtype=torch.float64)  # or fractional responsibilities

# One batch coordinate step; a later inference schedule owns this loop.
factors = emission.posterior_factors(X)
stats = emission.sufficient_statistics(X, weights, posterior=factors)
emission.update_from_statistics(stats)

# Recompute factors after changing B before taking the next coordinate step.
stats = emission.sufficient_statistics(X, weights)
emission.update_noise(stats)  # optional single-component noise M-step

evidence = emission.local_evidence(X)  # (N,), includes z prior/entropy once
objective = (weights * evidence).sum() - emission.parameter_kl()
```

The class starts at its proper prior. A zero loading mean with independent
centroid/loading prior rows can remain at the zero-loading stationary point of
coordinate ascent. A future mixture initializer must break that symmetry; the
emission does not draw random values or change global seeds/dtypes on import.
`set_posterior` changes the variational distribution, not the original prior.

## Model and posterior

For an observation vector `h` (its dimensions are not time coordinates):

\[
z_i\sim N(0,I_r),\qquad h_i\mid z_i,B\sim N(Bu_i,\operatorname{diag}\psi),
\quad u_i=[1,z_i^T]^T,\quad B=[\mu,W].
\]

Rows have independent proper Gaussian priors and posteriors:

\[
p(b_j)=N(b_{0j},\Lambda_{0j}^{-1}),\qquad q(b_j)=N(\bar b_j,V_j).
\]

`loading_prior_mean` broadcasts to `(d,r+1)`. `loading_prior_precision` accepts
a positive scalar times identity, a positive diagonal vector `(r+1,)`, a shared
SPD matrix `(r+1,r+1)`, or row-specific SPD matrices `(d,r+1,r+1)`.
The centroid/loading covariance within each row is retained.

The optimal component-conditioned local factor is

\[
R=I+\sum_j\psi_j^{-1}(\bar w_j\bar w_j^T+V_{ww,j}),\quad S=R^{-1},
\]
\[
a_i=\sum_j\psi_j^{-1}[(h_{ij}-\bar\mu_j)\bar w_j-V_{w\mu,j}],
\qquad m_i=S a_i,\qquad q(z_i)=N(m_i,S).
\]

`posterior_factors(X)` returns `FactorAnalysisPosterior`: `mean` `(N,r)`,
`covariance` `(r,r)` and `log_evidence` `(N,)`. The covariance is shared across
observations for one component; it is stored once. These are **conditional on
this component**, not MAP coordinates or an average over cluster coordinates.
Responsibilities affect sufficient statistics, not the conditional factor's
precision. All positive fractional weights contribute.

## Loading update

For `w_i = r_ik`, collect

\[
F=\sum_i w_i E[u_i u_i^T],\quad
g_j=\sum_i w_i h_{ij} E[u_i],\quad n=\sum_i w_i.
\]

`FactorAnalysisStatistics` holds `occupancy`, `factor_second_moment` F,
`observation_factor` g, and `observation_second_moment` (weighted h-squares).
It retains the `S + m m^T` covariance contribution, not just factor means.
The closed-form update is

\[
V_j=(\Lambda_{0j}+F/\psi_j)^{-1},\qquad
\bar b_j=V_j(\Lambda_{0j}b_{0j}+g_j/\psi_j).
\]

`update_from_statistics` always uses the **original prior**. Applying the same
batch statistics twice does not count observations twice. Add statistics from
disjoint minibatches before updating; do not add their posteriors. An empty
component returns to its proper prior. No arbitrary occupancy threshold is used.
`parameter_mean/covariance`, `mean`, `mean_variance`, `loadings`,
`loading_covariance` and `occupancy` expose independent snapshots.

## Local evidence and KL

The assignment evidence is

\[
\ell_i=-\frac d2\log(2\pi)-\frac12\sum_j\log\psi_j
-\frac12\sum_j\frac{(h_{ij}-\bar\mu_j)^2+V_{\mu\mu,j}}{\psi_j}
+\frac12 a_i^T S a_i+\frac12\log|S|.
\]

The implementation evaluates the same expression by completing the square,
avoiding subtraction of large quadratic forms. At `u_bar=[1,m_i]`,

\[
c_i-a_i^TSa_i=
\|h_i-\bar B\bar u_i\|_{\Psi^{-1}}^2+\|m_i\|^2
+\bar u_i^T\left(\sum_j V_j/\psi_j\right)\bar u_i.
\]

This equals `E log p(h|z,B) - KL(q(z)||N(0,I))` at the optimized factor.
**Do not add factor entropy or another latent score afterward.**
When `V_j` tends to zero, it agrees with the full normalized Gaussian
`N(h; mu, W W.T + diag(psi))`. `marginal_log_prob` evaluates that plug-in
Gaussian directly with low-rank calculations. With parameter uncertainty,
`local_evidence` is a variational contribution, not the exact Bayesian
posterior predictive density.

`parameter_kl()` sums the independent row-Gaussian KLs:

\[
\tfrac12\sum_j[\operatorname{tr}(\Lambda_{0j}V_j)
+(\bar b_j-b_{0j})^T\Lambda_{0j}(\bar b_j-b_{0j})
-(r+1)-\log|\Lambda_{0j}|-\log|V_j|].
\]

The centroid/loading row is counted jointly once. The KL is zero at the prior.
`per_dimension=True` exposes individual row KLs. There is no LDS/HDP scaling
or division by dimension or occupancy. A fixed-responsibility emission objective
is `sum_i w_i * ell_i - parameter_kl()`.

## Expected residuals and shared noise

`expected_residual` includes both factor covariance and loading uncertainty.
`noise_statistics(stats)` uses the current B posterior and saved factor moments
to return `(expected_residual_sum, occupancy)`:

\[
e_j=\sum_i w_i h_{ij}^2-2\bar b_j^Tg_j
+\operatorname{tr}[(V_j+\bar b_j\bar b_j^T)F].
\]

For shared diagonal noise, compute these statistics for **every component**,
then pool once:

\[
\psi_j=\max\left(\psi_{\min},\frac{\sum_k e_{kj}}{\sum_k n_k}\right).
\]

```python
pairs = [component.noise_statistics(stats)
         for component, stats in zip(components, component_statistics)]
shared_noise = FactorAnalysisEmission.estimate_shared_noise(
    pairs, noise_floor=1e-8, previous_noise=shared_noise,
)
# A later model-level owner can pass noise=shared_noise to each operation.
```

This is an exact constrained M-step for a point-estimated noise covariance;
there is no inverse-Gamma noise posterior or noise KL in milestone 1. Zero total
occupancy retains a supplied previous variance; without one, estimation raises
an error. Update factor moments for the current B before optimizing noise when
checking coordinate-ascent monotonicity.

## Prediction, memory and persistence

- `marginal_moments()` returns plug-in Gaussian moments; `predictive_moments()`
  integrates B uncertainty and a new factor `N(0,I)` (or supplied Gaussian
  factor moments). The latter distribution is generally not Gaussian.
- `FactorAnalysisMoments` returns `mean`, `factor`, `diagonal`; covariance is
  `factor @ factor.T + diag(diagonal)`. `variance` gives only its diagonal.
  Production operations never construct a dense observation covariance.
- Cholesky solves act on `r` or `r+1` matrices, batched over observed rows where
  appropriate. `chunk_size=1024` bounds local observation processing. Evidence
  and sufficient-statistic calls do not store all per-observation covariances.
- Internally computed second moments, factor precisions and solved covariances
  are symmetrized to remove floating-point skew. This matters near zero entries
  when division by small pixel variances amplifies accumulation roundoff.
  This preserves the conjugate formulas; no jitter or eigenvalue clipping is
  applied. External covariance/precision validation remains in place.
- CPU float64 is the reference; float32 and explicit PyTorch devices are
  accepted. CUDA behavior has not been validated for this milestone.
- `state_dict()` is versioned and JSON-compatible; `from_state_dict()` restores
  it without pickle/callbacks. `copy()` preserves device and makes independent
  tensors. Solves reject invalid Gaussian precisions instead of repairing them
  with a heuristic jitter that changes the specified prior.

## Validation and scope

Private milestone checks and the standalone CAVI demonstration live under
`agents/tests/mfa/` and `agents/experiments/mfa/`, following this workspace's
policy for test and experiment files. `agents/` is Git-ignored; these checks are
not included in the published distribution. A later release milestone must
arrange tracked CI coverage if these checks are to ship.

The mathematical specification supplied for this extension defines the Bayesian
row posterior and local evidence implemented here. Related MFA use in activation
geometry is described by [Shafran et al., ICML 2026](https://proceedings.mlr.press/v306/shafran26a.html);
this implementation does not claim that paper introduced these Bayesian updates.
The [DP-MFA schedule](dp_mfa.md) reuses `HDPVariational` and the native independent
assignment core. [HDP-HMM-MFA](hmm_mfa.md) additionally supplies Markov
assignments, sequence boundaries and adaptive birth/pruning without changing
the standalone emission mathematics.
