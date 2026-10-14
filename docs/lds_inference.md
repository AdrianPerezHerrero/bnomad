# LDS parameters, priors, and updates

This page expands the [short model-choice guide](model_choices.md). The
dynamic cluster model is

```text
f_t = A_k f_(t-1) + eta_t,       eta_t ~ N(0, Gamma_k)
y_t = C_k f_t + epsilon_t,       epsilon_t ~ N(0, Sigma_k).
```

`A,Gamma` describe movement of the latent state. `C,Sigma` describe how a
state produces an observation. More process noise can allow rapid latent
changes; more observation noise can tolerate deviations from the projected
state. Their effects also depend on matrix priors, state dimension, and the
GP representer. These controls do not make the decomposition identifiable
on their own.

## Map and covariance policies

`MatrixPolicyConfig` selects `cluster`, `shrinkage`, `shared`, or `fixed` for
`A` or `C`. `NoisePolicyConfig` selects `cluster`, `cluster_diagonal`,
`shared`, `shared_diagonal`, or `fixed` for `Gamma` or `Sigma`. The modes are
independent within each pair except that a *learned shared* map cannot be
combined with *learned cluster* noise, or vice versa: the current conjugate
posterior has no joint family for that mixed ownership. A fixed member can
be paired with either learned scope. The original per-cluster full MNIW
configuration remains the default.

For a learned map, shrinkage multiplies its matrix-normal column precision
and centers the prior on identity. A noise policy's `strength` multiplies
its prior degrees of freedom while preserving its covariance center. A fixed
map defaults to identity; a fixed or learned covariance may receive an
explicit positive-definite matrix. [The preset table](model_choices.md#lds-presets)
groups common choices without introducing another configuration mechanism.

These are the exact modes installed by `with_lds_preset`. “Cluster noise”
means a full learned covariance unless “diagonal” is shown. The `plastic`,
`stable`, and `all_shrunk` presets also strengthen the indicated noise
prior by `shrink_strength` without changing its center.

| Preset | `A` | `Gamma` | `C` | `Sigma` |
| --- | --- | --- | --- | --- |
| `all_free` | cluster | cluster | cluster | cluster |
| `plastic` | cluster | cluster | shrinkage | cluster, stronger prior |
| `stable` | shrinkage | cluster, stronger prior | cluster | cluster |
| `only_plastic` | cluster | cluster | fixed | fixed |
| `only_stable` | fixed | fixed | cluster | cluster |
| `shared_emission` | cluster | cluster | shared | shared |
| `shared_latent_process` | shared | shared | cluster | cluster |
| `all_shrunk` | shrinkage | cluster, stronger prior | shrinkage | cluster, stronger prior |
| `all_fixed` | fixed | fixed | fixed | fixed |
| `random_walk` | fixed at identity | fixed | fixed at identity | fixed |
| `shared_all` | shared | shared | shared | shared |
| `diagonal_noise` | cluster | cluster_diagonal | cluster | cluster_diagonal |
| `static_mixture` | absent | absent | fixed | cluster |

`static_mixture` also sets `hmm_switch=False`; no Markov transition model
remains for cluster labels. `all_fixed` leaves any configured custom fixed
matrices in place, whereas `random_walk` explicitly resets `A,C` to identity.
Neither guarantees statistically independent observations: the GP emission
and any remaining model structure still matter.

The full covariance posterior uses an inverse-Wishart factor. A diagonal
noise policy uses independent inverse-gamma factors, rather than setting
off-diagonal entries of a full inverse-Wishart draw to zero afterward. Model
level shared posteriors pool responsibility-weighted statistics across
clusters and count their prior and parameter KL once.

## Sequential and batch parameter learning

The package default is `conjugate_sequential`. Both online schedules process
one observation at a time and update the cluster LDS using its current
smoothed state moments. Offline inference may use that same sequential
update while scanning each cluster's assigned observations. Thus selecting
`schedule='offline'` does **not** itself select a batch LDS posterior.

Set `DynamicsConfig(lds_update='conjugate_batch')` for the distinct offline
method. It filters a cluster's assigned observations with matrices held
fixed during that pass, smooths the path, and updates the parameter posterior
once from the original prior. Later offline iterations may filter with the
previous final matrices, but their conjugate update starts again from the
configured prior so the same record is not counted twice. `conjugate_batch`
is unavailable online. Both methods weight statistics by responsibilities.
With `dtw_gp`, the offline batch method first projects each beat from its
fixed warped grid to the common LDS basis; this adds the approximation
described in [DTW–GP alignment](dtw_gp_alignment.md). Sampled `gp_flow`
input warps remain unsupported by batch LDS updates.

For a matrix/noise pair mapping input `x_t` to output `u_t`, its conjugate
statistics are

```text
S_xx = sum_t r_t E[x_t x_t^T]
S_ux = sum_t r_t E[u_t x_t^T]
S_uu = sum_t r_t E[u_t u_t^T],     N = sum_t r_t.
```

The `A,Gamma` update uses adjacent smoothed states and their cross
covariance. The first observation of a cluster contributes no transition.
The `C,Sigma` update uses observations and same-time smoothed states. For
known `y_t`, `E[y_t y_t^T]` is simply `y_t y_t^T`. In the usual per-cluster
learned case, the matrix-normal and inverse-Wishart factors update jointly.

Batch and sequential estimates need not match: the sequential latent path
is inferred while parameters evolve, whereas the batch path smooths under
matrices held fixed for that pass. The `natural_gradient` policy tempers
each sequential sufficient-statistic increment by `lds_step_size` in `(0,1]`;
values below one are not the exact all-data conjugate posterior. `legacy`
preserves the former heuristic update for comparison. The old name
`conjugate` is now `conjugate_sequential`.

## PCA observation map

`estimate_pca_observation(values, grid, components=r, max_beats=b)` selects a
minibatch, centers its waveforms, and takes the first `r` right singular
vectors as a fixed rectangular decoder `C` of shape `(points, r)`. The
calibration mean `mu` is kept separately:

```text
y_t = mu + C f_t + epsilon_t.
```

The pilot also estimates latent score variances and a diagonal residual
covariance. `with_pca_emission(config, estimate)` installs them and fixes
`C`; `A,Gamma` continue to be learned per cluster. Its default keeps
`Sigma` fixed. To learn it, pass an existing noise policy:

```python
from hdpgpc import HDPGPC
from hdpgpc.config import NoisePolicyConfig, with_pca_emission
from hdpgpc.dynamics import estimate_pca_observation

estimate = estimate_pca_observation(values, grid, components=12, max_beats=256)
config = with_pca_emission(
    config, estimate,
    sigma_policy=NoisePolicyConfig(mode='cluster_diagonal'),
)
model = HDPGPC(config).fit(values, grid=grid)
```

Other supported `Sigma` modes are `fixed`, `cluster` (full inverse-Wishart), `shared` (one
pooled full inverse-Wishart), and `shared_diagonal` (pooled inverse-gamma).
For fixed `C`, let `m_t,V_t` be the smoothed latent mean and covariance. The
observation-noise update uses the *waveform-space* expected scatter

```text
sum_t r_t [(y_t - mu - C m_t)(y_t - mu - C m_t)^T + C V_t C^T].
```

The diagonal policies keep its diagonal. The PCA minibatch estimates a
prior center, not a certainty about future waveforms: its explained-variance
ratio is in-sample and should be checked against other observations. The
rectangular `C` has `points * components` entries; `A,Gamma` are only
`components` by `components`. A learned **full** `Sigma` still has
`points` by `points` entries and may retain cubic dense-factorization cost
in the waveform dimension. A diagonal `Sigma` avoids that full noise
matrix, though the rest of the GP and filtering work still remains. The
PCA preset requires one output, a fixed observation grid, CPU, no warp, no
GP hyperparameter fitting, and final LDS scoring. Its `A,Gamma` remain
cluster-learned. Fixed or shared `A,Gamma` are mathematically possible but
are not currently implemented for this PCA preset.
The same fixed affine observation path also accepts a supplied pretrained
linear decoder; see [fixed observation models](fixed_observation_models.md).

## Inducing LDS basis

A separate reduced-basis option retains a square LDS map. Choose fewer
coordinates for `HDPGPCConfig.x_basis`, set
`GPEmissionConfig.inducing_points=True`, and pass the full waveform `grid`
to `fit`. The LDS states and matrices then have the reduced basis dimension,
while filtering and emission scores still use the complete observation
grid. Sequential, offline batch, and shared-observation updates project
the waveform to the basis before updating LDS parameters. The projection's
uncertainty is omitted from those conjugate sufficient statistics. This is a
practical approximation, not a sparse variational GP bound. It differs from
PCA: the PCA decoder maps latent coefficients back to the full waveform.
Reducing a dense LDS state from `points` to `basis_points` changes each
state covariance from quadratic storage and typical dense factorizations
from cubic work in `points` to the corresponding dimensions in
`basis_points`. GP projection and full-grid emission scoring still depend
on the complete waveform, so total runtime need not shrink by that ratio.

## Calibrate noise priors before fitting

`estimate_lds_noise_priors` samples contiguous blocks under a provisional
identity random walk (`A=C=I`). With differences `d_t=y_t-y_(t-1)`, its
moment equations are

```text
Var(d_t) = Gamma + 2 Sigma,     Cov(d_t,d_(t-1)) = -Sigma.
```

It returns scalar *variances*, pointwise diagnostics, and sampled block
locations. The full-record example optionally averages its two scalar
estimates and uses that common positive scale as both noise prior centers.
This is a rough order-of-magnitude initialization, not inference for `A,C`
or prior strength. Regime switches, correlated errors, and nonidentity
dynamics can bias it. Do this on a separate calibration record before a
strictly causal online evaluation. The estimator does not change a prior
mid-fit.

## Scores and interpretation

`lds_scoring='final'` scores all included observations with each cluster's
last `A,Gamma,C,Sigma`. `historical` scores with parameters associated with
each observation and reports the mean of the historical LDS parameter KLs.
The latter is an experimental scoring policy, not the standard final-posterior
ELBO. Both use the saved smoothed state moments. The emission score also
contains a trace for latent uncertainty, and the LDS parameter term contains
analytic parameter KLs. A low in-sample emission error is not a pre-update
predictive guarantee.

`birth_lds_kl_per_dimension=True` divides only the LDS parameter KL in
proposal scoring by the *latent LDS dimension*. This is enabled by default;
set it to `False` to use the unscaled LDS KL for proposals.
The same divisor applies to the baseline and proposal. It can ease births,
but the resulting comparison is an experimental score rather than the
original ELBO. The ordinary iteration report still uses the raw LDS KL.

The [configuration guide](configuration.md) gives the complete fields and
validation rules. The [record workflow](record_workflow.md) shows how to log
and inspect the fitted parameters.

## Observation-space LDS dynamics

The read-only `model_lds_dynamics` diagnostic uses the latest smoothed state
`f ~ N(m, P)` and the final cluster matrices `A, Gamma, C, Sigma` to show a
**one-step forecast**, not a previously observed beat. On the selected
observation grid, write the emission map as `y = b + H f + epsilon`. For a
fixed affine PCA or autoencoder decoder, `H = C` and `b` is its fixed offset.
For the ordinary LDS on its native grid, `H = C` and `b = 0`. With inducing
points, `H = L C`, where `L` is the fitted GP's basis-to-observation-grid
interpolation matrix; the offset is the GP prior-mean correction. The
observation covariance follows the same native/interpolated GP prediction
rule as the fitted emission.

```text
current mean       = b + H m
next mean          = b + H A m
expected change    = H (A - I) m
process covariance = H Gamma H.T
forecast covariance = H A P A.T H.T + H Gamma H.T + R
```

`R` is observation noise on the requested grid (and the GP interpolation
residual where the existing prediction rule includes it). The plot separates
propagated state, process, and observation/interpolation standard deviations;
the forecast band uses their **sum as covariances** before taking its square
root. The current-state band uses `H P H.T + R`. Process-change modes are the
leading eigenvectors of `H Gamma H.T`, scaled by two standard deviations.

The displayed observation-space transition `H A pinv(H)` is a visualization
of the learned dynamics only on the image of `H`. It is not a transition
model for noisy observed waveforms: a rectangular decoder leaves directions
outside its latent subspace undefined, and observation noise is not a latent
state. If `H` is rank deficient, the displayed matrix is only a least-squares
extension. With a tied orthonormal autoencoder `C = scale * Q`, this transition
is `Q A Q.T`, while the projected process covariance is
`scale^2 * Q Gamma Q.T`. These are diagnostic transforms, not extra fitted
parameters. For warped records, the requested grid is the model's aligned
coordinate system; the diagnostic does not invert per-beat warps.
