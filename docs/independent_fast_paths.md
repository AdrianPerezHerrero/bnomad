# Automatic independent inference fast paths

Two separate independence assumptions enable optimizations. No new model,
configuration system or user switch is introduced.

## Independent assignments: `hmm_switch=False`

For each schedule's existing local score `s_tk`, compute directly:

```text
logits_tk = s_tk + E[log pi_k]
soft responsibilities = softmax(logits_t, axis=clusters)
hard responsibilities = one_hot(argmax(logits_t))
```

`assignment_messages` dispatches to vectorized independent marginals, skipping
forward/backward recurrences. Both offline optimizers, offline classification
and standard online local terms use it; online_fast retains its existing direct
cache paths where applicable. Remaining low-level forward/backward/coupled-state
callers also have independent shortcuts. HMM-enabled calls execute their
original recurrences, and schedules keep their existing local score choices.

Return layouts remain compatible. Independent soft pair marginals are the
adjacent-state outer products; row zero has no predecessor and stays zero.
Existing hard/soft conversion remains in place. Pair arrays are still stored as
`(observations, clusters, clusters)`, so recurrent work is removed but quadratic
pair storage remains. Independent entropy and weight ELBO formulas are unchanged.

## Independent states: fixed `A=0` with affine observations

`conjugate_batch` holds parameters fixed during the filtering E-step. With
fixed zero A, independent Gaussian states can be conditioned in parallel:

```text
P_t = initial covariance for the first state; Gamma thereafter
V_t = inverse(P_t^-1 + r_t C.T Sigma^-1 C)
m_t = V_t (P_t^-1 prior_mean_t + r_t C.T Sigma^-1 (y_t - offset))
```

The affine filter's prior mean is A times the previous mean, hence zero here.
The first included observation keeps its configured initial covariance instead
of a Gamma transition. Encoded observations remain Gaussian likelihood factors,
not noiseless latent states.

Automatic trajectory batching requires:

- `lds_update='conjugate_batch'` and final LDS scoring.
- Explicit fixed A whose value is exactly zero, without an A-gradient request.
  A learned matrix temporarily equal to zero does not qualify.
- The built-in `AffineGaussianObservation`, with fixed C, no GP hyperparameter
  fitting and its common unwarped observation grid.
- Nonzero Gamma and a fresh batch filtering pass with no included states.

This covers compatible UCR PCA/affine-autoencoder and fixed Fourier/wavelet
configurations. HMM labels may still be enabled: label independence and latent
state independence are separate assumptions.

`condition_batch` shares prior/noise factorizations and reuses posterior factors
for repeated responsibility values. Gradient-bearing weights retain separate
differentiable computations. Failed batch Cholesky rows use scalar stabilization.
RTS gains are zero, so filtered and smoothed histories coincide. State metadata
and parameter history slots retain their native layouts. Existing batch LDS
posterior updates still receive the same observations, weights and moments;
their zero-gain transition statistics skip redundant factorizations.

Observation likelihoods batch the same preceding-member/prior state lookup and
retain the first member's small noise adjustment. Latent prior/entropy scores
batch independent states and keep the scalar initial-prior term. Shared-policy
pooling and KL ownership stay in their existing components.

Sequential/natural-gradient/legacy learning, learned A, GP-projected or warped
observations, custom observation implementations and incompatible cases retain
the general trajectory path. The affine optimizations below do not require
independent states. If the scalar filter's initial-covariance equality sentinel
recurs mid-path, independent trajectory batching falls back before committing states.

## Affine observations with learned or nonzero A

MIT-BIH PCA and affine autoencoder runs can still share observation algebra
while retaining chronological LDS filtering and HMM assignments. These automatic
optimizations do not fix A to zero or remove temporal dependence.

During a `conjugate_batch` filtering pass, the built-in affine decoder and Sigma
are constant until the M-step. A temporary conditioner computes diagonal inverse
variances or the full Sigma Cholesky factor, and

```text
H = C.T Sigma^-1 C
h_t = C.T Sigma^-1 (y_t - offset)
V_t = inverse(P_t^-1 + r_t H)
m_t = V_t (P_t^-1 prior_mean_t + r_t h_t)
```

The noise factor and H are reused; each changing P_t and observation information
are still computed for its state. The existing full-noise jitter and triangular
solves are retained. The cache is local to the GP backend's filtering pass and
removed even on exceptions, before smoothing and posterior parameter updates.
It is never stored in the immutable decoder or a checkpoint. Cache use requires
the built-in affine implementation, GP fitting disabled, matching Sigma object,
dtype and device, and no Sigma gradient request. Sequential parameter updates
continue to use uncached conditioning because their Sigma may change per beat.

For `lds_scoring='final'` on the configured common affine observation grid,
observation likelihoods now batch all selected smoothed state means and
covariances in one call, even when A is learned or nonzero. Exact member,
preceding member for a nonmember, and prior-before-first-member lookup are
unchanged. The first included member retains its existing small noise adjustment
in a separate call, unless `no_first=True`. Thus the ordinary full Sigma is
factored once per scoring call rather than once per distinct state. The method
retains residual, normalization and state-covariance trace contributions.

Historical scoring, irregular grids, GP-projected observations and custom affine
subclasses retain their existing scoring path. Filtering and smoothing remain
sequential when A is nonzero; batch MNIW updates, shared parameter ownership,
split/merge refinement and proposal acceptance are unchanged.

These algebraic optimizations can introduce floating-point differences from
reordered operations. They do not restrict proposal membership, change acceptance
rules or modify priors. Only static source checks have been performed for this
implementation; runtime equivalence and performance have not been measured.
