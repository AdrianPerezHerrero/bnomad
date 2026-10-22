# HDP-HMM mixtures of factor analyzers

The [MFA API/validation guide](mfa_validation.md) covers public type declarations,
strict data-only persistence and the independent finite-MFA comparison.

`HDPMFA` uses the same Bayesian [FA emissions](factor_analysis_emission.md)
with either independent labels or an HDP-HMM. The HMM models **discrete regime
transitions**. Continuous factors remain independent conditional on the regime;
this does not add an LDS to the factors.

## Basic use

```python
from hdpgpc import HDPMFA, HDPMFAConfig
from hdpgpc.config import HDPConfig, NumericalConfig

config = HDPMFAConfig(
    n_factors=2,
    hdp=HDPConfig(M=2, max_models=10, hmm_switch=True, hdp_hyp='more_sticky'),
    numerics=NumericalConfig(seed=42, num_threads=1),
    loading_prior_precision=0.1,
    noise_initial=0.4, learn_noise=True,
    component_strategy='birth_prune',
    n_explore_steps=2, proposal_refine_steps=3, prune_threshold=1.0,
    max_iter=200, verbose='progress',
)
# X: concatenated feature/activation matrices, shape (N,d).
# No transition connects the last row of one sequence to the next sequence.
lengths = [len(sequence_a), len(sequence_b)]
model = HDPMFA(config).fit(X, sequence_lengths=lengths)
labels = model.labels_
transitions = model.transition_matrix_
```

`hmm_switch=False` retains [DP-MFA](dp_mfa.md). Birth/pruning also supports
that mode. Default `component_strategy='fixed'` preserves the original fixed-K
MFA behavior; `hdp.M` then remains the number of represented states. With
`'birth_prune'`, it is the **initial** state count. `hdp.max_models` caps births;
when absent, the cap is N. Choose a practical cap for large datasets.

## Boundaries and assignments

`sequence_lengths` contains positive integer lengths summing to N. When omitted,
the entire matrix is one sequence. Each sequence:

- restarts the forward and backward recurrences;
- contributes its first singleton marginal to the start counts;
- contributes only its internal pair marginals to transition counts;
- contributes its own initial entropy and conditional chain entropies.

Let `r[t,k]=q(s_t=k)` and `xi[t,j,k]=q(s_(t-1)=j,s_t=k)`. Each sequence-start
row of xi is exactly zero. All other pair rows sum to one and their marginals
match the adjacent r rows. Training and queries use soft marginals; reported
labels are their MAP states, **not a Viterbi path**.

The native forward/backward recurrences live in `inference._hmm_messages`.
The existing GP entry points delegate to them, preserving their signatures and
default numerical behavior. MFA supplies its own dtype and a fresh message
context per sequence. Extreme potentials use a log-space fallback before
underflow clipping changes path probabilities. Normal passes retain the native
scaled probability recurrences. The HMM requires O(N K²) pair storage/work;
IID mode does not create HMM factors or pair arrays.

## HDP updates and objective

There are K represented states, K random Beta sticks, and a remaining tail.
The tail is not assigned observations. Transition Dirichlet factors have shape
**(K,K+1)**: active source rows and destinations including the tail. The tail
has no extra active source distribution to penalize.

For starts B, valid transition times E, and posterior-mean stick masses Eβ:

\[
n^0_k=\sum_{t\in B}r_{tk},\qquad
N_{jk}=\sum_{t\in E}\xi_{tjk},
\]
\[
\theta_{j,:}=\alpha E\beta+\kappa e_j+(N_{j,:},0),\qquad
\theta^0=\alpha_0E\beta+(n^0,0).
\]

Start and transition expected logs use digamma minus digamma of the **full**
Dirichlet sum, including tail mass. Each sweep updates the chain, Dirichlet
factors, and Beta factors with the existing `HDPVariational` formulas and
native stick optimizer. The optimizer receives exact expected logs from K
active transition rows and one start distribution. A numerically worse stick
update is discarded. This active-row adapter does not alter the legacy GP HDP
scoring path. Existing [HDP presets](hdp_presets.md), including sticky κ, apply.

The complete reported objective is

\[
\mathcal L_{bound}=\sum_{t,k}r_{tk}\ell_{tk}
+\mathcal L_{HDP}(q(v),q(\pi),q(\pi_0);N,n^0)
+H[q(s)]-\sum_k KL(q(B_k)\Vert p(B_k)),
\]
\[
H[q(s)]= -\sum_{t\in B,k}r_{tk}\log r_{tk}
-\sum_{t\in E,j,k}\xi_{tjk}\log\frac{\xi_{tjk}}{r_{t-1,j}}.
\]

Here B_k denotes the augmented centroid/loading matrix. FA local evidence
already includes optimized factor prior/entropy terms. They are not counted
twice. Shared diagonal noise is updated once from the residual sums and
occupancies of **all** components. There is no LDS term or dimension scaling.

**Inherited HDP approximation:** the native HDP calculation uses a lower bound
on expected Dirichlet prior normalizers under random sticks, following its
[bnpy source formulation](https://github.com/bnpy/bnpy/blob/master/bnpy/allocmodel/hmm/HDPHMMUtil.py).
This is a surrogate lower bound on the ELBO, not an exact evaluation of those
expected normalizers. Chain entropy and FA terms are exact within their stated
variational family. Fixed-K coordinate steps check nondecrease of this same
bound; structural proposals use it consistently. No extra birth penalty or
temperature is introduced.

## Births, pruning, and stopping

`component_strategy='birth_prune'` explores at most `n_explore_steps` proposals
per sweep. Birth candidates rank MAP-occupied groups by size-weighted spread;
a seeded two-means partition routes all fractional parent mass into two groups.
Pruning candidates have soft occupancy below `prune_threshold`; their state is
merged into the nearest remaining centroid. The threshold **shortlists** moves
and never forces deletion. Singletons may be pruned but cannot be split.
Birth and pruning candidates are interleaved when both are available.

Singleton and pair marginals are routed together with a stochastic state map.
This preserves normalization and sequence boundaries through splits, deletion
and slot compaction. Emission posterior seeds retain the **original configured
priors**; a parent's fitted posterior never becomes a newborn's prior.

Each candidate has isolated components, noise, assignments and HDP factors.
It receives `proposal_refine_steps` complete CAVI sweeps, allowing global
reassignment, then is compared with the current complete bound. At most one
move is accepted per outer iteration, requiring

```text
gain > 1e-6 + 1e-8 * max(1, abs(baseline), abs(proposal))
```

Visited MAP partitions, canonicalized over slot labels and paired with support
size, prevent structural cycles. Rejected proposals do not change scientific
state. `structure_history_` records proposals, scores, rejection reasons and
accepted moves. This is a bounded local search, not a guarantee of the global
optimum. Every candidate uses complete emission/noise updates; broader reuse
optimizations can be added later without changing the objective.
Component IDs denote current slots and are compacted after pruning; they are
not persistent identities across structural moves. All parameter and result
arrays follow the same current ordering.

Convergence requires small relative objective change, small changes in **both**
singleton and pair marginals, and no accepted structure move in the last sweep.
Reaching `max_iter` alone does not set `converged_=True`.

## Results, prediction and persistence

All [DP-MFA outputs](dp_mfa.md) remain available. Additional HMM results:

| Output | Meaning |
| --- | --- |
| `pair_responsibilities_` | (N,K,K), with zero rows at sequence starts |
| `start_counts_`, `transition_counts_` | Fractional counts over all sequences |
| `start_parameters_`, `transition_parameters_` | Dirichlet parameters, including tail destination |
| `start_probabilities_`, `transition_matrix_` | Posterior means conditioned on represented destinations |
| `sequence_lengths_`, `structure_history_` | Training boundaries and structural search records |

Pass query boundaries explicitly, for example:

```python
r = model.predict_proba(X_query, sequence_lengths=query_lengths)
xi = model.predict_pair_proba(X_query, sequence_lengths=query_lengths)
z = model.transform(X_query, sequence_lengths=query_lengths)
logp = model.score_samples(X_query, sequence_lengths=query_lengths)
```

Query methods default to a **single new sequence**, not the training boundaries.
`predict_proba` smooths using fitted variational expected-log potentials.
`score_samples` returns filtered predictive log-density increments for a
normalized **plug-in HMM**, using mean loading and mean start/transition
parameters conditioned on represented support. Their sum is the sequence
log likelihood. It is not an exact Bayesian predictive density or the ELBO.
`variational_score_samples` instead returns local log-partition increments for
variational potentials, excluding global KL terms and retaining tail effects.

JSON save/load preserves dynamic support size, shared noise, Gaussian factors,
Beta/Dirichlet factors, boundaries, both marginals and all histories. Loading
validates pair consistency, boundary rows and Dirichlet sufficient statistics.
Model schema version 2 also reads milestone-2 version-1 IID states. `fit`
always starts fresh; loading does not implement streaming or resuming CAVI.

## Runnable check

```sh
python examples/synthetic_hmm_mfa.py --components 1 --max-components 6
```

The seeded example concatenates three independent sequences, starts with one
component, reports clustering/bound/count statistics, checks save/load, and
saves one summary plot plus data under `agents/experiments/mfa/hmm_example`.
It also extracts MAP-selected local coordinates and plots reference versus
fitted observation covariances, matching components by nearest known centroid;
the factor bases themselves are identifiable only up to rotations.
On the CPU float64 validation run, two births recovered the three well-separated
synthetic regimes with ARI=NMI=1. This is a sanity check, not evidence for
recovery on arbitrary overlapping real data. Tests and reports remain under
ignored `agents/` as requested. Online MFA, continuous latent dynamics and a
noise posterior remain outside this milestone; CUDA is not validated here.
