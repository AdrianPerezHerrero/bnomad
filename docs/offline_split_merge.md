# Offline split–merge inference

The offline optimizer is selectable. Existing configurations retain the
original birth/reallocation/deletion algorithm:

```python
InferenceConfig(schedule='offline', offline_strategy='legacy')  # default
InferenceConfig(schedule='offline', offline_strategy='split_merge',
                split_merge_refine_steps=3, n_explore_steps=10,
                split_merge_stability_patience=3, split_merge_score_rtol=1e-3)
```

These are fields of the existing typed configuration. They are serialized
with the ordinary model recipe. To change an existing recipe:

```python
from dataclasses import replace
from hdpgpc import HDPGPC

config = replace(config, inference=replace(
    config.inference, schedule='offline', offline_strategy='split_merge',
    split_merge_refine_steps=3,
))
model = HDPGPC(config).fit(values, grid=grid)
```

All dataset runners expose the same switches:

```powershell
python examples/ucr_pca_clustering.py --datasets Coffee --offline-strategy split_merge
python examples/ucr_pca_clustering.py --datasets CinCECGTorso --offline-strategy split_merge --offline-representer-selection fixed
python examples/mitbih_waveform_clustering.py --records 100 --offline-strategy split_merge --split-merge-refine-steps 5 --exploration-steps 6
```

Use `--offline-strategy legacy` to select the original algorithm. The strategy
does not choose the LDS update rule: sequential, batch, natural-gradient and
legacy LDS estimation remain separate options with their existing restrictions.
Selecting split–merge for online/online_fast is rejected.

## What changes

The new optimizer starts from the configured initial component count (one by
default), refines the baseline, then explores **partitions of groups** and
**merges of occupied clusters**. An explicitly supplied initial count greater
than one uses deterministic contiguous initial groups, capped by the number
of observations and the cluster cap.

Each outer iteration:

1. Refines the current fixed-count model with native assignment and GP/LDS
   updates, retaining the best scored state.
2. Screens every eligible parent using up to two inexpensive weighted two-means
   partitions. Ranks these by their diagonal-Gaussian mixture likelihood gain
   minus a BIC-style complexity penalty. A minority of split slots retain
   emission/latent-score exploration.
3. Ranks merge candidates by distances between responsibility-weighted feature
   centers. The ranking uses no annotation labels.
4. Fits and refines candidates in isolated states. A split initially divides
   the parent's responsibility mass; a merge adds its two columns. Soft
   assignments remain soft during subsequent native message updates.
5. Selects the best candidate whose full-data score exceeds the refined baseline
   by the existing absolute/relative tolerance. Equal and nonfinite scores do
   not produce accepted moves.

`n_explore_steps` / `--exploration-steps` bounds expensive structural candidates
per outer iteration. When both types are available, roughly half the budget is
reserved for merges. Use a budget of at least two to examine both types in an
iteration; a budget of one prioritizes a split. Duplicate split partitions are
discarded before refitting. Native configuration accepts zero to disable
structural moves. The CLI retains its existing positive exploration requirement.

The feature-space calculation only initializes/ranks proposals. Every accepted
decision uses the configured GP/LDS observation likelihood, latent scores, warp
terms and priors over the whole batch. Thus PCA, affine autoencoder, Fourier,
wavelet, inducing-point and ordinary waveform paths keep their existing
mathematical implementations. Feature ranking is a heuristic and does not
guarantee that the best possible partition is examined.

## Fast split selection

Split ranking uses common fixed coordinates independent of each cluster's
learned C, Sigma or Gamma. For fixed affine observations (PCA, the affine
autoencoder, Fourier and wavelets), the analysis map is the pseudoinverse of
the calibrated decoder, applied after subtracting its offset. For PCA this is
the PCA projection. It is used only to propose groups, not to replace Gaussian
latent inference. Other emissions use waveforms. Outputs are concatenated,
constant coordinates removed and features standardized across the entire batch.
Above 20 coordinates, a deterministic orthogonal random projection reduces
them to 20, followed by global standardization. This projection uses a local
generator and leaves inference's random state untouched. Feature construction
and affine pseudoinverses are computed once per fit; merge ranking continues
using its original standardized waveform features.

Every parent with at least two MAP members gets up to two cheap appearance partitions,
starting respectively from its worst-scoring member and the member farthest
from its weighted center. The other seed is the farthest member from that seed.
Each partition uses at most six weighted two-means steps. Complement-equivalent
partitions are deduplicated. Singleton children are allowed, including a split
of two observations into two singletons. Both children must have positive parent
responsibility mass; in soft mode a singleton may carry fractional mass. The
existing variance floor keeps a singleton's proxy covariance nonzero. Full-model
refinement and objective-based acceptance still determine whether to retain it.

### Worst latent-score window

An additional proposal separates the worst `q_lat` window from its parent.
It uses the existing output aggregation and responsibility weights, with no
emission contribution. Eligible rows must be MAP members retained by every
output's parent model. Each output's first retained state is excluded because
its score describes the initial prior rather than a transition. Missing,
nonfinite or constant transition scores do not generate a window proposal.

Window length is `max(2, floor(sqrt(parent_MAP_count)))`, capped by eligible
states and by `parent_MAP_count - 1`. Prefix sums find the lowest weighted
average in linear time. Windows follow the parent's ordered retained-state
clock; adjacent states need not have adjacent global observation indices.
The proposal moves the parent's responsibility inside the selected interval
to the newborn, leaving other clusters' responsibilities intact. This also
routes fractional parent responsibility inside the interval in soft mode.
The newborn seed is the lowest-`q_lat` member of that window. The usual
`offline_representer_selection` policy still governs representative selection.

This is a temporal partition, not another appearance two-means initialization.
It can propose a coherent poor-fitting interval even when its individual
observations overlap other regimes in appearance. Native fitting recomputes
both children's histories; subsequent refinement can reassign observations.
`q_lat` contains state entropy as well as transition fit, so this is a search
heuristic, not a separate dynamics likelihood or an ELBO gain estimate.

### Screening and proposal budget

For each partition, parent weights `w_t = r_tk` over those MAP members supply
the count, first moment and diagonal variance of one Gaussian and each child.
Every fitted variance is floored at 0.01 in globally standardized coordinates,
equivalent to 1% of global feature variance. The same floor is used in both
models. The ranking gain is

```text
n = sum(w_t); pi_j = n_j / n
ell_one = sum_t w_t log N(z_t | mu, diag(v))
ell_two = sum_t w_t log(sum_j pi_j N(z_t | mu_j, diag(v_j)))
penalty = (2*p + 1)/2 * log(n)
proxy_gain = ell_two - ell_one - penalty
```

Here `p` is the retained feature dimension. A single diagonal Gaussian has
`2*p` fitted parameters; two plus one mixture weight have `4*p+1`. Mixture
densities use log-sum-exp, not separate hard-assignment log likelihoods. The
parameter estimates come directly from the tentative partition, without an
extra mixture-EM loop. With soft weights, use of responsibility mass in the
penalty is heuristic. This is a BIC-inspired ranking proxy, not an LDS ELBO,
exact Bayes factor or proof of multimodality. Projected/diagonal features can
miss structure, and a two-Gaussian fit can also favor some non-Gaussian unimodal
distributions.

The total gain is used, not gain per observation, so recurring structure in a
large parent can outweigh isolated outliers. When latent windows are available,
up to `max(1, split_slots // 3)` slots are reserved for them, ranked by the gap
between the parent's weighted mean transition score and its worst window mean.
Even with one split slot, a valid latent-window proposal receives that slot.
The remaining slots use the appearance proxy, reserving
`max(1, appearance_slots // 5)` for the previous parent-loss criterion when
there are at least two appearance slots. Spare appearance slots can try more
latent windows. Complement-equivalent partitions across both families are
deduplicated. Negative proxy gain does not veto a full-model trial: this ranking is
only a shortlist. Invalid or too-small children are excluded from both routes.

The cheap screening uses O(N*p) work per bounded two-means pass over MAP groups,
without GP/LDS fitting, matrix factorizations per parent or HDP updates. Only
shortlisted candidates allocate full proposal states. Split descriptions in
debug logs and `split_merge_history_` include `selection`, `proxy_gain`,
`proxy_likelihood_gain`, `proxy_complexity_penalty`, `proxy_child_mass` and
`proxy_dimension`. Actual proposal refinement still uses all responsibility
mass and the original observations, scores and matrices. Full-model acceptance,
cycle prevention, merge policy and the legacy optimizer are unchanged.
Latent proposals additionally record `selection='q_lat_window'`,
`proposal='worst_q_lat_window'`, `window_start`, `window_end`, `window_length`,
`window_q_lat_mean`, `parent_transition_q_lat_mean` and `window_q_lat_gap`.
Bounds and seeds use zero-based global observation indices.

## Objective and posterior reuse

The implementation calls `compute_q_elbo` for emission, latent, HDP and LDS
terms, and adds the existing batch chain entropy **once**. Candidate-specific
HDP weights/start/transition factors are recomputed before comparison. Shared
LDS policies pool all candidate cluster statistics using their existing update,
and their KL is counted once by the existing scorer.

The experimental `birth_lds_kl_per_dimension` policy is respected for both split
and merge decisions and for within-branch refinement. With it enabled, the
selection objective is the dimension-adjusted score, not a strict ELBO. The
unscaled objective is separately retained in `elbo_last`, iteration history and
proposal traces. With normalization disabled, both scores coincide. Historical
LDS scoring retains its existing mean-over-history semantics.

Both proposal and baseline receive local refinement. A proposal is allowed to
refine even if its initial score is below the baseline; there is no first-pass
score gate. `split_merge_refine_steps` controls additional refinement passes
after fitting a proposal. Conditional batch LDS fits can warm-start filtering
with previous matrices while recomputing parameter posteriors from their
original priors. A parent's posterior is never installed as a newborn's prior.

In this strategy, native message passing receives the combined local potential
`static_factor * q + dynamic_factor * q_lat`, after the existing output weighting.
This matches the pointwise terms of the selected objective with current
GP/LDS factors held fixed. The legacy optimizer's emission-only message input
is preserved in its separate implementation.

Pair statistics are transformed along with responsibilities. If
`a_t` allocates parent mass to a child, its pair contributions are multiplied
by `a_(t-1)` on the source axis and `a_t` on the destination axis; the complementary
parent contributions use `1-a`. A merge sums the relevant source and destination
axes, including both cross-transitions. Row zero remains empty. Native soft
forward/backward and pair-message calculations then refine these statistics.
Hard assignments use adjacent hard labels to keep transition counts coherent.

## Cost and state ownership

Independent assignments bypass forward/backward, and compatible fixed-zero A
affine models use batched state conditioning and scoring. See [automatic fast
paths](independent_fast_paths.md). Proposal refinement remains global; these
optimizations do not restrict outside observations from moving into a child.

- Candidate GP kernels reuse their fitted parent kernels. The first fit retains
  the configured GP fitting behavior and initial cross-output kernel sharing.
- Unaffected component scores/posteriors are reused when their responsibilities
  and alignment references are unchanged.
- Batch LDS refinement and shared learned policies require additional refits
  because latent moments and parameters depend on one another.
- DTW-GP input-warp posteriors are reused when observation references and
  inherited warp hyperparameters are unchanged. Only missing references invoke
  the existing alignment computation. Generic GP alignment keeps its native
  cache; stochastic GP-flow posteriors are recomputed.
- Proposal lists/settings/caches and model-owned parameter policies are isolated.
  Rejected branches cannot update the retained cluster models or shared matrices.

There is no whole-model deep copy or separate GP/LDS implementation. Necessary
emission copies use `gpmodel_deepcopy`; shared policies and active warpers receive
independent ownership. Stochastic warp inference still produces estimated
objectives, so deterministic monotonic true-ELBO claims do not apply to that case.

## Stopping and diagnostics

The optimizer stops after `split_merge_stability_patience` consecutive outer
iterations (default 3) satisfying all of the following:

- No accepted split or merge.
- Full state and pair responsibility arrays retain their shapes and are equal
  within absolute tolerance `1e-6` (zero relative tolerance). Unchanged occupancy
  totals or MAP labels alone do not suffice for soft assignments.
- The absolute change in the selected score is at most
  `1e-6 + split_merge_score_rtol * max(1, abs(previous_score), abs(score))`.
  The default relative tolerance is `1e-3` (0.1%).

Any failed condition resets the stable-iteration count. Thus small continuing
parameter improvements with stationary assignments do not keep a settled
clustering running to the iteration limit. This is an approximate clustering
stopping rule; it does not claim the parameter posterior has fully converged.
It is assessed after the current iteration's structural candidates have been
examined. Inner proposal refinement retains its existing stricter rule.

Tune the outer rule with `--split-merge-stability-patience` and
`--split-merge-score-rtol` in any dataset runner. Both are fields of the ordinary
`InferenceConfig`; patience must be positive and the tolerance finite and
nonnegative. A smaller tolerance or larger patience allows more fitting.
The selected score is used, including configured experimental LDS normalization;
the raw unscaled objective is still reported separately.

The configured outer iteration limit remains authoritative. Reaching the
cluster cap disables further splits, **not merges or refinement**. An accepted
split or merge resets the plateau count; inference cannot stop for convergence
on that iteration. The legacy optimizer's stopping rule is unchanged.

Progress logs print the final reason and score change; debug verbosity also
shows the assignment-stability check, threshold and patience counter on every
iteration. `model.split_merge_convergence_` exposes these final diagnostics,
the stopping iteration and reason (`stable_assignments_and_small_score_change`
or `iteration_limit`). Dataset reports save them as `split_merge_convergence`
in `results.json`. Legacy fits return an empty dictionary.

Structural moves also have a partition-cycle safeguard. The optimizer remembers
the initial partition, refined baselines and accepted partitions during this fit.
A split or merge returning to one of these partitions is rejected, even if extra
parameter fitting produces a higher score. Membership keys are invariant to
cluster numbering and use every observation's MAP assignment, not just cluster
sizes. This prevents split/merge reversals disguised by slot renumbering or
different representers. Keys are checked before fitting to avoid wasted work,
and again after refinement in case assignments return to an earlier partition.

Fixed-count parameter and responsibility refinement continues normally. With
soft assignments, this guard deliberately treats states with the same MAP
partition as the same structural configuration even when soft weights differ.
It is a search restriction: previously visited partitions cannot be revisited
by a structural move during the same fit, and approximate cycles with different
memberships are not detected. Rejected/unselected proposals are not added to the
visited set. Storage is one compact digest per retained partition; no posterior
copies are kept for this safeguard. Cycle rejections record their phase and the
earlier iteration in `split_merge_history_` and print a reason in debug verbosity.
The legacy optimizer is unaffected.

Existing labels, representers, `q`, `q_lat`, SNR weights and final-state caches
retain their usual layouts. Merge slots are removed consistently across models,
responsibilities, pair tensors, representers, warpers and per-cluster settings.
The `sorted`/`fixed` representer policy is retained; a split's first warp uses a
member of the new child as its proposed reference. Sorted representative
selection uses current cluster scores; fixed selection preserves distinct seeds
with the existing fallback rule.
If fixed references have moved outside their clusters and greedy selection
cannot give an occupied cluster a distinct representative, split/merge retries
with stale references cleared. References still owned by their cluster are
retained, and the others are selected by score within their current MAP groups.
This exceptional repair protects singleton groups without rejecting an otherwise
valid proposal. Ordinary successful fixed-reference selection is unchanged;
debug logs report the repaired cluster slots and resulting references.
All dataset runners expose `--offline-representer-selection sorted|fixed`
(default `sorted`), and print the selected mode in the initial configuration
summary. Fixed mode retains the newborn's initial center-nearest representative;
it does not change how the initial split seed is chosen.

`model.split_merge_history_` returns detached candidate scores, component terms,
accepted/rejected decisions and numerical rejection reasons. Dataset reports
save this list in `results.json`. Progress verbosity prints responsibility sums
and accepted moves; debug adds score components and proposal changes. Silent
mode prints only the dataset runner's final results.

Existing optional birth/warp diagnostic capture and plotting are reused for
splits where that capture is supported. Merges appear in score history and final
plots; they do not create birth-specific warp figures. No guarantee of a global
optimum or lower total runtime is implied: wider search and more refinement
cost more GP/LDS passes.
