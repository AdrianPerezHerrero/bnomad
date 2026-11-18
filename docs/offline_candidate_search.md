# Offline candidate search

`InferenceConfig.offline_candidate_search` selects a screening policy separately
from `offline_strategy='legacy'|'split_merge'`. Both optimizers default to
`'combined'`; online schedules ignore this offline control.

| Policy | Screening rule |
|---|---|
| `score_guided` | Original poor-fit score search: simple emission scores and combined emission/latent scores select birth seeds in the legacy optimizer; standardized feature partitions, fit losses and latent-window hints guide split/merge. |
| `noise_aware` | Calibrated affine information coordinates discount common orthogonal density and weak variance directions, preserving the existing latent-score hints. |
| `dynamics` | Local one-versus-two conditional transition fits, including innovation covariance and the intercept when enabled; native worst-latent-score windows supplement the operator candidates. |
| `combined` | Independently ranked family pools, with quotas for `score_guided`, `noise_aware` and `dynamics`. This is the default. |

The previous candidate name `legacy` is accepted as an alias for `score_guided`,
including in saved configurations. New configurations normalize it to
`score_guided`. The optimizer name `offline_strategy='legacy'` is unchanged.

```bash
python examples/mitbih_autoencoder_clustering.py --records 207 \
  --offline-strategy split_merge --offline-candidate-search combined \
  --proposal-family-weights 1 1 1 --dynamics-clock physical-clock
```

```python
from hdpgpc.config import InferenceConfig

inference = InferenceConfig(
    schedule='offline', offline_strategy='split_merge',
    offline_candidate_search='combined',
    proposal_family_weights=(1., 1., 1.), n_explore_steps=12,
)
```

## Quotas and fit budgets

`proposal_family_weights`, or `--proposal-family-weights SCORE NOISE DYNAMICS`,
sets relative quotas in that order. The default `(1, 1, 1)` divides slots equally.
`(1, 1, 2)` gives dynamics half of a sufficiently large pool; zero disables that
family in the combined policy. Weights must be finite, nonnegative and have
positive total. Selecting an individual policy ignores these relative weights.

Each family ranks its own proxy scores. Scores from different proxies are never
added or compared numerically. Partition duplicates are removed before fitting;
legacy births also deduplicate representative seeds. Slots left by duplicates,
unavailable families or exhausted pools are reassigned to enabled families.
Actual use can therefore differ from requested quotas. If fewer slots than
enabled families are available, the starting family rotates with outer iteration.

The total expensive-candidate limit remains `n_explore_steps`. Split/merge first
reserves up to half for available unique merges and divides the remaining split
budget among families. Merge proposals have a separate family allocation;
unused split slots can examine additional merges. With 12 steps and six reserved
merge slots, splits can use two candidates per family. The legacy optimizer
uses its birth budget directly, allowing four per family. Its pre-existing
reallocation-first schedule and convergence rules remain in force.

Candidates that hit an existing partition-cycle guard can consume a screening
slot without a refit; existing cluster caps and topology constraints also apply.
Native configuration accepts zero exploration steps to disable structural moves.

`model.offline_search_diagnostics_` reports available families, screening
dimensions, weights, requested/used quotas, duplicates and unused slots per
search round. Dataset `results.json` saves these diagnostics. Split/merge history
also identifies each proposed candidate's `candidate_family`.

## Dynamics screening

The screen uses fixed common observation coordinates. Affine observations supply
their full noise-whitened information coordinates before the appearance variance
screen. GP observations use a fixed waveform frame. Lag-one and lag-two moments
select at most eight screening directions, avoiding a dependence exclusively on
high marginal variance. This compact projection affects only candidate screening;
the fitted LDS and emission retain their full configured dimensions.

Within each parent, cached weighted transition statistics support a bounded grid
of cuts and windows. The proxy compares a regularized conditional Gaussian
regression against two local regressions, with a BIC-style complexity penalty.
The regression includes an intercept only when `constant_input=True`. Innovation
covariance is included, so differences in transition noise can inform screening.
Merge ranking compares pooled and separate conditional fits in the same frame.
Learned per-cluster A/C matrices are never directly compared across latent bases.

Cluster-clock transitions connect consecutive observations actually retained by
the parent. Physical-clock screening uses only adjacent observed global ticks;
gaps are excluded, never interpreted as one physical transition. Sparse parents
can therefore lack operator candidates. Native latent-score windows provide
additional candidates, and combined quotas can reassign unused slots. Ordered
topology allows only valid time-boundary splits and neighboring merges.

Crossing transitions are excluded from both one- and two-operator comparisons,
so their likelihoods use the same data. Soft endpoint responsibility products
are proxy weights, not claimed posterior pair expectations. The regressions
operate on noisy encoded observations; they are auxiliary proposal heuristics,
not exact latent posterior estimates or a new inference objective.

## Noise-aware screening and compatibility

The [affine screen](offline_split_merge.md#fast-split-selection) uses calibrated
decoder/noise information and a Gaussian reference threshold. Its weak-signal
fallback and native latent-score hints are retained. Without a common affine
decoder this family is unavailable: individual `noise_aware` selection retains
the original score-guided fallback, while `combined` redistributes its slots.

Combining original score-guided candidates with noise-aware candidates broadens
the search but does not preserve strict invariance of the complete candidate
list to irrelevant observation coordinates. The noise-aware family's own
calculation retains the tested invariance properties.

Every accepted move still uses native posterior updates and the complete
configured acceptance score, including the existing LDS-KL/d policy. Inputs,
priors, emission learning, clocks, warping and dynamics update presets retain
their existing mathematical implementations and supported combinations. Default
combined selection can change results and runtime by trying different candidates
within the same budget. Select `score_guided` to reproduce original candidate
selection, or `noise_aware` to retain the preceding affine screening policy.
