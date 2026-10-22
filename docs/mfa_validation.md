# MFA public API and validation

The offline MFA extension is available through explicit imports:

```python
from hdpgpc import HDPMFA, HDPMFAConfig
from hdpgpc.config import HDPConfig, NumericalConfig
from hdpgpc.model import MFAFactors

config = HDPMFAConfig(
    n_factors=2,
    hdp=HDPConfig(M=5, max_models=10, hmm_switch=False, hdp_hyp='more'),
    numerics=NumericalConfig(seed=42, num_threads=1),
    loading_prior_precision=0.1, noise_initial=0.4,
    max_iter=200,
)
model = HDPMFA(config)
labels = model.fit_predict(X)  # X: (N, observed_features); labels: (N,)
factors = model.transform_components(X)  # MFAFactors: mean (N,K,r), covariance (K,r,r)
```

Set `hdp.hmm_switch=True` for discrete Markov regimes and supply
`sequence_lengths` to fitting and queries. Continuous factors still have
independent standard-normal priors. See [DP-MFA mathematics](dp_mfa.md),
[HDP-HMM-MFA](hmm_mfa.md) and the [FA component](factor_analysis_emission.md).

## Typed contract

Public methods and properties have concrete return types and input annotations.
Lazy exports expose those declarations to type checkers without changing the
existing GP root `__all__` contract. The distribution includes `py.typed`.
Results are CPU NumPy snapshots, even when an explicitly configured model runs
on another supported device; the validated execution target is CPU float64.

| Operation | Result |
| --- | --- |
| `fit(X, sequence_lengths=None)` | Fresh fit, returning the same facade |
| `fit_predict(X, sequence_lengths=None)` | Stored training MAP labels, without a second assignment update |
| `predict_proba(X, sequence_lengths=None)` | Soft query singleton marginals, shape (N,K) |
| `predict(X, sequence_lengths=None)` | MAP of query marginals; HMM labels are not Viterbi decoding |
| `transform_components(X)` | Separate component-conditioned coordinates and covariances |
| `transform(X, assignment='map', sequence_lengths=None)` | One selected local basis per row, shape (N,r) |
| `score_samples(X, sequence_lengths=None)` | Normalized plug-in log-density contributions, shape (N,) |
| `score(X, sequence_lengths=None)` | Mean of `score_samples`, rejecting an empty sample matrix |
| `variational_score_samples(X, sequence_lengths=None)` | Local variational log-partition contributions, excluding global KLs |
| `n_features_in_`, `n_factors_` | Fitted observation dimension and local factor rank |
| `state_dict()`, `from_state_dict(data)` | Data-only fitted-state snapshot and restoration |
| `save(path)`, `load(path)` | Atomic JSON save and validated restore |

Array-valued outputs are independent, read-only snapshots. Histories/configs
are independent copies. `fit` is transactional: a failed new fit preserves a
previously fitted model. Querying never updates fitted parameters. HMM query
lengths default to one new sequence; training boundaries are not silently reused.
`score_samples` accepts empty queries; their mean `score` is undefined and raises.

For IID models, scores use mean loading parameters and conditional mean stick
weights. For HMM models, they are filtered predictive increments using mean
start/transition parameters normalized over represented destinations. Summing
these increments gives the plug-in sequence likelihood. Smoothed assignment
marginals and filtered density increments answer different questions.
Neither is an exact parameter-integrated Bayesian predictive density.

Local factor bases are identifiable only up to rotations, and cluster IDs only
up to permutations. Pruning compacts slot IDs. Do not average unrelated local
coordinates or assess recovery by comparing raw loading entries.

## Recipes and fitted-state persistence

```python
recipe_text = config.to_yaml()  # stable JSON-form YAML; no YAML dependency needed
config_copy = HDPMFAConfig.from_yaml(recipe_text)
recipe_dict = config.to_dict()

model.save('mfa.json')
restored = HDPMFA.load('mfa.json')
# Or HDPMFA.from_state_dict(model.state_dict()).
```

The recipe reuses the existing typed HDP/numerical sections and data-only codec;
general block YAML requires the existing optional PyYAML extra. Model schema 2
also reads schema-1 IID snapshots. Shared learned noise is stored exactly once.
Loading checks dimensions, exact array shapes, finite positive noise, priors
against the recipe, occupancies, posterior KLs, objective histories and HMM pair
consistency/counts. Invalid JSON data, cycles, unsupported schemas, inconsistent
state and redundant cluster noise are rejected. No arbitrary class import,
pickle execution, reinitialization, refitting or RNG reset occurs during load.

Saving writes a temporary file in the destination directory and atomically
replaces the target. A failed replacement preserves the old file and cleans up
the temporary file. The parent directory must exist. Serialization preserves
results and prediction, not a streaming or resumable-optimization contract.

## Standard finite-MFA reference

The validation reference is an independent finite, maximum-likelihood MFA in
`hdpgpc.experiments._mfa_reference`. It is an experimental comparator, not another
HDP implementation or production facade. It follows joint centroid/loading EM
with shared diagonal noise, as derived in Appendix B of
[Ghahramani and Hinton, CRG-TR-96-1](https://www.cs.utoronto.ca/~hinton/absps/tr-96-1.pdf).
It uses the same rank and variance floor as the Bayesian comparison, normalized
Gaussian densities and rank-sized Cholesky solves. It does not call the Bayesian
FA posterior or HDP code.

For current responsibilities and Gaussian factor moments, with u=[1,z]:

\[
F_k=\sum_i r_{ik}E[u_i u_i^T],\quad
G_k=\sum_i r_{ik}x_iE[u_i]^T,\quad
B_k=G_kF_k^{-1},\quad \pi_k=N_k/N.
\]

The shared variance is pooled expected reconstruction residual divided by N,
clipped to the declared positive floor. Several initializations are selected
by training likelihood only. There is no loading prior, posterior covariance,
parameter KL, DP tail or HMM in this finite reference.

Tests check Gaussian densities/conditioning against dense references, the joint
M-step against explicit expected residuals, EM monotonicity, reproducibility,
and the single-component likelihood against scikit-learn FactorAnalysis.

## Reproducible comparison

```sh
python examples/synthetic_mfa_comparison.py --seeds 7 19 42
# Short run:
python examples/synthetic_mfa_comparison.py --cases separated --seeds 7
# Also available from a wheel, outside the source checkout:
python -m hdpgpc.experiments.mfa --cases separated --seeds 7
```

The generator draws disjoint training/test sets from known low-rank Gaussian
components. Cases cover separated means, overlapping components, 48-dimensional
observations, and independently restarted Markov sequences. The compared fits:

- Bayesian DP-MFA with fixed support;
- Bayesian DP-MFA with objective-gated birth/pruning, initially one component;
- HDP-HMM-MFA on the ordered case;
- finite MFA with the known generative K, explicitly labeled **oracle K**;
- finite MFA with the same support size as fixed-support DP-MFA;
- finite MFA selected by training BIC over K=1,...,support.

The oracle uses the generator's K, not labels in parameter learning. BIC is the
conventional parameter-count heuristic with factor-rotation redundancy removed;
mixtures are singular models, so its asymptotic justification is limited.
Default reference fitting uses three starts; Bayesian fits use one seeded
initialization. Runtime and initialization budget are reported rather than
treated as equal. HDP-HMM uses the existing sticky preset, so that ablation
changes the discrete prior as well as the use of sequence information.

All fitting, initialization selection and K selection use training observations
only. Tests explicitly change test values/labels and verify fitted parameters
and selected K stay unchanged. Labels serve evaluation and component matching.

**Compare held-out normalized densities, ARI/NMI, occupied K, covariance/mean
recovery and runtime. Do not rank the Bayesian training ELBO against finite-MFA
training log likelihood.** On the Markov case, a common IID emission-mixture
diagnostic is reported separately from each model's own sequence density.
The HMM common IID diagnostic uses its mean global stick masses; it is not a
stationary Markov marginal. Covariance metrics match components using training
contingencies and report unmatched regimes/components explicitly. Perfect
partition recovery does not imply perfect parameter recovery.

Output defaults to `agents/experiments/mfa/comparison/`, with content-derived
run folders, settings/provenance, logs, results JSON, model JSON and NPZ arrays.
The root has `comparison.csv`, `comparison.json` and `aggregate.json`. Per-run
plots show common display-PCA projections, separate training objectives,
held-out densities, covariance heatmaps and, for ordered data, regime sequences.
Display PCA is fitted on training features and is not used by inference.

## Validation and limitations

Workspace tests/validation scripts remain under ignored `agents/tests/mfa/`
as requested. Run:

```sh
python -m unittest discover -s agents/tests/mfa -p 'test_*.py' -v
python agents/tests/mfa/validate_gp_mfa_regression.py
python agents/tests/mfa/verify_mfa_artifacts.py
```

The GP harness compares six existing test-suite outcomes with a fresh archive
of untouched HEAD, then compares live public/scientific GP outputs numerically.
Pre-existing fixture failures are reported, not counted as new passes. Frozen
fixtures are not regenerated. Artifact checks build wheel/sdist, verify typing
and MFA module contents, and test noneditable installation in a new environment
outside the checkout. The offline artifact check shares existing dependency
files and uses the local build backend; it does not validate fresh dependency
resolution or a complete OS/Python build matrix.

The reported validation target is CPU float64. Annotation-contract checks are
not a claim that the entire research package passes a static type checker.
MFA remains offline; no continuous latent LDS, nonlinear observation model,
warping, Bayesian noise posterior, stochastic inference or validated CUDA path
is added by this milestone. Birth/pruning is a bounded local search and can
remain at one component when its proposal screening/refinement is insufficient.
