# HDP presets and birth sensitivity

`HDPConfig(hdp_hyp='balanced')` selects a starting bundle according to
`hmm_switch`. This switch concerns cluster assignments and is independent of
the LDS matrix A.

## Markov assignments (`hmm_switch=True`)

These are the **current HMM implementation values**, unchanged:

| Preset | `gamma` | `transAlpha` | `startAlpha` | `kappa` |
| --- | ---: | ---: | ---: | ---: |
| `less` | 1000 | 1000 | 1 | 0 |
| `balanced` | 1000 | 100 | 1 | 0 |
| `more` | 10 | 100 | 1 | 0 |
| `max` | 1 | 0.01 | 1 | 0 |

Select a bundle with `HDPConfig(hdp_hyp='max')`, or pass
`--hdp-preset max` to a dataset runner. The name is a preset label, not a
guarantee that every dataset will produce more clusters.

`gamma` controls the global stick concentration, `transAlpha` controls
transition-row concentration around those weights, `startAlpha` affects the
initial-state distribution, and `kappa` favors self transitions. These have
different roles in the [hierarchical Dirichlet process](https://web4.cs.columbia.edu/~blei/papers/TehJordanBealBlei2006.pdf)
and the [sticky HDP-HMM](https://people.eecs.berkeley.edu/~jordan/papers/AOAS395.pdf).
Changing one does not imply a monotonic change in this implementation's birth
score. The older preset comparisons in `agents/` used earlier preset values;
their numerical ordering does not characterize the current table.

## Independent assignments (`hmm_switch=False`)

Only gamma is active in the independent stick-weight prior and linear ELBO:

| Preset | `gamma` |
| --- | ---: |
| `less` | 0.001 |
| `balanced` | 0.1 |
| `more` | 1.0 |
| `max` | 100.0 |

Larger gamma spreads prior mass over more mixture components. These
values give the names an increasing concentration
ordering; the final cluster count still depends on the observations, priors,
objective and search. They are not calibrated birth probabilities.
`transAlpha`, `startAlpha` and `kappa` keep their bundle values as inactive
fields; neither assignment weights nor the independent-mixture ELBO uses them.
The independent weight posterior remains
`q(v_k) = Beta(1 + N_k, gamma + sum_{j>k} N_j)`.

Resolution is automatic when the model is built, including for UCR runners
and LDS presets that disable HMM. No extra CLI flag or configuration system
is needed. For example, `HDPConfig(hdp_hyp='max', hmm_switch=False)` uses
gamma=100; fixing A=0 alone with HMM still enabled retains the HMM bundle.
Saved recipes retain the preset name and dependence switch; rebuilding an
older non-HMM recipe now resolves the updated mixture gamma values. Historical
fitted-state snapshots retain the hyperparameters stored in those snapshots.

## A controlled birth experiment

The package's HDP-only diagnostic compares a one-cluster 200-beat baseline
with splits of 198+2 or 180+20. It updates the baseline global stick and
transition parameters twice, then initializes the proposed newborn by the
production temporary-growth path. `startAlpha=0.1`, `kappa=0`. The entries
below are **HDP score of the birth proposal minus HDP score of the baseline**;
a larger number favors birth. GP, LDS, and warp scores are excluded.
This is an HMM transition diagnostic; it does not characterize the independent
mixture preset table above.

| Newborn beats | `transAlpha` | `gamma=0.01` | `gamma=0.1` | `gamma=1` | `gamma=10` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.01 | -20.94 | -17.97 | -12.89 | -20.93 |
| 2 | 1 | -21.27 | -18.30 | -13.29 | -21.54 |
| 2 | 10 | -22.82 | -19.88 | -15.13 | -26.20 |
| 20 | 0.01 | -20.88 | -17.90 | -12.82 | -20.85 |
| 20 | 1 | -24.02 | -21.05 | -15.95 | -22.79 |
| 20 | 10 | -38.86 | -35.91 | -30.95 | -39.21 |

This fixture penalizes all shown births. Moving `gamma` from 0.01 to 1
reduces the penalty, but moving to 10 increases it again. Larger
`transAlpha` penalizes the 20-beat split particularly strongly. The effect
depends on both concentrations and proposed occupancy. A clustering run
also includes the emission, latent, and LDS terms, so these entries cannot
predict the full decision alone.

The [full decade-grid diagnostic](hdp_birth_grid.md) records every
combination from `1e-8` through `1e8` for both concentrations. Extreme
settings exposed optimizer fallbacks and very large score excursions; finite
values there are stress-test results, not recommended prior settings. The
diagnostic characterizes the implemented variational score, not a proof of
its derivation or a data-independent birth probability.
