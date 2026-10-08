# HDP presets and birth sensitivity

`HDPConfig(hdp_hyp='balanced')` selects a starting bundle for the HDP
transition model. These are the **current implementation values**:

| Preset | `gamma` | `transAlpha` | `startAlpha` | `kappa` |
| --- | ---: | ---: | ---: | ---: |
| `less` | 1000 | 1000 | 1 | 0 |
| `balanced` | 1000 | 100 | 1 | 0 |
| `more` | 10 | 100 | 1 | 0 |

`gamma` controls the global stick concentration, `transAlpha` controls
transition-row concentration around those weights, `startAlpha` affects the
initial-state distribution, and `kappa` favors self transitions. These have
different roles in the [hierarchical Dirichlet process](https://web4.cs.columbia.edu/~blei/papers/TehJordanBealBlei2006.pdf)
and the [sticky HDP-HMM](https://people.eecs.berkeley.edu/~jordan/papers/AOAS395.pdf).
Changing one does not imply a monotonic change in this implementation's birth
score. The older preset comparisons in `agents/` used earlier preset values;
their numerical ordering does not characterize the current table.

## A controlled birth experiment

The package's HDP-only diagnostic compares a one-cluster 200-beat baseline
with splits of 198+2 or 180+20. It updates the baseline global stick and
transition parameters twice, then initializes the proposed newborn by the
production temporary-growth path. `startAlpha=0.1`, `kappa=0`. The entries
below are **HDP score of the birth proposal minus HDP score of the baseline**;
a larger number favors birth. GP, LDS, and warp scores are excluded.

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
