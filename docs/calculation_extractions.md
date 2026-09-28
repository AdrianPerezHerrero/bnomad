> Historical 0.2 evidence. Source paths and compatibility instructions below describe the archived release, not 0.3. Use [the current architecture](architecture.md), [migration guide](migration.md), and [0.3 validation report](migration_0_3/report.md) for this release.

# Faithful calculation extractions

This change extracts existing expressions into private functions with explicit
inputs and outputs. It is category D (structural); the mathematical operations,
normalization, stabilization, inference ordering, and cache ownership are not
being redesigned. The starting revision is `56f5d9b`.

## Selected boundaries

| Logical extraction | Calculation and new location | Existing caller retained | Reason for this boundary |
|---|---|---|---|
| 1 | `numerics._normalization.max_shift_log_scores` | `GPI_HDP.LogLik` | Stateless NumPy/PyTorch maximum shifting, reusable outside a model; not log-sum-exp. |
| 2 | `hdp._entropy.state_entropy_terms`, `pair_entropy_terms` | `GPI_HDP.calc_Hstart`, `calc_Htable` | Assignment-specific objective terms, so they belong to HDP rather than generic numerics. |
| 3 | `alignment._covariance.warp_prior_rbf_covariance` | `WarpPriorAMTGP._rbf_cov` | Explicit grid, prior parameters and stabilization; no cache ownership. |
| 4 | `alignment._grids.monotone_grid_from_expanded_controls`, `batch_monotone_grid` | Scalar method and batch closure | Warping-specific transforms with distinct interpolation and axis conventions. |

The original methods remain compatibility entry points. New modules use leading
underscores and are not added to the public `__all__` lists. Bare namespace imports
remain lightweight. No model lifecycle or state is moved into these helpers.

## Contracts that must not be normalized away

- Maximum shifting retains the exact `type(x) is torch.Tensor` dispatch, axis
  branches, NumPy float64 scalar special case, and whole-input bypass for any
  infinite maximum. The bypass returns the same input object. Unsupported shape
  behavior is not broadened.
- `calc_Hstart` sums all supplied responsibility rows despite its name. Pair
  entropy divides by a row sum plus epsilon, then adds epsilon again before an
  in-place logarithm of its newly allocated temporary. Input arrays are not mutated.
- RBF prior coordinate normalization and grid transforms retain their own
  denominators and jitter policies; they are not generic covariance utilities.
- Scalar and batch alignment are not merged into one implementation. Their
  coordinate interpolation and reduction axes are distinct existing behavior.

No algebraic simplification, stabilization replacement, vectorization, casting,
detach, clone, or new validation is introduced by these extractions.

## Focused test strategy

`tests/test_calculations.py` exercises values, shape/dtype preservation, mutation
and alias behavior, nonfinite/empty inputs, and gradient retention where supported.
NumPy entropy expectations come from small explicit probability examples. The
normalization gradient has an analytic expected value.

`tests/fixtures/calculation_reference.json` contains seven small reference cases
captured from the unchanged AMTGP implementation before its extraction: four RBF
cases, one scalar grid and two batch grids. It records the original module hash,
revision and Torch version. The batch nested function was lifted via AST with its
original closure inputs; no training or substitute formula generated its expected
values. Tests read this file; they never regenerate it. These are separate focused
fixtures, not replacements for the workspace characterization baselines.

## Deferred candidates

Expected-log transition formulas remain at their callers: different paths use
digamma-of-sums or log-of-exponentiated-digamma sums and different CPU conversion
boundaries. They must not be unified under a conventional formula. Emission
scores remain coupled to their cached identity/factorization policies and their
omitted determinant convention. Matrix stabilization remains local. The linear
HDP objective also mutates a supplied count array in its current slack calculation;
that needs an explicit mutation contract before extraction. None of those paths
is changed here.

## Validation log

The full workspace characterization suite must pass between logical extractions.
Focused tests run first; golden characterization files remain unchanged.

| Stage | Focused calculation tests | Characterization tests |
|---|---|---|
| Baseline | Not yet introduced | 15 passed, 32.061 s |
| 1: maximum shift | 7 passed | 15 passed, 30.353 s |
| 2: HDP entropy | 12 passed | 15 passed, 30.207 s |
| 3: prior covariance | 15 passed, 1 CUDA skip | 15 passed, 29.840 s |
| 4: monotone grids | 19 passed, 1 CUDA skip | 15 passed, 30.194 s |

The two import-boundary tests also passed after the last extraction (7.570 s).
CUDA is unavailable in the reference environment; GPU covariance checks are
explicitly skipped and no GPU equivalence claim is made.

An AST audit against revision `56f5d9b` compared all six calculation bodies after
replacing the specific `self` attributes with explicit arguments (and leaving
scalar coordinate expansion in its caller). Arithmetic statements and ordering
were identical. Alignment reference values and gradients matched exactly on CPU;
characterization passed at its unchanged tolerances (`rtol=1e-7`, `atol=1e-9`).
No detected numerical differences required a baseline update. Sub-tolerance
end-to-end differences were not measured separately.

Timing is whole-suite wall time, not an optimization benchmark. No performance
improvement is claimed. The extraction adds ordinary Python calls around the
same numerical operations.

From the repository root, run focused and import tests with:

```text
python -B -m unittest discover -s tests -p test_calculations.py -v
python -B -m unittest discover -s tests -p test_package_structure.py -v
```

Run the unchanged characterization suite from the surrounding workspace:

```text
python -B -m unittest discover -s tests/characterization -t . -q
```
