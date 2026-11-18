# Fast sigma/gamma prior screening

Add `--screen-priors` to a shared experiment command to run a small, label-free
prior study and stop **before the full-data clustering fit**. The usual model
and experiment behavior is unchanged when the flag is absent. Screening never
changes model defaults or re-estimates priors midway through a production fit.

```sh
python examples/mitbih_autoencoder_clustering.py --records 203 \
  --components 10 --dynamics-clock physical-clock --latent-equilibrium \
  --sigma-mode cluster_diagonal --screen-priors
```

The command prints a recommended pair as ordinary arguments, for example
`--sigma-prior ... --gamma-prior ...`. To run the complete experiment, remove
`--screen-priors` and add the printed pair. Keep the representation, calibration,
dimension, clock, topology, prior strengths and other scientific settings fixed.
The same option is available for the MIT-BIH/Daphnet, UCR, expression, flutter,
XJTU-SY, CAS-Colon and synthetic-image shared runners. Each selected record and
configuration is screened independently. The usual output directories, content
identities, `--resume`, and master `--no-plots` switch apply.

## Budgets and artifacts

Defaults use at most **128 training rows and 32 validation rows per clip**, five
independent clips distributed across the input, two offline iterations and one
exploration step per fit, at most **60 fits**, and a **120-second** screening
budget. Five non-overlapping clips cover up to **800 distinct observations**
(640 training and 160 validation); small records use shorter or overlapping clips.
Each active prior axis has **5 logarithmically spaced values from
1e-3 to 10.0**, giving a nominal **25-pair grid** when both priors are active.
The fit budget includes adaptive refinements and repeat checks; it is not a
count of distinct prior pairs. Range corners are tested early to
preserve coverage when the fit or time budget stops the grid search.
These are pilot budgets, not new model defaults. Model capacities, HDP/transition
policies, covariance families, update schedule and latent dimensions are retained.
Offline split/merge branch refinement is limited to one pass for screening.

```sh
python examples/mitbih_pca_clustering.py --records 203 --components 10 \
  --screen-priors --screen-samples 48 --screen-iterations 2 \
  --screen-max-fits 16 --screen-seconds 20 \
  --screen-grid-points 5 \
  --screen-sigma-range 0.001 10.0 --screen-gamma-range 0.001 10.0
```

The time budget is **checked between fits**: an in-progress native fit may finish
after the deadline. Data preparation, encoder training and plotting are outside
this timer. The representation is calibrated once by the runner, never once per
candidate. A very high-dimensional waveform model or expensive native fit can
still dominate the wall time. Fixed-work comparisons should allow enough seconds
and hold `--screen-max-fits` fixed; a tight time cutoff can change which candidates
are reached across machines. Small datasets use shorter clips and need at least
eight input rows. Clip fits restart their own priors and are never concatenated
across distant time positions.

Saved files are:

- `prior_screen.json`: calibration diagnostics, grid axes and work reserves,
  clip boundaries, every attempted pair/fit, failures, losses, assignment
  agreement, covariance-growth diagnostics,
  tested working pairs and the recommendation's confirmation coverage.
- `prior_screen.csv`: compact candidate scores and status.
- `prior_screen.png`: the left panel colors tested sigma/gamma pairs using only
  their first-clip validation NLL at the initial seed. Colors never change when
  confirmation fits are added. The right panel separately lists promoted
  candidates' across-clip/seed mean NLL, descriptive standard error, seed
  assignment agreement and coverage, highlighting the selected pair and
  explaining the selection threshold. Missing first-clip scores appear as gray
  crosses. Untested combinations are not interpolated; omitted with `--no-plots`.
- `config.yaml` and the usual run provenance: the supplied full-run recipe, not
  a claim that a full fit occurred. `results.json` has `mode=prior_screen` and
  `fitted=0`; no full-record `assignments.csv` is generated.

Candidate errors remain visible. If no pilot validates successfully, the screen
reports an error rather than inventing a recommendation. A budget-limited result
can be unconfirmed or partially confirmed; inspect `confirmation_complete` and
the evaluated blocks/seeds before treating it as a robust interval.

## Calibration and refinement

The screener builds one initial emission to resolve the actual noise units. It
estimates random-walk difference moments on individual contiguous clips in that
emission's coordinates. Negative/invalid moment components remain diagnostics;
they are not made positive with absolute values. Regime changes and nonidentity
dynamics can bias even positive moments, so the proposed centres are restricted
to within sqrt(10) of the supplied scales. The supplied pair is also tested when
it lies inside the search bounds. Static/independent configurations with no active
process-noise argument do not waste a gamma grid.

A five-by-five logarithmic grid uses inclusive default bounds of **1e-3 to 10.0
for both sigma and gamma**. Adjacent values differ by a factor of approximately
10 (one decade). Change the density with `--screen-grid-points` and the
total work with `--screen-max-fits`; increasing density does not raise the fit
budget automatically. The supplied anchor and range corners precede the
remaining grid pairs, which are ordered by calibration distance. Up to four
local refinements use half the grid's logarithmic step (a factor of approximately
3.16 by default). Both refinement and the anchor respect the search bounds;
explicit `--screen-sigma-range` and `--screen-gamma-range` replace those bounds.
Fit and time budgets reserve work for promotion of up to three survivors to
another clip and another seed. Scientific configuration and decoder changes between candidate factories
are rejected. The sigma/gamma arguments keep the runner's original units: they
can be covariance multipliers or absolute variances depending on representation.

## What validation measures

Each pilot uses the actual bnomad inference code on its training clip. Validation
then freezes the fitted parameters and scores each subsequent observation before
conditioning on that observation. It uses a Gaussian mixture over represented
components, with fitted transition probabilities conditional on those components
for HMM assignments, or empirical training occupancies for IID assignments.
The validation state is updated locally for the MAP component. Physical-clock
states all propagate at each tick; cluster-clock states commit only the selected
component's step. Affine observations use low-rank Gaussian solves, so diagonal
noise does not require factoring a waveform-sized predictive covariance.

This is a fast **MAP filtering approximation**, not the native variational
objective or an exact marginal likelihood integrating all assignment histories.
There are no component births during validation. Scores therefore describe the
known regimes in each clip and can miss unseen rare regimes. The validation
suffix is held out from component fitting; the runner's fixed encoder/projection
may have been calibrated retrospectively on the record. Use separate calibration
data for a strictly causal forecasting evaluation. The scorer currently requires
CPU and `--alignment none`, because normalized held-out warp densities are not
implemented. It never silently disables warping. The old
`--calibrate-lds-noise-priors` option cannot be combined with screening.

Ranking uses mean validation negative log density per observed feature. A
predictive plateau contains promoted candidates within one empirical standard
error of the best candidate (minimum width 0.001 NLL units); seed assignment
agreement breaks choices within that plateau. This standard error describes
variation across pilot fits, not a calibrated confidence interval. Nearby-pair
assignment agreement, singleton fractions, capacity saturation and fixed-matrix
gap growth are also reported. They are diagnostics, not hard rules that remove
singletons, force multiple clusters or forbid physical instability. Purity, ARI
and NMI against reference labels are never used by the screen.

Grid colors and confirmation averages represent different evaluation stages.
The selection rings and star refer to the final decision shown in the right
panel, so the star need not have the lowest first-clip NLL. A candidate with
only an initial fit has not been compared across clips/seeds. No score
normalization or additional global evaluation is needed for this display;
both panels use values already saved in `prior_screen.json`.

`working_pairs` lists tested combinations. Independent min/max bounds would
include untested or poor cross-combinations, so the report does not promise a
rectangular safe region. A useful predictive region does not guarantee the
domain's preferred semantic segmentation. Confirm several survivors with a
larger budget before an expensive full experiment.

## Python API

```python
from dataclasses import replace
from hdpgpc.model_selection import NoisePriorScreenConfig, screen_noise_priors

def candidate_config(sigma, gamma):
    # For an ordinary GP/LDS with no custom fixed covariance values.
    # A fixed affine decoder needs its Sigma policy value scaled instead;
    # the shared experiment runners already perform that conversion.
    return replace(config,
        emission=replace(config.emission, ini_sigma=sigma),
        dynamics=replace(config.dynamics, ini_gamma=gamma))

report = screen_noise_priors(values, candidate_config, grid=grid,
    sigma_prior=0.1, gamma_prior=0.1,
    options=NoisePriorScreenConfig(seconds=120, max_fits=60, grid_points=5))
print(report['recommendation'])
```

The callback must construct fresh configurations using one frozen representation.
There is no label argument and no final fitted model in the result. Caller CPU
RNG state, initialized CUDA RNG states, thread count, default dtype and
deterministic-algorithm settings are restored after screening.
