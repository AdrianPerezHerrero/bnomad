# Synthetic images with known LDS dynamics

Generate a small ordered image experiment before using complex videos:

```bash
python examples/synthetic_lds_images.py
python examples/synthetic_lds_images.py --state-mode per_regime
```

The default run contains **720 grayscale images at 32 x 32 pixels**, six latent
coordinates, and three **overlapping physical-field regimes**: slow waves,
faster waves and reverse waves. All share a broad background, wave bands and a
localized rotating dipole. Six 120-frame segments revisit these regimes.
This is a visual climate/physical-field analogy, not a calibrated climate or
fluid simulator. The distinct tissue textures are still available explicitly.
Seeds and every generative option are recorded. Repeating
an identical command refuses to overwrite its existing output; choose a new
output directory or seed to create another run.

## Shared generative equations

Each regime has its own history, as in the model's cluster-state convention:

\[
z_{k,0}\sim N(m_k,P_k),\qquad
z_{k,n}=A_kz_{k,n-1}+\epsilon_{k,n},\qquad
\epsilon_t\sim N(0,\Gamma_{s_t}),
\]

\[
y_t=\mu+Cz_t+\eta_t,\qquad \eta_t\sim N(0,\sigma^2 I).
\]

The **same C, offset and observation noise** are used for every regime. There
are no regime-specific decoder matrices or nonlinear image warps.

## Default: physical_fields, an intermediate appearance/dynamics experiment

```bash
python examples/synthetic_lds_images.py --scenario physical_fields
python examples/synthetic_lds_clustering.py --representation pca
```

All regions use the same family of smooth wave and eddy-like image structures.
Their nominal images differ only by weak large-scale gradients (amplitude scale
0.018 at appearance_strength=1). The shared background is much stronger than
these mean differences. Four dynamic coordinates produce broad wave bands and
a localized rotating dipole through a fixed linear map.

The two appearance coordinates start with mean-vector norm 0.65 and standard
deviation 0.55 per coordinate. They then undergo random-walk drift with standard
deviation 0.04 per step. Thus within-regime appearance variation is substantial
relative to the regime mean differences, rather than nearly fixed templates.
The default oscillator-induced pixel SD scale is 0.06 and observation SD is
0.01. No per-frame contrast normalization or clipping changes the likelihood.

| Regime | First angle | Second angle | First pair variance | Second pair variance |
|---|---:|---:|---:|---:|
| slow_wave | 0.08 | 0.12 | 1.00 | 0.70 |
| fast_wave | 0.14 | 0.17 | 0.85 | 1.00 |
| reverse_wave | -0.10 | -0.14 | 0.95 | 0.85 |

Angles are radians per native regime update. Damping is rho=0.995; each
oscillator's Gamma=(1-rho²) times its stationary covariance. Activity amplitudes
overlap between regimes. The appearance block has A=I and Gamma=0.04² I, so it
is a slowly evolving field anomaly, not a stationary process.

`--appearance-strength` controls weak mean-field differences, `--contrast`
controls oscillatory field strength and `--appearance-drift` controls variation
within a regime. This is a designed intermediate case; clustering difficulty
and recovery are not guaranteed or established merely from its image previews.

Both physical_fields and appearance_dynamics require per_regime histories.

## Previous distinct-tissue case: appearance_dynamics

```bash
python examples/synthetic_lds_images.py --scenario appearance_dynamics
```

The first two coordinates encode three different nominal textures, around a
common dark central lumen. They start at distinct means and evolve as very slow
random walks: A=I and Gamma=appearance_drift² I. Their default initial standard
deviation is 0.03 and per-step drift standard deviation is 0.002. Distinct means
persist without adding a bias to the state transition. They are not stationary.

The remaining four coordinates contain two damped oscillators. Each uses
rho=0.995, with the following rotation angles and stationary coordinate variances:

| Appearance | First rotation | Second rotation | First pair variance | Second pair variance |
|---|---:|---:|---:|---:|
| folds | 0.10 | 0.17 | 1.00 | 0.15 |
| rings | 0.42 | 0.63 | 0.15 | 1.00 |
| patches | -0.20 | -0.31 | 0.65 | 0.65 |

For each oscillator, Gamma=(1-rho²) times its stationary covariance. Consequently
the regions differ in their nominal image pattern, fluctuation amplitudes, speed
and direction of evolution. C has six columns: two appearance directions and
four dynamic spatial directions, constructed to be orthogonal to each other.

`--appearance-strength` scales nominal texture differences (default 1).
`--appearance-drift` controls slow latent drift. `--contrast` controls the
oscillator-induced pixel standard deviation before activity scaling; its default
is 0.05 for this scenario. `--observation-std` defaults to 0.01 (variance 0.0001).
Changing these controls does not change the fitting policies or the ELBO.

These two appearance-bearing scenarios require `--state-mode per_regime`. A conventional global
switching state would not automatically jump between distinct persistent
appearance means under the same homogeneous transition equation. This generator
therefore rejects that combination explicitly. Each revisited region resumes
its previous state. No extra transition parameter was added to HDP-GPC.

## Original dynamics-only control

```bash
python examples/synthetic_lds_images.py --scenario dynamics_only
python examples/synthetic_lds_images.py --scenario dynamics_only --state-mode switching
```

The old four-coordinate generator remains available, with the same seed-based
algorithm and default contrast 0.12. Its two damped rotations are:

| Regime | First pair | Second pair |
|---|---:|---:|
| slow | 0.10 | 0.17 |
| fast | 0.42 | 0.63 |
| reverse | -0.20 | -0.31 |

All regimes share Gamma=(1-rho²)I, the decoder C, offset 0.5 and observation
variance 0.01²=0.0001. All have the same stationary latent distribution N(0,I),
and therefore the same marginal image distribution. A single-frame appearance
classifier has no population-level regime information. Finite samples can still
have accidental appearance differences. Temporal correlations distinguish them.

C contains four smooth, localized spatial patterns in two image patches.
Oscillating coefficients change their light/dark bands. These are exactly affine
images, rather than rasterized moving objects whose pixel mapping is nonlinear.
The dynamics-only default contrast sets the largest latent-induced marginal
pixel standard deviation to 0.12. Images are Gaussian data, so values need not stay in [0,1].
Never clip or independently normalize frames for the controlled experiment.

## State evolution and initial distributions

- `--state-mode switching` (dynamics_only): a single state evolves at every global frame; the
  current regime chooses A. This is a conventional switching LDS.
- `--state-mode per_regime`: each regime has its own state history, resumes it
  when revisited, and starts from its saved initial distribution. This matches
  the current HDP-GPC convention of evolving a cluster between its successive
  assigned observations, even when other clusters occur between them.

In both cases labels are a prescribed schedule, not a fitted or sampled HMM.
`previous_indices` records which earlier state generated each transition;
-1 means an initial draw. The process matrix is indexed by the destination
frame's label. No artificial transition is applied before an initial draw.
`initial_state_means` and `initial_state_covariances` store the true per-regime
initial distributions. `initial_covariance` is the shared pooled covariance used
to initialize the oracle observation model; regime means are not supplied to the
clustering engine. Old NPZs load with zero means and their saved shared covariance.

## Outputs

Every run gets a separate directory under `agents/synthetic_lds_images/`:

- `dataset.npz`: raw and noiseless images, package-shaped values, true states,
  labels, segment IDs, previous-state indices, A/Gamma/C/Sigma diagonal,
  offset, common model initialization covariance, per-regime initial means and
  covariances, nominal image patterns, grid, seed, scenario and regime names.
- `labels.csv`, `metadata.json`: readable labels and settings.
- `images.png`: frames from every segment with common intensity limits.
- `trajectories.png`: regime timeline and true states.
- `parameters.png`: true A/Gamma and image basis patterns.
- `regime_patterns.png`: nominal physical-field/tissue appearances (not dynamics_only).
- `sequence.gif`, `frames/*.png`: visual previews; turn off with `--no-gif` or
  `--no-frames`. Previews use a recorded common linear uint8 mapping; fit the
  floating-point NPZ rather than reloading preview pixels.

`--fps` controls playback only. `--segment-length`, `--regime-order`,
`--image-size`, `--scenario`, `--rho`, `--contrast`, `--observation-std`, `--seed` and
`--output-dir` control generation. Noise arguments are standard deviations;
the saved Sigma diagonal is their square.

## Existing fixed-affine observation interface

```python
from hdpgpc.datasets import generate_switching_lds_images

data = generate_switching_lds_images(state_mode='per_regime')
observation = data.observation_model()  # existing AffineGaussianObservation
values = data.values                 # (720, 1024, 1)
grid = data.grid                     # flattened pixel indices
```

The oracle observation model uses the known C, offset, Sigma and initial state
covariance. It is an explicit controlled baseline, not a PCA fitted to labels.
An alternative experiment can calibrate the existing PCA/linear autoencoder on
`values`, then keep its affine representation fixed. Do not use a GP kernel or
1D warp over flattened pixel indices: these indices have no spatial distance
interpretation. No change to model inference is required by the generator.

For an eventual clustering experiment, learn cluster A/Gamma, enable the HMM,
and compare ordered data against shuffled frames and A=0. Report purity/NMI,
boundary recovery and one-step prediction. Similar regime occupancies here
avoid the large majority-label baseline in some CAS-Colon videos. This generator
does not run inference or claim successful recovery.

## Cluster the images

```bash
# Generate per-regime data in memory; learn A/Gamma with the known fixed C/Sigma.
python examples/synthetic_lds_clustering.py

# Load either of the datasets already generated on disk.
python examples/synthetic_lds_clustering.py --input agents/synthetic_lds_images/per_regime_seed_42_6b1c23194133/dataset.npz

# PCA defaults to six coordinates for fields/tissue, four for dynamics_only.
python examples/synthetic_lds_clustering.py --representation pca

# Reproduce the original equal-appearance benchmark in new runs.
python examples/synthetic_lds_clustering.py --scenario dynamics_only --representation pca --components 4

# Two controls: remove temporal order, or remove the latent mean propagation
# and HMM conditioning. Neither control modifies the original source data.
python examples/synthetic_lds_clustering.py --shuffle
python examples/synthetic_lds_clustering.py --a-mode fixed --no-hmm-switch

# Change inference, priors and diagnostics using the same names as other examples.
python examples/synthetic_lds_clustering.py --hdp-preset more --gamma-prior 0.01 --sigma-prior 1 --iterations 50 --verbose debug
```

The clustering entry point is `examples/synthetic_lds_clustering.py`. It reuses
the ordinary `HDPGPC(config).fit(values, grid=grid)` and experiment reporting;
there is no separate inference model. Defaults are offline split/merge, hard
assignments, eight-cluster cap, balanced HDP, final LDS scoring and conjugate
batch updates. The three true regime labels are never passed to fitting or used
to initialize proposals. The known A matrices are **not** used as A priors.

The `oracle` representation supplies true C, offset and observation covariance.
`--sigma-prior` multiplies that covariance (or the PCA-calibrated covariance).
`--sigma-mode` accepts the existing fixed/full/diagonal, cluster/shared policies.
Without `--gamma-prior`, the process prior uses the true mean variance expressed
in decoder coordinates; that is an explicit truth-informed calibration control,
including for PCA. Set `--gamma-prior` to an absolute variance to use a chosen
isotropic process prior instead. Gamma remains learned by default. The oracle
initial state covariance is the saved common initialization (pooled appearance
and fluctuation covariance for fields/tissue, I for dynamics_only); PCA uses its
calibration score covariance. True regime-specific initial means are not passed
to the clustering engine.

`--a-mode fixed` sets A=0, not I. Gamma still learns unless `--gamma-mode fixed`
is also supplied; the current affine interface only permits fixed Gamma with
fixed A. HMM dependence is a separate option, `--no-hmm-switch`. These controls
remove predecessor conditioning but still retain a learned latent covariance.
The script supports offline inference only because its image reports need full
posterior state histories. Fixed-affine observations require final scoring,
no legacy LDS update, no GP fitting, and no warp. Invalid combinations fail
explicitly. All images stay in their original floating-point units.

When `--input` is supplied it determines the data, including seed, image shape,
scenario and state mode. The generation arguments are unused. `--max-series` selects a
prefix before `--shuffle`; assignment CSVs record source-frame indices. PCA
calibrates without labels on the selected frames and requires more calibration
frames than components. This is an in-sample experiment, not held-out accuracy.

Outputs live in `agents/synthetic_lds_clustering_runs/<dataset>/<run_id>/`:

Physical-field names begin with `physical_fields_`, tissue names with
`appearance_dynamics_`; legacy loaded data
keep their old dataset name. Previous data and result directories are preserved.

- `clustering.log`, `settings.json`, `config.yaml`, `results.json` and a shared
  results CSV use the same scheme as CAS-Colon/UCR examples. Run identity includes
  settings, source fingerprint and package code; `--resume` skips complete runs.
- `assignments.csv` contains true labels, clusters, majority labels and original
  frame IDs. Metrics include purity, ARI, NMI and the majority-label confusion
  matrix. Majority labels are derived for evaluation, not used for inference.
- `clusters_output_0.png` uses the existing image gallery/trajectory summary.
  Intensity limits are common across source frames; display clipping of model
  reconstructions does not alter fitting. No plot window opens.
- `projection.png`, `latent_trajectories.png` and `lds_parameters_output_*.png`
  provide the existing representation and LDS diagnostics. Add
  `--lds-dynamics-plots` for emission-space LDS dynamics. `--no-plots` disables
  figure generation; `--min-cluster-samples` filters detail panels, not metrics.
- `ground_truth_recovery.png` compares true regimes, inferred clusters,
  cluster-majority regimes and true/smoothed states in fit order.
- `ground_truth.npz`, `latent_trajectories.npz`, `lds_parameters.npz` and
  `observation_calibration.npz` preserve the comparison arrays.
- `ground_truth_comparison.json` reports smoothed state RMSE in true latent
  coordinates, a zero-state baseline and per-cluster parameter comparisons.
  A/Gamma relative errors are only reported in oracle mode, where coordinates
  are identical. PCA state comparison projects into the true latent space;
  discarded dimensions cannot be recovered. Majority mappings and smoothed
  errors are descriptive in-sample diagnostics, not predictive validation.

Prefer `per_regime` for the first model-compatible experiment. Conventional
`switching` data and shuffled data intentionally violate the current resumed
cluster-state convention; record that mismatch when interpreting their results.
This script does not guarantee three recovered clusters. Regime discovery from
dynamics alone can still be difficult for the current proposal initialization.
