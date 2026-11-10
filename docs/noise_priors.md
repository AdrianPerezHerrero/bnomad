# Observation and process noise defaults

Starting with the 2026-11-04 default update, both noise-prior defaults are **0.1**:

- Core typed configuration: `GPEmissionConfig.ini_sigma` and
  `DynamicsConfig.ini_gamma`.
- Low-level dynamic covariance initialization: `Sigma=0.1 I`, `Gamma=0.1 I`.
- All dataset CLI parsers: `--sigma-prior 0.1 --gamma-prior 0.1`, including
  every representation and both Xenopus clutches.

Common values are owned by `hdpgpc._defaults`. Explicit constructor arguments,
saved recipes, CLI flags and caller-supplied parser defaults still override them.
The HDP concentration `HDPConfig.gamma` is a separate assignment parameter;
its presets are unaffected.

## Previous defaults

| Entry point | Previous sigma prior | Previous gamma prior |
| --- | ---: | ---: |
| Typed core configuration | `None` (required explicitly) | `None` (required explicitly) |
| Low-level dynamic initialization | 0.25 | 0.01 |
| MIT-BIH waveform runner | 50 | 10 |
| MIT-BIH PCA runner | 1 | 10 |
| MIT-BIH autoencoder entry script | 1 | 0.1 |
| Daphnet waveform runner | 0.1 | 0.1 |
| Daphnet calibrated representations | 1 | 0.1 |
| UCR runners | 0.001 | 0.001 |
| Expression MFA runners | 0.001 | 0.01 |
| Expression PCA/autoencoder runners | 1 | 0.01 |
| Xenopus autoencoder entry script | A: 0.01; B: 0.005 | 0.1 |
| CAS-Colon runners | 1 | 0.01 |
| Flutter runner | 1 | 1 |
| XJTU-SY runners | 0.1 | 0.1 |
| Synthetic LDS image clustering | 1 | Truth-informed calibrated variance |

## Meaning of the value

For ordinary GP waveform models, these inputs initialize covariance variances,
typically as a scalar times identity. They are not standard deviations and are
not squared. For calibrated affine representations, sigma-prior multiplies the
calibrated observation covariance. UCR also uses gamma-prior as a multiplier of
calibrated latent/process covariance. Other runners generally use gamma-prior
as a per-coordinate process variance. Additional explicit calibration/noise
scales continue to apply. See [experiment workflows](experiment_workflows.md).

The default does not replace supplied covariance matrices or change a model's
structural policies: static dynamics retain Gamma=0; independent MFA factors
retain their fixed Gamma=I convention. A default of 0.1 therefore does not mean
every fitted or calibrated covariance matrix equals `0.1 I`.

Explicit historical tutorial/reference recipes and frozen numerical fixtures
retain their stated values. Historical initializer regression workloads now
supply their old 0.25/0.01 values explicitly; separate tests verify the new
defaults, actual core initialization, calibrated scaling and CLI overrides.

See [fast prior screening](prior_screening.md) for the opt-in `--screen-priors` workflow, which estimates and tests joint prior pairs without a full-data fit.
