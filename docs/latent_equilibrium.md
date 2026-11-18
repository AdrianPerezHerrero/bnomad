# Constant input and latent equilibrium

All shared experiment parsers accept `--constant-input` and
`--latent-equilibrium` as aliases for the same optional affine transition:

\[
z_{t+1}=A_kz_t+b_k+\epsilon_t,
\qquad \epsilon_t\sim\mathcal N(0,\Gamma_k).
\]

For example, add the option to an existing MIT-BIH command:

```sh
python examples/mitbih_autoencoder_clustering.py --records 207 \
  --dynamics-clock physical-clock --latent-equilibrium --trajectory-plots
```

The option is off by default. `--no-constant-input` and
`--no-latent-equilibrium` disable it explicitly. Neither alias constrains the
eigenvalues of A or changes the sampling-clock interpretation. Existing
restrictions on emissions, schedules, alignment and parameter policies still
apply; enabling a constant input does not enable otherwise unsupported
combinations.

The typed configuration is:

```python
from dataclasses import replace

config = replace(config, dynamics=replace(
    config.dynamics, constant_input=True, constant_input_strength=1.0))
```

The default-off fields are omitted from serialized recipes, so old recipes
continue to deserialize with zero latent input. Existing LDS presets preserve
an explicit constant-input choice.

## Estimation

The model estimates the intercept b directly rather than repeatedly solving for
an equilibrium during inference. The transition regression uses the predictor
`[z_t; 1]` and the coefficient matrix `[A, b]`. The constant predictor is not
an additional stochastic latent component. The usual expected sufficient
statistics include the latent-state covariance and successive-state
cross-covariance.

The extension supports batch, sequential, natural-gradient and ML updates.
When enabled with the `legacy` schedule, transition learning uses the affine
conjugate update; the historical square-matrix heuristic does not extend to
the rectangular `[A,b]` regression. Disabling the option retains that legacy
heuristic and its original covariance scaling.

`--constant-input-strength` (alias `--latent-equilibrium-strength`) controls
the zero-centered intercept column's prior precision, independently of
shrinkage on A. Its default is 1; any finite positive value is valid. Larger
values pull the intercept more strongly toward zero. The intercept prior is
scaled by the transition covariance in the conjugate matrix-normal family.
This is a prior on b, not a direct Gaussian prior on the derived equilibrium.

Shared transition policies own a shared b together with shared A/Gamma.
Cluster transition policies retain a separate b per component. A fixed A or
Gamma can coexist with a learned intercept. Parameter learning, KL scoring and
structural proposals must retain the same intercept ownership.

With the option enabled, the standard `lds_parameters.npz` export includes
`output_<output>_cluster_<cluster>_b`, shaped `(latent_dimension, 1)`, alongside
A and Gamma. Latent-trajectory arrows and waveform-space one-step diagnostics
include this input. The existing current-state plots and early-stop views keep
their respective observation-time conventions.

## Meaning of the equilibrium

When `I-A` is invertible, the fixed point is obtained by solving

\[
(I-A_k)\mu_k=b_k.
\]

It is attracting only if the spectral radius of A is below one. In that case,
the conditional mean through h unobserved ticks is

\[
m_{t+h}=\mu_k+A_k^h(m_t-\mu_k).
\]

The covariance still propagates with process noise; retaining a nonzero mean
does not remove uncertainty during a gap. A matrix with an eigenvalue near one
can give a very uncertain equilibrium even if b is reasonably estimated.
The preset named `stable` shrinks A toward identity and does not enforce an
attracting fixed point.

Special cases keep their usual mathematical interpretation:

- With fixed `A=I` (random walk), b is a drift per step. The transition has no
  unique equilibrium unless further constraints are imposed.
- With `A=0` and independent factors with positive Gamma, b is a latent
  component location rather than temporal mean reversion.
- Legacy static GP dynamics with zero Gamma retain their frozen latent state;
  the constant input has no transition to modify.

In cluster-clock mode, a step joins consecutive retained component
observations. In physical-clock mode, it joins consecutive global sampling
ticks, including those without a measurement from that component. A missing
measurement is not replaced with a zero-valued observation, and predicted
states alone are not independent new measurements.

For fixed PCA/AE/Fourier/wavelet decoders, the calibration fixes the latent
coordinate system. With learned affine emissions, latent location and
observation intercept can trade off; the fitted priors anchor the selected
representation. Equilibrium coordinates should be compared only in a common
latent basis.

See [the two dynamics clocks](dynamics_clocks.md),
[LDS parameter policies](lds_inference.md), and
[experiment arguments and saved results](experiment_workflows.md).
