# Factor-analysis emissions in HDPGPC

MFA is an observation extension of `HDPGPC`, not a separate model or inference
engine. The native assignment state, HDP presets, births/deletions, offline
`legacy` and `split_merge` strategies, responsibilities, stopping rules and LDS
scoring are retained.

For cluster k the observation is

\[
y_t = \mu_0 + \delta\mu_k + C_k z_t + \epsilon_t,
\qquad \epsilon_t\sim N(0,\operatorname{diag}(s_k)).
\]

The common calibration center is mu_0. Each cluster **learns** its intercept
delta_mu, rectangular loading matrix C and diagonal noise. Calibration does
not freeze the emission or fit a separate mixture. Feature indices are not GP
input coordinates; the native affine path bypasses GP interpolation and fitting.

## Basic use

```python
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from hdpgpc import HDPGPC, HDPGPCConfig
from hdpgpc.config import (
    GPEmissionConfig, DynamicsConfig, AlignmentConfig, InferenceConfig,
    HDPConfig, NumericalConfig, with_mfa_emission,
)
from hdpgpc.emissions import estimate_mfa_observation

# X: (observations, features), e.g. flattened grayscale images.
grid = np.arange(X.shape[1], dtype=float)[:, None]
initial = estimate_mfa_observation(
    X, grid, components=10, max_samples=400, random_state=42,
)
config = HDPGPCConfig(
    x_basis=np.arange(10, dtype=float)[:, None],
    emission=GPEmissionConfig(kernels=[RBF(1.)], fit_gp=False, ini_sigma=.1),
    dynamics=DynamicsConfig(ini_gamma=1., lds_update='conjugate_batch'),
    alignment=AlignmentConfig(policy='none'),
    hdp=HDPConfig(hdp_hyp='balanced', max_models=12),
    inference=InferenceConfig(schedule='offline', it_limit=30,
        offline_strategy='split_merge', hard_assignments=False),
    numerics=NumericalConfig(seed=42, num_threads=1),
)
config = with_mfa_emission(config, initial)
model = HDPGPC(config).fit(X)
parameters = model.observation_parameters_
```

The kernel remains construction metadata of the existing trajectory container;
it is neither trained nor used to interpolate factor coordinates. The helper
uses the existing typed matrix/noise policies. Serialized configuration records
`observation.family='factor_analysis'`.

`estimate_mfa_observation` uses a label-free spectral minibatch initializer,
scaling principal directions for unit-variance factors and subtracting a pooled
residual variance. This is a prior initialization, not the maximum-likelihood
solution of a diagonal-noise FA model. Uniform sampling is retrospective;
choose `sampling='prefix'` or independent calibration data for causal use.

## Static default and evolving factors

`with_mfa_emission(..., dynamic=False)` defaults to

- independent assignments (`hmm_switch=False`);
- independent factors z_t ~ N(0,I), via fixed A=0 and Gamma=I;
- cluster-learned intercept/C and diagonal Sigma.

This uses the native **dynamic LDS machinery with fixed zero A**, rather than
the older `model_type='static'` preset (which means zero process noise and is a
different model). Neither observation order nor the last assigned waveform
determines the static factor prior. During offline scoring each observation
receives its own conditional factor posterior for every candidate cluster,
including clusters to which it is not currently assigned.

`with_mfa_emission(..., dynamic=True)` instead learns native cluster A/Gamma:

\[
z_t=A_k z_{t-1}+\eta_t,\qquad \eta_t\sim N(0,\Gamma_k).
\]

The input configuration's HMM assignment switch is retained in this case, so
assignment dependence and factor evolution remain separate choices. Transitions
follow the package's retained cluster-local state clock, not wall-clock gaps.
The native LDS approximation/update and scoring conventions apply to this mode.

## Posterior and scoring

Let u=[1;z] and B_k=[delta_mu_k,C_k]. Each feature row has the conjugate prior

\[
B_{kj}\mid s_{kj}\sim N(m_{0j},s_{kj}K_0^{-1}),\qquad
s_{kj}\sim IG(n_0/2+1,n_0 s_{0j}/2).
\]

The prior mean is [0, calibrated C]. `factor_mean_precision` and
`factor_loading_precision` set the two blocks of K_0; helper arguments are
`mean_precision` and `loading_precision`. The configured LDS degrees of freedom
and Sigma policy strength determine n_0. Variance values are already squared
units. Each feature's prior mean noise variance is s_0j.

For responsibilities r_tk and latent moments m_t,S_t, pool

\[
N_k=\sum_t r_{tk},\quad
S_{uu}=\sum_t r_{tk}\begin{bmatrix}1&m_t^T\\m_t&S_t+m_tm_t^T\end{bmatrix},
\]
\[
S_{yu}=\sum_t r_{tk}(y_t-\mu_0)[1,m_t^T],\qquad
s_{yy,j}=\sum_t r_{tk}(y_{tj}-\mu_{0j})^2.
\]

Then

\[
K=K_0+S_{uu},\quad M=(M_0K_0+S_{yu})K^{-1},\quad n=n_0+N_k,
\]
\[
b_j=b_{0j}+\tfrac12(s_{yy,j}+m_{0j}K_0m_{0j}^T-M_jKM_j^T),
\qquad a=n/2+1.
\]

The implementation evaluates the noise update using positive residual/latent
uncertainty/prior-deviation contributions, and only computes its diagonal.
The deterministic intercept is a parameter, not an extra LDS latent dimension.
Batch updates reset to the original prior; sequential and natural-gradient
updates follow the corresponding native schedule. A first observation can
update the observation posterior without a transition observation.

Conditioning and observation scores include intercept/loading uncertainty,
not only the posterior mean C. With V=K^-1 and tau_j=E[1/s_j], the observation
factor contributes latent precision and information

\[
H=\bar C^T\operatorname{diag}(\tau)\bar C+d V_{zz},\qquad
h=\bar C^T\operatorname{diag}(\tau)(y-\mu_0-\overline{\delta\mu})-dV_{z0}.
\]

Gaussian conditioning adds H and h to the LDS prior's natural parameters.
Expected observation log likelihood uses E[log(1/s)], expected squared residuals
and both state and coefficient uncertainty. For independent factors the latent
term is -KL(q(z_t|k)||N(0,I)). Assignment weights multiply sufficient statistics
and local scores, rather than tempering the component-conditional static factor
posterior itself. Noise/intercept/loading parameter KL is counted once through
the existing observation contribution to the LDS parameter score.

**The existing `birth_lds_kl_per_dimension` flag remains in effect.** It divides
the LDS parameter penalty by the latent LDS dimension, excluding the intercept.
That is the package's experimental proposal score, not an unmodified ELBO.
Set it to False to use unscaled parameter penalties. No additional MFA-only
normalization, objective or stopping rule is introduced.

## CAS-Colon example

```bash
python examples/cas_colon_mfa_clustering.py --records 1 --image-size 24 24 --components 10 --offline-strategy split_merge
python examples/cas_colon_mfa_clustering.py --records 1 --offline-strategy legacy
python examples/cas_colon_mfa_clustering.py --records 1 --dynamic --hmm-switch --gamma-prior .1
```

The existing CAS workflow handles data, logs, content-addressed run folders,
purity/ARI/NMI, assignment plots, image galleries and latent trajectories.
`--sigma-prior` multiplies calibrated noise prior variances; `--gamma-prior`
sets the process prior only in dynamic mode (static Gamma is I).
`--mean-precision` and `--loading-precision` change emission prior precision.
`mfa_posteriors.npz` additionally stores learned offsets, augmented coefficient
means, column precision, noise scales and degrees of freedom. Display projections
use the common initialization; learned cluster-local factors need not share an
identifiable coordinate system.

## Scope and validation status

The same native offline engines and online entry points are used. Batch static
conditioning uses the existing independent-state optimization. The current
affine route requires CPU, one output, no warp/GP fitting, final LDS scoring and
a conjugate or natural-gradient update. C is learned per cluster and Sigma is
learned diagonal per cluster; shared FA parameters, full Sigma and historical
FA parameter scoring are rejected explicitly. The native restrictions on A/Gamma
policies still apply.

`FactorAnalysisEmission` is an immutable observation snapshot implementing the
affine observation contract. Its state dictionary round-trips its predictive
and scoring moments. The native cluster owns the trainable conjugate posterior;
ordinary native model pickling retains it. Snapshot serialization alone is not
a training checkpoint. Existing configuration serialization also records the
family, initialization and hyperparameters.

For this implementation change, syntax and diff formatting were checked.
Numerical inference, runtime plotting, serialization round trips and GP
regression tests have not been run. Cluster recovery and monotonic convergence
are not claimed from source inspection alone.
