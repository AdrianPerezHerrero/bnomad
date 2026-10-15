# Fixed observation models

The LDS state remains Gaussian. `FixedObservationModel` defines the observation
boundary for a fixed decoder on one waveform grid:

- `condition(prior_mean, prior_covariance, observation, responsibility, noise)`
  returns the Gaussian state posterior used by the filter;
- `encode_likelihood(observation, noise)` returns a beat's exact latent
  information factor when the decoder is affine Gaussian;
- `encode(observation)` supplies a Gaussian latent initializer, and
  `decode(latent)` supplies the conditional waveform mean;
- `expected_log_likelihood(observation, latent_mean, latent_covariance, noise)`
  supplies the waveform-space emission term in the variational objective;
- `predictive_moments(latent_mean, latent_covariance, noise)` supplies the
  waveform mean and full covariance for prediction and plots;
- `state_dict()` supplies data-only parameters for independent serialization.

`AffineGaussianObservation` implements all methods exactly for

\[
  y\mid f \sim \mathcal N(b+Cf,\Sigma), \qquad
  q(f)=\mathcal N(m,P).
\]

It computes

\[
  E_q\log p(y\mid f)=-\tfrac12\{(y-b-Cm)^T\Sigma^{-1}(y-b-Cm)
  +\operatorname{tr}(\Sigma^{-1}CPC^T)+\log|\Sigma|+D\log(2\pi)\}.
\]

Each beat is encoded in information form:

\[
  H=C^T\Sigma^{-1}C,\qquad h=C^T\Sigma^{-1}(y-b).
\]

The LDS filter combines that beat's factor with its cluster-specific dynamic
prior precision \(P^{-1}\): posterior precision is \(P^{-1}+rH\), and posterior
information is \(P^{-1}m+rh\), where \(r\) is the responsibility. This is
algebraically the same Gaussian conditioning as scoring the waveform directly;
it does not add an independent encoder likelihood or count the beat twice.
For full-column-rank \(C\), `encode_measurement` also returns the latent
measurement \(H^{-1}h\) and covariance \(H^{-1}\). Information form still works
for rank-deficient \(C\). The separate `encode` method remains a posterior
initializer under the calibrated zero-mean latent prior.

The decoder is fixed; Sigma may use any currently supported fixed, cluster, or
shared noise policy. PCA is one way of calibrating this affine decoder. A pretrained linear
decoder can be installed directly:

```python
from hdpgpc.config import with_fixed_observation_model
from hdpgpc.emissions import AffineGaussianObservation

decoder = AffineGaussianObservation(
    C=decoder_matrix, Sigma=observation_covariance,
    observation_grid=waveform_grid, offset=decoder_offset,
    latent_covariance=initial_latent_covariance,
)
config = with_fixed_observation_model(config, decoder)
```

[The autoencoder record example](../examples/autoencoder_record_clustering.py)
trains a tied linear encoder/decoder on 400 beats, estimates the fixed Gaussian
decoder noise from its residuals, and then fits the HDP-GPC on the complete
record. Every beat is encoded by the fixed observation model before its latent
state is fused with the cluster's dynamic prior. For isotropic Sigma and the
tied orthonormal weights in this example, \(H^{-1}h\) is exactly the trained
encoder output. The script reports their numerical agreement on calibration
beats. The final latent state may differ from that output because of the
dynamic prior.

The helper sets a latent grid of component indices, a fixed C, and learned
per-cluster A/Gamma. It requires one output, CPU inference, a fixed GP kernel,
and no alignment warp. The serialized configuration contains the arrays and
policies rather than a Python callback. The decoder's own `state_dict()` is a
separate, data-only representation.

`make_fourier_observation(grid, window_size=16, n_harmonics=6)` constructs this same affine
class with a real orthonormal windowed Fourier synthesis matrix. A full basis retains
the waveform information; a selected basis reduces the LDS dimension. Its
encoding, Gaussian score, predictive covariance, Sigma policies and
serialization use the same methods as PCA and the linear autoencoder. See
[fixed windowed Fourier observation](fourier_observation.md) for the coefficient
ordering and a runnable example. Fixed identity emission can also use this
interface by supplying `C=np.eye(N)` to `AffineGaussianObservation` and
installing it with `with_fixed_observation_model`. The usual learned-C LDS
continues to learn its emission matrix through its existing parameter policy.

New `with_pca_emission` configurations delegate to this generic installer.
`make_wavelet_observation(grid)` also uses this installer and the same affine
Gaussian methods, with a full orthonormal Haar decoder. See
[fixed wavelet observation](wavelet_observation.md) for ordering and the ACSF1 example.
Previously saved `pca_*` metadata are still accepted when loading recipes.

[The UCR archive example](../examples/ucr_autoencoder_archive.py) then fixes
A to the zero matrix while learning a separate Gamma for each cluster from a
calibrated latent-covariance prior. It also sets `hmm_switch=False`. Thus the
archive's arbitrary row order does not enter through either the HDP transition
matrix or the LDS state transition. The fixed affine decoder accepts this
fixed-A/learned-Gamma configuration alongside the original learned pair.

## Nonlinear encoders and GPLVM decoders

An encoder can initialize latent inference, but its output is not itself an
observation likelihood. To integrate a nonlinear decoder, an adapter must
implement a justified Gaussian approximation to `condition` and an expected
**waveform-space** log likelihood under the resulting `q(f)`. It must also
propagate latent uncertainty through its predictive moments. The method used
for these expectations (for example, analytic integration, quadrature, or
specified Monte Carlo sampling) must be stated by the adapter. A GPLVM
adapter must include decoder predictive uncertainty as well as observation
noise. The current conjugate LDS fitter accepts only the exact affine class;
it rejects other models instead of scoring an encoder reconstruction as if it
were a likelihood.
