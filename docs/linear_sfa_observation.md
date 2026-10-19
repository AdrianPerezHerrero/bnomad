# Linear SFA as a fixed affine observation

Linear Slow Feature Analysis (SFA) calibrates a representation from ordered
observations. It selects directions by temporal change relative to their
variance, rather than by reconstruction variance alone. Its decoder is installed
through the same `AffineGaussianObservation` and `with_fixed_observation_model`
interface as PCA, the tied linear autoencoder and fixed Fourier/wavelet bases.
There is no additional inference engine or matrix-policy system.

## CAS-Colon example

Use the extracted dataset layout described in the [PCA image guide](cas_colon_pca.md).

```powershell
python examples/cas_colon_sfa_clustering.py --records 1
python examples/cas_colon_sfa_clustering.py --records 9 --image-size 24 24 --components 8 --max-series 150
python examples/cas_colon_sfa_clustering.py --records 1 --calibration-samples 400 --sfa-whitening-components 64 --sfa-lag 1
python examples/cas_colon_sfa_clustering.py --records all --resume
```

Defaults retain the image experiments' grayscale [0,1] pixels, ten latent
dimensions, offline conjugate-batch inference, final scoring, HMM transitions,
and split/merge search. C and the calibration offset are fixed; A/Gamma learn
per cluster. Sigma defaults to fixed diagonal reconstruction variance. Existing
`--sigma-mode`, `--sigma-prior`, `--gamma-prior`, `--lds-degrees`, stopping and
plot controls apply. GP fitting and 1D alignment remain disabled.

## Temporal calibration

Let the centered image be x = y - mu, and let S be its empirical population
covariance on the calibration rows. Linear SFA solves

\[
 \min_W \operatorname{tr}(W^T D W),\qquad W^T S W=I,
 \quad D=\frac1{M}\sum_{(i,j)}
 \frac{(y_j-y_i)(y_j-y_i)^T}{(t_j-t_i)^2}.
\]

The implementation computes an SVD of centered calibration rows, discards
singular values below `rank_tolerance * largest_singular_value`, and whitens
the retained covariance directions. It then diagonalizes the derivative second
moment in that space and selects the smallest eigenvalues. Differences are
**not centered**: a constant nonzero drift must still incur a temporal cost.
This is linear SFA without nonlinear expansion. The general slowness objective
and normalization are described by
[Wiskott and Sejnowski (2002)](https://www.ini.rub.de/PEOPLE/wiskott/Reprints/WiskottSejnowski-2002-NeurComp-LearningInvariances.pdf).

The CAS-Colon pilot whitening rank defaults to 64 (`--sfa-whitening-components`).
It must be at least the requested feature count. Increase it to expose weaker
variance directions to temporal selection, at additional cost. The effective
rank is capped by numerical covariance rank and the number of derivative pairs;
the output dimension is reduced if necessary. This avoids selecting directions
whose zero derivative cost merely reflects too few sampled differences. A
zero-rank calibration or absence of valid pairs raises a clear error.

`--sfa-rank-tolerance` defaults to 1e-6. It is a relative singular-value cutoff,
not a learned parameter or a covariance jitter. The helper uses all supported
whitening directions when `whitening_components=None`, still subject to its
numerical-rank and pair-count caps.

### Preserve actual adjacency

- Prefix calibration uses the first `--calibration-samples` rows and only pairs
  within that prefix.
- Uniform calibration samples genuine lagged pairs from the selected segment,
  using their unique endpoint rows, never more than the sample budget. If the
  whole segment fits the budget, all valid pairs are used. Otherwise up to
  `floor(calibration_samples / 2)` pairs are sampled without replacement.
- `--sfa-lag 1` uses neighboring selected frames. A lag of L pairs rows i and
  i+L, with their actual elapsed seconds as the denominator. `--frame-step`
  changes the selected frame cadence as in the other image runners.
- Frame gaps are already separated into independent runs by the dataset loader.
  The general helper also accepts `segment_ids` and never pairs across a segment
  boundary, even if an ID reappears later. Times may reset between segments.

Uniform calibration can therefore use fewer unique frames than its budget.
Its covariance is estimated on those unique rows, while the derivative objective
uses the saved sampled pairs. It never forms differences between two randomly
selected frames merely because they became neighbors in a minibatch.

## Decoder and exact LDS inference

Raw SFA features are z = W.T x. A least-squares decoder is fitted on the same
calibration rows:

\[
 C=S_{xz}S_{zz}^{-1},\qquad
 e_i=y_i-\mu-Cz_i,\qquad
 \Sigma_0=\operatorname{diag}(\max(\operatorname{Var}(e_{:,j}),\epsilon)).
\]

Residual variances use ddof=1; epsilon is `--noise-floor` (default 1e-6). The
usual `--sigma-prior` multiplies Sigma once before installation. SFA's unit
population feature covariance gives the initial latent covariance P0=I.
Gamma's prior remains `--gamma-prior * I`, default 0.01; it is not a derivative
variance estimate. Noise variances and SFA derivative variances have different
meanings and are not substituted for one another.

During inference the observation remains

\[
 y_t\mid f_t\sim\mathcal N(\mu+C f_t,\Sigma),\quad
 H=C^T\Sigma^{-1}C,\quad h=C^T\Sigma^{-1}(y_t-\mu).
\]

The existing filter combines H/h with the cluster's latent prior. Observation
and latent scores, parameter KLs, responsibilities, birth proposals and
serialization follow their existing paths. There is no extra SFA likelihood or
smoothness penalty in the ELBO, so observations are not counted twice.

**Raw SFA encoding generally differs from Gaussian likelihood encoding.** The
fitted decoder is generally nonorthogonal, and diagonal residual noise does not
make its precision-weighted inverse equal to W.T. SFA provides the calibration
coordinates and decoder subspace; actual states are the Gaussian LDS posterior
under that decoder. No raw feature is forced to be a noiseless latent state, and
posterior trajectories are not guaranteed to attain the raw SFA slowness.

## Reuse independently of images

```python
from hdpgpc.emissions import estimate_sfa_observation
from hdpgpc.config import with_fixed_observation_model

estimate = estimate_sfa_observation(
    observations, grid, components=10, max_samples=400,
    times=timestamps, segment_ids=sequence_ids,
    whitening_components=64, sampling='uniform', random_state=42,
)
raw_codes = estimate.transform(observations)
config = with_fixed_observation_model(config, estimate.model)
```

Inputs are finite `(observations, points)` or `(observations, points, 1)` arrays
in chronological order within each segment. The grid describes observation
coordinates; it is separate from the timestamps. Without timestamps the helper
uses row-step time. The fixed affine configuration restrictions still apply.
An SFA decoder calibrated independently can be used by the package's compatible
online schedules as well; the CAS-Colon runner remains offline to retain full
image trajectory histories. C, offset, Sigma and P0 use the ordinary affine
state dictionary and typed configuration, requiring no new model type on load.

## Reports and limitations

Runs are saved under `agents/cas_colon_sfa_runs/<video_segment>/<run_id>/`.
They retain the same purity/ARI/NMI header, three regime scatter plots, image
galleries, latent trajectories, projection and optional LDS diagnostics as the
other CAS-Colon examples. The raw reconstruction gallery uses **W.T**, not the
decoder pseudoinverse. Its row can therefore differ from the smoothed LDS row.
The projection figure uses raw SFA codes.

`sfa_calibration.npz` contains encoder W, offset, calibration indices, actual
pair indices/times, raw codes, derivative eigenvalues and rank information.
`observation_calibration.npz` contains the installed affine model, including the
applied Sigma. Standard reports identify the actual latent dimension and
calibration pair count. Saved affine state is sufficient for inference without
the SFA calibration arrays; retain those arrays to reproduce raw features.

Labels are excluded from calibration. Ambiguous labels retain the same handling
as the PCA/AE runners. Each segment has its own representation. Uniform
calibration uses future frames; reported metrics are in-sample descriptions.
Small pilot spaces can discard low-variance temporal signals. Large ones can
amplify noise. Slow illumination or camera drift can be selected instead of
anatomy, and linear SFA cannot generally linearize large geometric deformations.
No anatomical improvement over PCA or the affine autoencoder is claimed without
an experiment.
