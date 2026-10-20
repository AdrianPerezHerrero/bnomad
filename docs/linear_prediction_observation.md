# Fixed affine features trained for linear prediction

This calibrator learns a linear representation using both image reconstruction
and latent prediction. It returns the existing `AffineGaussianObservation`, so
the fitted HDP-GPC still uses its ordinary Gaussian observation likelihood,
filtering, smoothing, matrix policies and serialization. The representation is
calibrated independently, then frozen before clustering.

## CAS-Colon experiment

Use the extracted data described in the [CAS-Colon PCA guide](cas_colon_pca.md).

```powershell
python examples/cas_colon_linear_prediction_clustering.py --records 1 --components 20 --prediction-weight 0.1 --min-cluster-samples 5
python examples/cas_colon_linear_prediction_clustering.py --records 1 --components 20 --prediction-weight 1
python examples/cas_colon_linear_prediction_clustering.py --records 1 --components 20 --prediction-weight 0
```

The last command is the PCA control **on the same sampled calibration rows**.
It need not match the separate PCA runner's frame sample, because temporal
calibration samples pairs and their unique endpoints. To shorten an initial
experiment, add `--image-size 16 16 --max-series 100 --iterations 5`.
This control also uses **whitened PCA coordinates** with P0=I. Its raw
reconstruction matches PCA on those rows, but the same isotropic Gamma prior
does not impose the same image-space regularization as unscaled PCA coordinates
in the dedicated PCA runner. Compare weights within this runner for a controlled
representation-objective comparison.

Defaults use ten latent components, at most 400 calibration frames per
contiguous segment, 64 pilot whitening dimensions, 250 optimizer steps and
learning rate 0.02. The prediction weight defaults to 1. The ordinary image
workflow retains chronological order and HMM assignments, learns A/Gamma per
cluster and fixes C/offset. Sigma defaults to fixed diagonal reconstruction
noise; `--sigma-mode` selects the existing supported alternatives. Model fitting
remains offline, with final scoring, no warp and no GP kernel fitting.

## Objective and normalization

For centered calibration observations x=y-mu, learn

\[
 z=W^Tx,\qquad \widehat y=\mu+Cz,\qquad \widehat z_j=Bz_i,
 \qquad \frac1N\sum_i z_i z_i^T=I.
\]

The zero mean follows from centering. Unit covariance prevents latent scale
collapse. The implemented objective is

\[
 \mathcal L=
 \frac{\sum_i\|x_i-Cz_i\|^2}{\sum_i\|x_i\|^2}
 +\lambda\left[
 \frac{\sum_{(i,j)}\|z_j-Bz_i\|^2}{M d}
 +\frac{\epsilon_B\|B\|_F^2}{d}\right].
\]

N is the number of unique calibration rows, M the number of temporal pairs and
d the latent dimension. The reconstruction term is the fraction of centered
calibration variance left unexplained; prediction is mean residual variance
per normalized latent coordinate. These normalizations make the trade-off less
dependent on pixel scale and image dimension. Accordingly, `--prediction-weight`
is not a coefficient on unnormalized sums of squared errors. The explicit
predictor ridge defaults to 1e-6 and stabilizes its regression.

Control interpretation:

| Argument | Meaning |
| --- | --- |
| `--prediction-weight` | Nonnegative weight lambda; 0 returns the PCA control; larger values emphasize predictability |
| `--components` | Requested LDS/feature dimension |
| `--prediction-whitening-components` | Maximum candidate whitening rank, at least `components`; default 64 |
| `--epochs`, `--learning-rate` | Adam steps and learning rate for feature calibration |
| `--prediction-lag` | Lag in selected frame steps; default 1 |
| `--prediction-ridge` | Positive ridge epsilon_B on calibration B; default 1e-6 |
| `--prediction-rank-tolerance` | Relative singular-value cutoff; default 1e-6 |

The feature objective is not a clustering ELBO term. It is used only before
model fitting. Changing lambda does not multiply the model's emission or latent
ELBO factors.

## Efficient constrained calibration

1. Thin SVD of centered calibration rows supplies a whitening map L and a pilot
   decoder G, with H=XL and H.T H/N=I.
2. Write W=LQ, C=GQ with Q.T Q=I. This enforces feature covariance I. The decoder
   is the least-squares solution for those coordinates, rather than an unrelated
   independently trained matrix.
3. Fit B for every Q using
   `B = S10 @ inverse(S00 + prediction_ridge * I)`, implemented with a solve.
4. Optimize Q through a QR parameterization and Adam. Only small pilot/latent
   covariance matrices are needed per step; the pixel arrays are not reconstructed
   repeatedly during optimization.
5. Keep the basis with the smallest recorded total objective, including a PCA
   basis recorded at step zero. This protects against returning a worse total
   objective than that baseline; it does not guarantee a global optimum or an
   improvement in each individual loss term.

The whitening rank is limited by numerical covariance rank and pair count; the
effective latent dimension may be smaller than requested. A zero-rank sequence
raises an error. At zero prediction weight, or when all available whitening
coordinates are retained, no feature optimization is needed. In the latter case
orthogonal coordinate rotations cannot improve this objective. Decoder column
signs are fixed for reproducible artifacts. Random initialization uses a local
generator and does not change the model's global random seed.

Sigma is the diagonal ddof=1 variance of raw decoder reconstruction residuals,
floored by `--noise-floor`. The usual `--sigma-prior` multiplies it once. P0=I
initializes the latent state in these normalized coordinates. Gamma remains
`--gamma-prior * I`, default 0.01, independently of predictor residuals.
`--lds-degrees` retains its usual covariance-prior meaning.

## Temporal sampling and the role of B

Sampling is shared with [Linear SFA](linear_sfa_observation.md): prefix uses
initial rows and their actual pairs, while uniform sampling chooses genuine
lagged pairs and their unique endpoints within the frame budget. No difference
or prediction pair is formed across a segment boundary. All temporal pairs and
timestamps are saved. Labels are never used in calibration.

**One calibration B models one fixed lag duration.** Supplied timestamps must
increase within segments and every sampled pair must span the same duration.
Irregular durations are rejected; resample the observations or use separate
calibrations. SFA can use derivative differences at irregular times, whereas a
single discrete prediction matrix cannot silently represent different durations.

B is a calibration nuisance parameter and diagnostic, not a replacement for
the native cluster transitions. After calibration, B is frozen and saved
separately. Each cluster learns its own A/Gamma through the existing inference
scheme. A still advances between observations included in that cluster, not
automatically according to elapsed video time. The calibration does not impose
global linear dynamics on all final clusters or initialize their A from B.

## Gaussian inference versus raw features

The fixed observation still has

\[
 y_t\mid f_t\sim\mathcal N(\mu+C f_t,\Sigma).
\]

Its exact information encoding is H=C.T Sigma^-1 C and
h=C.T Sigma^-1(y-mu). The native LDS combines this factor with its dynamic
prior. Raw W.T(y-mu) is a calibration feature and need not equal the Gaussian
likelihood encoding when C is nonorthogonal and Sigma is diagonal. It is not
clamped into the latent state and not counted as another likelihood. Posterior
trajectories therefore need not achieve the raw features' calibration loss.
This extension does not introduce approximate nonlinear inference.

Independent use:

```python
from hdpgpc.emissions import estimate_linear_prediction_observation
from hdpgpc.config import with_fixed_observation_model

estimate = estimate_linear_prediction_observation(
    observations, grid, components=20, max_samples=400,
    whitening_components=64, prediction_weight=0.1,
    epochs=250, learning_rate=0.02, times=timestamps,
    segment_ids=sequence_ids, sampling='uniform', random_state=42,
)
raw_codes = estimate.transform(observations)
config = with_fixed_observation_model(config, estimate.model)
```

Arrays follow the same one-output structure as other affine calibrators.
The returned model is the same exact class accepted by compatible offline and
online schedules; only the CAS-Colon experiment requires offline histories for
its image trajectory reports.

## Reports and comparison

Outputs are in
`agents/cas_colon_linear_prediction_runs/<video_segment>/<run_id>/`.
The runner preserves image galleries, trajectory plots, projection, metrics,
majority-label timelines and `--min-cluster-samples` selection. Raw reconstruction
uses the trained W, not an assumed decoder pseudoinverse. The ordinary one-step
gallery row still uses the fitted cluster A.

Additional artifacts:

- `linear_prediction_calibration.npz`: raw encoder, B, offset, calibration row
  and pair indices/times, raw codes, loss history, best iteration and rank controls.
  Loss columns are total, reconstruction, prediction residual and predictor ridge.
- `linear_prediction_calibration.png`: loss history and calibration source,
  target, target reconstruction and B-predicted target images. These are
  in-sample calibration diagnostics, not held-out forecasts.
- `observation_calibration.npz`: the ordinary installed affine state, including
  the applied Sigma. Config and decoder state require no new model type or
  callback for save/load.

The log records the PCA-control and selected total objectives, reconstruction
fraction and selected optimizer step. The JSON metadata includes separate
control/best loss components and actual step count. With zero weight, step
count is zero even though the configured epoch budget is positive.

Compare reconstruction and trajectory quality across lambda values at the same
component count, frame sample, seed and clustering configuration. The objective
is nonconvex and in-sample; it can learn predictable illumination or camera
artifacts and can overfit B when few pairs support many latent dimensions.
Increasing the prediction weight can worsen reconstruction. No anatomical
performance improvement or faithful camera-motion recovery is assumed.
