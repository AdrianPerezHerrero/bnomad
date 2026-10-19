# CAS-Colon: ordered images with an affine autoencoder

Use the same extracted frames and annotations as the [PCA experiment](cas_colon_pca.md).
This runner changes the calibrated representation while retaining the dataset
loader, chronological segments, HDP-GPC inference and image trajectory reports.

```powershell
python examples/cas_colon_autoencoder_clustering.py --records 1
python examples/cas_colon_autoencoder_clustering.py --records 9 --image-size 24 24 --components 8 --max-series 150
python examples/cas_colon_autoencoder_clustering.py --records 1 --calibration-samples 400 --epochs 250 --learning-rate 0.02
python examples/cas_colon_autoencoder_clustering.py --records all --resume
```

## What is trained

The existing **tied linear autoencoder** is trained on up to 400 frames from each
selected contiguous segment. Defaults are 32x32 grayscale pixels in [0,1], ten
latent dimensions, 250 Adam epochs and learning rate 0.02. Dimensions are capped
by the calibration sample count and image size. Uniform sampling is seeded;
`--calibration-sampling prefix` uses the first calibration frames instead.

For calibration mean mu, global scale s and an orthonormal learned basis Q:

```text
encoder:       z = Q.T @ (y - mu) / s
decoder:       y = mu + C @ f + noise,   C = s * Q
noise:         Gaussian(0, Sigma)
latent model:  f_next = A @ f + process noise
```

The encoder is applied to every frame for the projection report. In inference,
the fixed observation model provides the equivalent affine encoding and its
uncertainty. The LDS estimates latent states by conditioning and smoothing;
states are not forcibly set to noiseless encoder outputs. The common decoder,
offset and encoder coordinates remain fixed during clustering. A and Gamma are
learned per cluster by the existing model. HMM transitions remain enabled.

This is an exact affine Gaussian extension of HDP-GPC. It is not a nonlinear or
convolutional image autoencoder. A converged linear autoencoder targets the same
reconstruction subspace as PCA; its learned coordinates can be rotated and
scaled differently. It therefore provides an integration comparison, rather
than an additional nonlinear representation capability.

## Covariances and controls

- **Sigma:** diagonal sample variance of autoencoder reconstruction residuals.
  Each variance is floored at the larger of `--noise-floor` (default 1e-6) and
  `--noise-floor-fraction` times the calibration pixel variance (default 0.05).
  `--sigma-prior` multiplies this covariance once; default 1 preserves it.
  Sigma stays fixed by default. For example, `--sigma-mode cluster_diagonal`
  learns per-cluster diagonal observation noise with this prior center.
- **Gamma:** prior mean `--gamma-prior * I`, default 0.01, learned per cluster.
  It is not estimated from temporal differences of encoded frames.
- **Initial state covariance P0:** covariance of calibration encoder outputs,
  with a small ridge. This is separate from Gamma.
- **Prior concentration:** `--lds-degrees` controls learned covariance priors.

The AE divides codes by a global calibration scale; PCA uses unscaled principal
coordinates. Identical numerical Gamma priors across the two runners need not
impose identical regularization in image space. Inspect reconstructions and
trajectories when comparing their results.

Shared inference/prior options and the same restrictions as the PCA runner
apply: offline inference, final scoring, non-legacy LDS updates, no warp and no
GP kernel fitting. `--a-mode fixed` selects A=0 for an independence comparison;
`--gamma-mode fixed` requires A fixed. `--no-hmm-switch` separately removes the
Markov assignment model. These controls use the existing typed configuration.

## Outputs

Each video/segment gets a run directory:

```text
agents/cas_colon_autoencoder_runs/video_1_segment_00/<run_id>/
  config.yaml, settings.json, clustering.log, results.json
  assignments.csv, observation_calibration.npz, training.npz
  lds_parameters.npz, latent_trajectories.npz
  clusters_output_0.png, projection.png
  latent_trajectories.png, autoencoder_calibration.png
```

The image summary compares observed images, encoder/decoder reconstructions,
smoothed LDS reconstructions and final-A one-step predictions. The summary
header includes purity, ARI and NMI. The taller timeline section includes
a third scatter plot of regimes colored by their assigned majority anatomical
label, matching reference-label colors. Ambiguous labels are excluded from the
majority vote; clusters without a reference label are gray. Each gallery samples
up to ten distinct observations using `--seed`, displayed in time order;
`--frames-per-cluster N` changes that maximum. The calibration
figure shows training loss and example image reconstructions. `training.npz`
stores calibration indices and losses; `observation_calibration.npz` contains
the fixed decoder, offset, initial covariance and applied Sigma. These affine
parameters also determine the encoder through the decoder pseudoinverse.
`--no-plots` disables all figures; saving does not open GUI windows.

Labels are used only for reporting. Identical images with ambiguous anatomical
annotations are fitted once and excluded from label metrics, as in the PCA
runner. Calibration sees the selected segment; metrics describe in-sample
clustering. No held-out anatomical recognition performance is claimed. Each
discontinuous segment gets its own autoencoder and model, so its coordinates
should not be compared directly with those from another segment.
