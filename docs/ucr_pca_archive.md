# UCR archive with a fixed PCA emission

```powershell
python examples/ucr_pca_clustering.py --datasets Coffee
python examples/ucr_pca_clustering.py --datasets ACSF1 --components 12
python examples/ucr_pca_clustering.py --resume
```

Omit `--datasets` to process every dataset in the local UCR archive. Use
`--download` if the official ZIP is absent, or `--archive` to choose its path.
The runner reuses the same archive loading and normalization as the direct
waveform experiment. The default output folder is ignored `agents/ucr_pca_results`.

## Calibration and model

Normalization uses TRAIN statistics. For each dataset, the existing
`estimate_pca_observation` pilot fits PCA to up to 400 randomly selected TRAIN
series, without labels. The default is 12 components, capped at waveform points
minus one and calibration-series count minus one. `--pca-samples`, `--components`
(also `--latent-dim`), `--max-points`, and `--seed` control calibration/resolution.
Datasets with fewer than two TRAIN series produce an error report.

The calibrated decoder is affine:

\[
y=b+Cz+\epsilon,\qquad \epsilon\sim N(0,\Sigma).
\]

C contains orthonormal PCA loadings; b is the calibration mean. Both stay fixed
through clustering. Initial latent covariance is the diagonal matrix of PCA
score variances. Sigma initially contains per-point reconstruction residual
variances. The pilot floors both types of variance at `--noise-floor` (1e-6).
`--process-noise` multiplies the latent prior variances; its default is 1.
`--observation-noise-scale` multiplies the residual Sigma prior, also default 1.
`--sigma-prior` and `--gamma-prior` additionally multiply these calibrated
covariances (both runner defaults are 0.001):

```text
Sigma_initial = sigma_prior * observation_noise_scale * calibrated_residual_covariance
Gamma_initial = gamma_prior * process_noise * calibrated_latent_covariance
```

These are covariance multipliers, not standard deviations and not prior degrees
of freedom. The latent initial-state covariance receives the same gamma factor.
Use `--sigma-prior 1 --gamma-prior 1` to retain the earlier calibrated noise
scales. Noise policies still decide whether these values stay fixed or initialize
learned posteriors. The saved calibration NPZ and configuration contain the
effective scaled matrices.

Installation uses the shared `with_fixed_observation_model` helper and
`AffineGaussianObservation` implementation. Filtering consumes waveform-space
observations through the exact affine Gaussian likelihood factor. Inference
does not simply treat unregularized PCA scores as error-free latent states.
Observation likelihoods and reconstructed cluster curves remain in waveform
space. No autoencoder or new inference engine is introduced.

UCR rows do not have meaningful temporal order. HMM conditioning is disabled
and A is fixed to zero. Gamma describes cluster-specific latent spread rather
than evolution through a sequence. Default Gamma is learned full per cluster;
default Sigma is learned diagonal per cluster. `--gamma-mode` and `--sigma-mode`
each accept fixed, cluster, cluster_diagonal, shared, or shared_diagonal. These
are existing policies compatible with fixed A and C.

The GP kernel remains fixed and alignment is disabled, as required by the
current affine PCA path. The common warp switch is accepted but enabling it is rejected for affine observations. Use the direct
waveform runner for DTW-GP comparisons.

## Execution and evaluation

The default schedule is offline with conjugate batch updates. `--schedule`
also supports online and online_fast using conjugate sequential updates.
Defaults include soft responsibilities, HDP preset max, excess LDS degrees 10,
ten birth exploration steps, thirty clusters and thirty offline iterations.
Use `--hard-assignments`, `--hdp-preset`, `--lds-degrees`, `--exploration-steps`,
`--max-clusters`, `--iterations`, and `--verbose` to change these settings.

TRAIN and TEST both participate in unsupervised clustering. A seeded permutation
keeps waveforms and labels aligned; `--max-series` optionally selects a random
subset. Neither normalization nor PCA uses TEST values. However, clustering
scores, including the TEST-only ARI, are **transductive descriptions**, not
held-out classification accuracy. The majority-label confusion mapping is
fitted and evaluated on the clustered observations.

Each dataset runs independently. Failures are logged and processing continues.
`--resume` skips successful datasets with the same archive/settings identity
and required saved plots. A settings/implementation change produces a new run ID.

## Outputs

Every dataset/settings combination has its own `<dataset>/<run_id>/` folder.
`settings.json` records source/code fingerprints; `config.yaml` is the resolved
recipe, and `clustering.log` captures console output. Common files include
`results.json`, `assignments.csv`, `lds_parameters.npz`, `projection.png` and
`clusters_output_0.png`. PCA calibration is saved as
`observation_calibration.npz`; TRAIN indices and explained variance are in
results metadata. Root `results.csv` aggregates status and common scores.

LDS plots are off by default for UCR. Enable `--lds-plots` and
`--lds-dynamics-plots` to inspect them. `--no-plots` skips all PNGs. Plots are
saved and closed without opening windows. The same annotated cluster gallery
is used for every representation. See the [common workflow guide](experiment_workflows.md)
for arguments, artifact names, aliases and resume behavior.
