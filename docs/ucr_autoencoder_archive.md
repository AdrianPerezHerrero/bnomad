# UCR archive with a fixed autoencoder emission

```powershell
python examples/ucr_autoencoder_clustering.py --datasets Coffee --components 12
python examples/ucr_autoencoder_clustering.py --datasets ACSF1 --epochs 60
python examples/ucr_autoencoder_clustering.py --download --resume
```

The runner shares its loader, config/run stages and result schema with
[UCR PCA](ucr_pca_archive.md). Omit `--datasets` to run the full archive.
TRAIN statistics normalize both splits; only TRAIN calibrates the encoder.
TRAIN+TEST are clustered together, so all scores are transductive descriptions.

The reusable `hdpgpc.emissions.estimate_autoencoder_observation` trains a tied,
orthonormal **linear** encoder/decoder with an affine offset and scale. It
freezes the calibrated decoder before clustering. This is the exact Gaussian
fixed observation path, with a waveform-space likelihood; it does not introduce
nonlinear approximate inference or replace HDPGPC with a latent clustering model.
The calibration network is reused only for diagnostics after fitting.

Defaults retain up to 400 calibration TRAIN series, 40 latent components
(capped by calibration rank and waveform dimension), 60 epochs, fixed isotropic
Sigma, a 0.1 multiplier on calibrated observation noise, and a 0.05
multiplier on calibrated latent covariance. Use `--components`,
`--calibration-samples`, `--epochs`, `--process-noise` and
`--observation-noise-scale` for controlled comparisons. The PCA script uses
12 components and a latent covariance multiplier of 1 by default: set these
explicitly, together with observation-noise scaling, to compare representations
under matching dimensions/scales.

The UCR `--sigma-prior` and `--gamma-prior` arguments also apply to the decoder:
they multiply calibrated Sigma and latent covariance respectively, after the
existing observation/process noise-scale controls. Both default to 0.001 in the
shared runner. To retain the previous calibrated values, explicitly use
`--sigma-prior 1 --gamma-prior 1`. The gamma multiplier affects both initial
latent-state covariance and the Gamma noise prior. These factors multiply
variances/covariances without squaring them; fixed/learned policies are unchanged.

A=0 and HMM conditioning is off, since archive row order is arbitrary. Gamma
learns per cluster. `--gamma-mode` and `--sigma-mode` select existing compatible
noise policies; C and the decoder offset remain fixed. GP kernel fitting and
warping are excluded by the current fixed affine implementation.

Outputs default to `agents/ucr_autoencoder_results/<dataset>/<run_id>/`.
They include the shared cluster gallery and code projection, calibration model,
training losses/indices, and an AE calibration reconstruction plot. LDS matrix
and dynamics plots are optional. No transition heatmap is drawn for UCR.
See [experiment workflows](experiment_workflows.md) for the common CLI, exact
file schema, run identities, failure handling and resume semantics.
