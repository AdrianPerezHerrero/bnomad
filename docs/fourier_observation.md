ï»¿# Fixed real windowed Fourier observation

Fourier coordinates use the existing `AffineGaussianObservation` and
`with_fixed_observation_model` installer. Inputs, likelihoods and predictive
plots remain in waveform space. The basis now uses fixed non-overlapping
rectangular windows rather than one global DFT.

```python
from hdpgpc.config import NoisePolicyConfig, with_fixed_observation_model
from hdpgpc.emissions import make_fourier_observation

decoder = make_fourier_observation(
    grid, window_size=16, Sigma=0.1, latent_covariance=1.0,
)
config = with_fixed_observation_model(
    base_config, decoder,
    sigma_policy=NoisePolicyConfig(mode='cluster_diagonal'),
)
```

All coefficients are retained by default. With 64 waveform points and
16-point windows, C consists of four independent 16-by-16 Fourier blocks and
64 latent coordinates. `real_fourier_basis` returns the synthesis matrix.
There is no padding, overlap, tapering, phase stripping or power-spectrum
preprocessing. Window length must be an integer >= 2; a length exceeding the
series length produces one window containing the series.

## Coordinates and reconstruction

Columns are ordered by window from left to right, then by local frequency:
DC, cos(1), sin(1), ..., with one Nyquist cosine for even window lengths.
For a window of L points indexed locally by j, its nonzero entries are

\[
C_{j,0}=1/\sqrt L,\quad
C_{j,k,\cos}=\sqrt{2/L}\cos(2\pi kj/L),\quad
C_{j,k,\sin}=\sqrt{2/L}\sin(2\pi kj/L).
\]

The even-length Nyquist column is (-1)^j/sqrt(L). Entries outside a column's
window are zero. The final window may be shorter; a one-point tail has a unit
DC column. With all coefficients, C.T @ C = C @ C.T = I and
`z = C.T @ (y-offset)` reconstructs exactly as `offset + C @ z`.
Coefficients within each window match its orthonormal rFFT: real DC, then
sqrt(2) times the real part and minus sqrt(2) times the imaginary part of
ordinary positive bins, with a real Nyquist coordinate for even lengths.

Optionally pass `n_harmonics=h` to retain bins 0..h in each window, or
`frequency_bins=[0, 2, 5]` to select local bins. These are mutually exclusive.
Selections must fit the longest window; unavailable bins are omitted in a
short final window. If none are available, that window has no latent coordinates
and is explained by the offset and observation noise. Selected columns remain
orthonormal, but omitted frequencies cannot be reconstructed and remain in
waveform residuals used to evaluate the likelihood.

Physical frequency for bin k in a length-L window is k/(L * grid_spacing).
A shorter tail consequently has different physical bin frequencies.
All grids must be real, finite, increasing and uniformly spaced.
Windowing localizes oscillations and repeated activity but retains phase and
position. It does not make displaced spikes identical or remove background.
A full basis with consistently transformed unrestricted priors and covariances
remains a change of coordinates, not a guaranteed clustering improvement.

## LDS integration and persistence

The model remains `y_t = offset + C f_t + epsilon_t`. C stays fixed, Sigma is
waveform-space noise, and A/Gamma govern evolution between observations,
not between windows within a waveform. Existing fixed or learned full/diagonal
per-cluster/shared Sigma policies remain available.

Sigma and latent_covariance accept a positive scalar variance, diagonal
variance vector or positive definite full covariance. Offset is an optional
fixed waveform mean, default zero. `encode_measurement` supplies the latent
Gaussian measurement; with isotropic Sigma its mean is C.T @ (y-offset).
`encode` also incorporates the initial latent prior and need not equal the
unregularized Fourier coefficients.

Saved recipes store the actual C and other affine arrays. Older global-Fourier
recipes retain their stored decoder when loaded. To explicitly construct the
former global basis, use `window_size=len(grid)`. The factory's new default
is 16-point windows, and frequency selections are local to each window.
No new configuration system, callback or inference engine is introduced.

Ordinary fixed-affine restrictions apply: one output, CPU, no alignment warp,
fixed GP kernel, final LDS scoring, and existing sequential/natural-gradient
or offline batch LDS updates.

## ACSF1 example

```powershell
python examples/ucr_fourier_clustering.py --schedule offline --window-size 16
python examples/ucr_fourier_clustering.py --window-size 16 --harmonics 4
python examples/ucr_fourier_clustering.py --schedule online_fast --sigma-mode cluster_diagonal
```

The example defaults to ACSF1 from the local UCR archive. Use `--archive`,
`--download` or `--dataset` to select the input. All TRAIN and TEST series
participate in fitting by default; labels only provide descriptive transductive
clustering metrics. The loader interpolates missing values, resamples to at
most --points (default 64), normalizes using TRAIN statistics and permutes
rows with labels kept aligned. Downsampling can remove narrow events before
Fourier encoding.

UCR rows have no temporal ordering: HMM conditioning is disabled and A is fixed
to zero. Gamma learns per cluster; Sigma follows --sigma-mode. The ordinary
configuration can still learn A for ordered data. Iteration, cluster-count,
seed and subset controls remain available.

Outputs use the common `<dataset>/<run_id>/` layout under
`agents/ucr_fourier_results`. Cluster galleries, projections, calibration arrays,
configuration, logs, assignments and JSON/CSV metrics use the same schema as
PCA and the autoencoder. LDS plots are optional. See
[experiment workflows](experiment_workflows.md). The example defaults to ACSF1;
use `--datasets all` for the archive. The singular `--dataset` remains an alias.
