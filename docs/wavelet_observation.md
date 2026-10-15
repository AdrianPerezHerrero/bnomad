# Fixed orthonormal wavelet observation

The full real Haar basis uses the existing `AffineGaussianObservation`.
Supply waveforms to HDPGPC as usual; the model encodes their Gaussian likelihood
into wavelet coordinates and reconstructs predictions in waveform space.
There is no separate clustering engine or nonlinear feature preprocessing.

```python
from hdpgpc.config import NoisePolicyConfig, with_fixed_observation_model
from hdpgpc.emissions import make_wavelet_observation

decoder = make_wavelet_observation(grid, Sigma=0.01, latent_covariance=0.1)
config = with_fixed_observation_model(
    base_config, decoder,
    sigma_policy=NoisePolicyConfig(mode='cluster_diagonal'),
)
```

`real_wavelet_basis(grid)` returns the synthesis matrix C. Currently the only
supported family is `wavelet='haar'`. The grid must be real, finite, increasing,
uniformly spaced, and contain N = 2^J points with J >= 1. No implicit padding or
truncation is performed. Explicitly resample before constructing the decoder
when a different point count is needed.

Columns are ordered as the constant scaling function, then details from coarse
to fine scale, left to right within each scale. The scaling column is 1/sqrt(N).
A detail supported on L consecutive points equals +1/sqrt(L) on the first
half, -1/sqrt(L) on the second half, and zero outside its support.
There are 1 + 1 + 2 + ... + N/2 = N columns, with

\[
C^T C = CC^T = I,\qquad z=C^T(y-b),\qquad y=b+Cz.
\]

All coefficients, including the mean and fine-scale details, are retained.
The transform preserves energy and spikes in the supplied waveform. It does
not remove background, threshold coefficients, or make events independent of
their position. A full basis with consistently transformed unrestricted priors
is a change of coordinates, not a guaranteed improvement in clustering.

`Sigma` specifies waveform-space noise, and `latent_covariance` specifies the
initial state prior. Each accepts a positive scalar variance, diagonal variance
vector, or positive definite full covariance. `offset` is an optional fixed
waveform mean, default zero. For this square orthonormal basis,
`encode_measurement(y)` returns z and covariance C.T @ Sigma @ C. `encode(y)`
instead returns the Gaussian posterior after combining the measurement with
the initial prior. Expected log likelihood and uncertainty remain in waveform
coordinates. C stays fixed while the existing policies control A, Gamma and
Sigma. Serialization stores the actual basis and arrays through the standard
affine state dictionary and configuration, with no new configuration system.

The ordinary fixed observation restrictions apply: one output, CPU, no warp,
fixed GP kernel, final LDS scoring. The existing offline conjugate batch and
sequential update paths, including online and online_fast, are used unchanged.

## ACSF1 example

```powershell
python examples/wavelet_clustering.py --points 64 --schedule offline
python examples/wavelet_clustering.py --schedule online_fast --sigma-mode cluster_diagonal
```

The example defaults to ACSF1 from the local UCR ZIP; use `--archive` to choose
the archive or `--download` to fetch it. It fits all TRAIN and TEST series by
default, normalizes using TRAIN statistics, and permutes rows with labels kept
aligned. Labels only contribute descriptive purity, ARI and NMI after fitting.
This is transductive clustering, not held-out classification evaluation.

UCR rows have no temporal order: HMM conditioning is disabled and A is fixed to
zero. Gamma learns per cluster in wavelet coordinates. `--sigma-mode` chooses
fixed, learned full/diagonal per cluster, or learned full/diagonal shared Sigma.
`--iterations`, `--max-clusters`, `--hdp-preset`, `--max-series`, `--seed`, and
`--verbose` control the experiment. The default 64-point resampling can discard
narrow bursts before the transform; increase `--points` to a power of two that
fits the original resolution if preserving such events is necessary.

Outputs default to ignored `agents/wavelet_example`: `config.yaml`,
`results.json`, and `<dataset>_wavelet_clusters_output_0.png`. The plot uses the
same summary function as the Fourier and MIT-BIH examples, is saved, and is
closed without displaying an interactive window.
