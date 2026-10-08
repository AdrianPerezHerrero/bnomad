# Choose a model setup

Start with the [small offline or online examples](tutorials.md). The options
below change how clusters explain a sequence of waveforms; none requires an
ECG dataset. An observation has shape `(points, outputs)` and a batch has shape
`(observations, points, outputs)`.

## The first decisions

| Decision | Starting choice | Change it when |
| --- | --- | --- |
| Inference schedule | `offline` when the record is available; `online` for arriving observations | You need sequential updates or cannot hold the whole record |
| LDS maps and noise | `all_free` (the original cluster-specific approach) | You want to constrain how much a cluster's observation map or dynamics can adapt |
| Latent dimension | Full observation grid | Full LDS matrices are too large, or a lower-dimensional state is scientifically appropriate |
| Warp | `warp=False` | Timing variation is obscuring shape differences |
| Birth prior | `balanced` HDP preset | Birth proposals are too frequent or too rare *after inspecting all score terms* |

The LDS is `f_t = A_k f_(t-1) + eta_t` and `y_t = C_k f_t + epsilon_t`,
with process covariance `Gamma_k` and observation covariance `Sigma_k`.
Cluster labels normally form a Markov chain. The `static_mixture` preset
instead makes labels independent and removes the LDS transition process.

## LDS presets

`with_lds_preset(config, name)` fills the existing matrix and noise policies.
`shrinkage` means a learned map pulled toward identity by a stronger prior;
it does not freeze it. `shared` means one learned parameter for all clusters.

| Preset | Main effect |
| --- | --- |
| `all_free` | Each cluster learns all four LDS parameters |
| `plastic` | Shrink `C` and strengthen the `Sigma` prior; favor explanation through dynamics |
| `stable` | Shrink `A` and strengthen the `Gamma` prior; favor a steadier latent process |
| `only_plastic` | Fix `C` and `Sigma`; learn `A` and `Gamma` |
| `only_stable` | Fix `A` and `Gamma`; learn `C` and `Sigma` |
| `shared_emission` | Learn one `C,Sigma` pair across clusters |
| `shared_latent_process` | Learn one `A,Gamma` pair across clusters |
| `all_shrunk` | Shrink both maps and strengthen both noise priors |
| `all_fixed`, `random_walk` | Fix all four; `random_walk` forces identity `A,C` |
| `shared_all` | Share both learned pairs across clusters |
| `diagonal_noise` | Learn per-cluster maps and diagonal noise matrices |
| `static_mixture` | Independent cluster labels and no latent transition |

These are prior choices, not guaranteed fitted behaviors. A fixed LDS does
not by itself remove Markov dependence between *cluster labels*. The complete
matrix-mode table, prior mathematics, and update/scoring choices are in
[LDS inference](lds_inference.md).

```python
from hdpgpc.config import with_lds_preset

config = with_lds_preset(config, 'stable', shrink_strength=10.0)
```

## Reduce the state dimension

Two different reductions are available:

- **Inducing LDS basis:** choose fewer coordinates for `config.x_basis`, set
  `emission.inducing_points=True`, and pass the original full grid to `fit`.
  The GP projects waveforms to the shorter LDS basis. `A`, `Gamma`, `C`, and
  `Sigma` then live on that basis. Its conjugate update omits uncertainty in
  this projection, so it is an approximation.
- **PCA observation:** fit `estimate_pca_observation` on a representative
  minibatch, then call `with_pca_emission`. This fixes a rectangular PCA
  decoder `C` from latent coefficients to the *full waveform*. `A,Gamma`
  remain cluster-learned. `Sigma` may be fixed, learned full or diagonal per
  cluster, or learned full or diagonal across clusters.

The printed PCA explained variance is measured on its calibration minibatch.
Inspect reconstruction error on other observations before interpreting it as
evidence that the decoder captures every regime. See [LDS inference](lds_inference.md)
for PCA setup and limits.

## HDP and alignment

The current `less`, `balanced`, and `more` HDP presets are fixed bundles of
`gamma`, `transAlpha`, `startAlpha`, and `kappa`. Their names do **not** give a
monotonic birth penalty in every setting. See [HDP presets and sensitivity](hdp_presets.md)
for current values and the measured gamma/transAlpha tables.

`gp` is the existing deterministic GP aligner. `dtw_gp` is a cheaper
deterministic warp with a smooth GP displacement prior; `gp_flow` fits a
variational warp posterior with ODE solves and is usually much more expensive.
See [alignment choices](alignment_choices.md) before enabling one.

## Prior noise scales and scoring

`estimate_lds_noise_priors` is an optional *pre-fit* pilot. It estimates the
order of observation and process variances under an identity random-walk
assumption. Do not interpret its values as general estimates of every LDS
matrix. Offline `conjugate_batch` and sequential `conjugate_sequential` update
LDS parameters differently; they need not return identical final matrices.
The usual score uses final LDS matrices for all beats. Historical scoring and
division of the LDS birth KL by latent dimension are experimental objectives,
not the ordinary model ELBO. [LDS inference](lds_inference.md) explains both.
