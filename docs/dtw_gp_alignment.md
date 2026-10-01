# DTW–GP point alignment

`AlignmentConfig(policy='dtw_gp')` selects a cheap deterministic input warp.
The default remains `policy='gp'`. `policy='gp_flow'` remains available for a
variational posterior over GP-flow warps. Set `warp=True` for either inference
schedule.

```python
from dataclasses import replace

config = replace(
    config,
    alignment=replace(config.alignment, policy='dtw_gp',
                      dtw_points=24, dtw_band=0.15,
                      dtw_inducing=10, dtw_lengthscale=0.2,
                      dtw_amplitude=0.1, dtw_anchor_noise=0.04),
    inference=replace(config.inference, schedule='offline', warp=True),
)
```

For each observation and candidate cluster, the warper first evaluates the
current cluster GP/LDS predictive mean on the original grid. It finds a DTW
path on at most `dtw_points` sampled positions, within `dtw_band` of the
diagonal (a fraction of the sampled grid length). The DTW distance compares
standardized waveform shapes jointly across output channels. Only the time
coordinates are changed; the original observation amplitudes go into the
existing GP emission likelihood and state update.

The path gives displacement anchors `d_i = g_DTW(x_i) - x_i`. On coordinates
normalized to `[0, 1]`, the prior has RBF covariance
`k(s,t) = amplitude² exp(-(s-t)² / (2 lengthscale²))`. With inducing coordinates
`U`, let `K_UU = L Lᵀ` and `B = K_XU L⁻ᵀ`. The whitened inducing values have
prior `z ~ N(0,I)`. Treating the DTW anchors as noisy measurements with
variance `dtw_anchor_noise²`, the regularized estimate is

`ẑ = (I + BᵀB / noise²)⁻¹ Bᵀ(d/span) / noise²`,

`g_smooth(x) = x + span B ẑ`.

The final map is projected to strictly increasing coordinates with fixed
endpoints. This last projection preserves a usable input grid; it means the
whole two-stage procedure is **not** an exact MAP solution of a joint
DTW–GP model. The candidate score is the existing cluster GP emission log
likelihood evaluated at this one warped grid, minus the inducing GP prior
penalty `½ ẑᵀẑ`. The penalty is counted once per observation, split across
channels before existing output aggregation. Offline responsibilities weight
candidate scores and the existing emission updates use the selected warped
input grids. Online inference estimates the candidate warps from the current
model and the next observation, including a separate birth proposal.

This policy does not infer a warp distribution. `warp_posterior` is available
only with `gp_flow`. Warp diagnostics show the fitted point maps without a
posterior uncertainty band. The score is an approximate point alignment
criterion, not an ELBO or marginal likelihood. A small `dtw_band`, moderate
`dtw_points`, and `dtw_inducing` keep warp inference light, although the
existing GP/LDS and HDP updates still determine total record run time.

`examples/full_record_clustering.py` exposes `ALIGNMENT_POLICY` and the DTW
controls together. Select `"dtw_gp"` there to use this policy for a record.
