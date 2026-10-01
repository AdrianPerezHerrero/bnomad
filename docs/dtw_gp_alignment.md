# DTW-GP point alignment

`AlignmentConfig(policy='dtw_gp')` selects a deterministic input warp. Set
`warp=True` for online or offline inference. The GP warp controls live in
`AlignmentConfig`; the full-record example uses their package defaults.

```python
from dataclasses import replace

config = replace(
    config,
    alignment=replace(config.alignment, policy='dtw_gp',
                      dtw_points=24, dtw_inducing=10,
                      dtw_lengthscale=0.2, dtw_amplitude=0.005,
                      dtw_anchor_noise=0.0005),
    inference=replace(config.inference, schedule='offline', warp=True),
)
```

For each observation and candidate cluster, the warper evaluates the current
cluster GP/LDS predictive mean on the original grid. Unrestricted DTW finds a
path on at most `dtw_points` sampled positions. It compares standardized
waveform shapes across output channels. The observed amplitudes are retained
for the emission likelihood and state update.

DTW supplies displacement anchors `d = g_DTW(x) - x`. Coordinates and
displacements are normalized by the grid span. An RBF GP with amplitude `a`
and lengthscale `l` has covariance `k(s,t) = a^2 exp(-(s-t)^2/(2 l^2))`.
For inducing coordinates `U`, let `K_UU = L L^T` and `B = K_XU L^-T`.
With whitened coefficients `z ~ N(0,I)` and anchor noise `sigma_a`, the
smoothed coefficients and map are

```text
z_hat = (I + B^T B / sigma_a^2)^-1 B^T (d/span) / sigma_a^2
g_smooth(x) = x + span B z_hat.
```

The final map `g` is projected to increasing coordinates with fixed endpoints.
The penalty is evaluated on **this final map**, not on the map before
projection. Write `v = (g(x)-x)/span` and `K = B B^T + sigma_a^2 I`. The
candidate's warp penalty is

```text
P(g) = 1/2 v^T K^-1 v
     = 1/2 (||z_v||^2 + ||v - B z_v||^2 / sigma_a^2),
z_v = (I + B^T B / sigma_a^2)^-1 B^T v / sigma_a^2.
```

This is the GP displacement prior quadratic with a nugget; its log determinant
is constant across candidate clusters on the same grid. The candidate
emission score is `log p(y | g, cluster GP) - P(g)`. Both inference schedules
place this score in the observation terms used for assignments and birth
comparisons. In offline inference, the selected observation term is summed
over assigned beats in the existing objective. The penalty follows the
package's existing output weighting: it is divided by the number of outputs
in each channel score, so it enters exactly once when those channel scores
are summed with unit weights. Other SNR weights change its effective scale.

This is a **point-warp surrogate objective**, not a rigorous variational ELBO:
DTW and GP smoothing are separate from the subsequent monotonic projection,
and no distribution over warps is inferred. A new cluster is encouraged only
when its improvement in this overall objective outweighs the existing HDP
birth costs. Every candidate still receives a warp, including a discordant
beat. `gp_flow` remains the policy for a variational warp posterior.

The nonnegative `P(g)` is **subtracted** from the observation score, so a
larger penalty makes the objective smaller. Online verbose output prints this
signed contribution as `warp_ELBO_term=-P(g)`. The emission term and reported
total already include it; it must not be subtracted again.

Smaller `dtw_amplitude` penalizes displacement more strongly. Larger
`dtw_lengthscale` favors broader, smoother changes. `dtw_anchor_noise` sets
both how closely the smoothed map follows DTW anchors and the nugget in the
final-map penalty. Scaling both amplitude and anchor noise by a factor `c`
leaves the smoothed warp approximately unchanged while scaling the penalty
for a fixed warp by `1/c^2`. The defaults were changed from `0.01, 0.001`
to `0.005, 0.0005`, giving about four times the penalty for a fixed warp.
`dtw_points` and `dtw_inducing` control DTW resolution and GP flexibility.
There is no DTW path band or displacement cutoff.
