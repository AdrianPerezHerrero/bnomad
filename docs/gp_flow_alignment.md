# Variational GP-flow alignment

`AlignmentConfig(policy='gp_flow')` selects an opt-in uncertain input warp for
HDP-GPC. The default `policy='gp'` retains the existing alignment algorithm.
For a cheaper GP-smoothed point estimate, see [DTW–GP alignment](dtw_gp_alignment.md).
The new policy follows the monotonic GP-flow construction of
[Mikheeva et al. (AISTATS 2022)](https://proceedings.mlr.press/v151/mikheeva22a.html)
while keeping HDP-GPC's existing cluster GP and LDS emissions. Consequently it
is **not** the complete Aligned Multi-Task Gaussian Process model: the paper's
sparse multi-task emission GP and task-embedding posterior are not present.

```python
from dataclasses import replace
from hdpgpc import HDPGPC
from hdpgpc.examples import configuration

config = configuration('dynamic')
config = replace(
    config,
    alignment=replace(config.alignment, policy='gp_flow',
                      flow_inducing=10, flow_features=64,
                      flow_samples=4, flow_steps=10,
                      flow_train_iter=30),
    inference=replace(config.inference, schedule='online', warp=True),
)
model = HDPGPC(config)
model.update(first_waveform)
candidate_posteriors = model.warp_posterior(next_waveform)
model.update(next_waveform)
```

For batch inference, use `schedule='offline', warp=True` and `model.fit(X)`.
`warp_posterior` also works after a batch fit. It returns one independent,
read-only `WarpPosterior` per current candidate, or a single-entry dictionary
when `cluster=k` is supplied. Each contains sampled warped grids, posterior
mean and standard deviation of displacement, whitened Gaussian parameters,
expected log likelihood, KL and the candidate score by output. It does not
assign or commit the next observation. The results are conditioned on each
candidate cluster; this method does not combine them into one posterior by
marginalizing the cluster assignment.

For one observation `i` and candidate cluster `k`, the policy fits a Gaussian
posterior over whitened drift inducing values. A coherent pathwise GP draw is
held fixed throughout each Euler ODE solve. Its flow produces a sampled
monotonic input grid `g_i(x_i)`. The observed values `y_i` are left unchanged.
The candidate objective uses

`E_q(g_ik)[log p(y_i | F_k(g_ik(x_i)))] - KL(q_ik(v) || p(v))`.

The warp is shared across all output channels of the observation. The KL is
counted once per candidate by dividing it across the per-output scores before
the existing output aggregation. The offline HDP responsibility weighting then
weights those candidate scores. Given a fixed sampled warp posterior, the
Gaussian emission state update averages the observation precision and linear
natural parameter over sampled grids; it does not substitute the mean grid for
the expectation. For the current Gaussian prior `N(m,P)`, responsibility
`r_ik`, and sampled warp-specific observation matrix `H_g` and residual
covariance `R_g`, the state update uses

`J = P^{-1} + r_ik E_q(g)[H_g^T R_g^{-1} H_g]`,

`eta = P^{-1} m + r_ik E_q(g)[H_g^T R_g^{-1} y_i]`,

then `covariance = J^{-1}` and `mean = J^{-1} eta`. In the online schedule this update is applied to the newly
arrived observation. In the offline schedule, candidate posteriors are
re-estimated from the current cluster GP representers during the existing
batch iterations, then used by the emission pass.

The input grid must be strictly increasing. `flow_lengthscale` and
`flow_amplitude` control the drift GP on the grid normalized to `[0, 1]`;
`flow_inducing`, `flow_features`, `flow_samples`, `flow_steps`,
`flow_train_iter`, and `flow_lr` control the approximation and optimizer.
These settings are part of the configuration codec. The fitted model and
online posteriors can be pickled; `HDPGPCConfig.to_dict()` only stores the
construction recipe, not fitted posterior state.

Current limits: the emission likelihood is the existing per-cluster GP/LDS
predictive distribution, not the paper's multi-task GP. Its Gaussian
expectation and flow ELBO use Monte Carlo samples, so scores vary with the
random seed and are more expensive than point alignment. The native online
engine has a separately characterized multi-output failure, including with
alignment disabled. `classify_batch(..., warp=True)` and
`fit_labels(..., warp=True)` do not implement this policy and raise
`NotImplementedError`.

## Warp diagnostics

`hdpgpc.diagnostics.model_warp_arrays(model, grid, output=0)` extracts the
stored per-observation displacement and, for `gp_flow`, posterior standard
deviation. `plot_warp_diagnostics(grid, displacements, labels=..., posterior_std=...)`
returns a Matplotlib figure with selected warp maps, a displacement heatmap
covering every observation, and cluster mean curves with a 10–90% range.
Optional `grid_samples` overlays posterior warp draws in the map panel.
When posterior standard deviations are available, dotted curves show twice
the mean within-observation posterior standard deviation; that is distinct
from variation between observations. The figure is returned for the caller
to save, display or close. `examples/full_record_clustering.py` can generate
these plots with `--plot-warps`; `--no-plot-warps` disables them.
