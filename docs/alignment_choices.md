# Choose an alignment policy

Set `InferenceConfig(warp=True)` and select `AlignmentConfig(policy=...)`.
`warp=False` skips fitting a warp, regardless of the configured policy.
Alignment acts on the time coordinates of an observation; it does not replace
the GP emission or LDS model.

| Policy | What it estimates | Practical cost and interpretation |
| --- | --- | --- |
| `none` | Identity map | No alignment work; a useful baseline |
| `gp` | One monotone optimized point map | Iterative optimization per aligned observation; smoothness and displacement controls, with a GP displacement score after fitting |
| `dtw_gp` | A DTW path smoothed with a GP displacement prior | Usually cheaper than a trained flow; one point warp and a **negative** warp penalty in the observation score |
| `gp_flow` | A variational distribution over monotone GP-drift ODE warps | Multiple sampled ODE solves and optimization steps; represents warp uncertainty but is normally the most expensive option |

These are qualitative algorithmic costs, not measured runtime ratios. Length
of a waveform, number of clusters and proposals, GP covariance work, sample
count, ODE steps, and offline refits can change wall time substantially.
For `dtw_gp`, the coarse dynamic-programming path uses work proportional to
the square of `dtw_points`; the inducing GP solve has cubic work in
`dtw_inducing` when its covariance is factored. `gp_flow` repeats sampled ODE
integration for each training step and candidate. Its cost therefore grows
at least with `flow_train_iter * flow_samples * flow_steps`; dense
observation-covariance scoring can add a cubic factor in waveform points.
The `gp` policy repeats its optimization objective for each configured
training iteration. These counts omit the underlying cluster GP/LDS work.
In online inference each candidate warp should be conditioned on the current
cluster GP representer and the new observation. `warp_posterior(...)` exposes
that candidate-conditioned posterior for the `gp_flow` policy without
committing the observation.

The `gp` implementation is a deterministic monotone waveform aligner. Its
optimized loss uses data fit, curvature, and displacement penalties; its GP
displacement prior is scored after optimization. It is **not** the full
probabilistic AMTGP model. `dtw_gp` also supplies a point estimate; the GP
prior smooths and penalizes unusual displacement, but it does not propagate
warp uncertainty into predictions. See [DTW–GP details](dtw_gp_alignment.md).

`gp_flow` uses sampled GP drift functions and ODE integration, inspired by
the monotonic flows in [Ustyuzhaninov et al. (2020)](https://proceedings.mlr.press/v108/ustyuzhaninov20a/ustyuzhaninov20a.pdf)
and [Mikheeva et al. (2022)](https://proceedings.mlr.press/v151/mikheeva22a/mikheeva22a.pdf).
The package keeps its own cluster GP and LDS emission; it does not implement
the paper's complete aligned multi-task GP. Sampling a warp may be fast in a
different experimental setup, while repeated candidate-conditioned training
and dense emission scores over a full clustering record remain costly here.
See [GP-flow details](gp_flow_alignment.md) for objective, controls, and limits.

Start with `none` to understand the unaligned data. Use `dtw_gp` to inspect
whether a smooth point warp changes assignments. Use `gp_flow` when the
posterior over possible alignments matters and its extra computation is
acceptable. A warp can also erase scientifically meaningful shape differences,
so inspect the fitted maps alongside cluster diagnostics.
