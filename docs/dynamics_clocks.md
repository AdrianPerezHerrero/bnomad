# Cluster-clock and physical-clock dynamics

Both clocks support the optional [constant latent input](latent_equilibrium.md),
enabled with `--constant-input` or `--latent-equilibrium`. This adds an affine
transition intercept without changing which observations define a clock step.

The default remains **cluster-clock**. Select the optional shared sampling clock
through the ordinary typed configuration:

```python
from dataclasses import replace
from hdpgpc import HDPGPC

physical_config = replace(config, dynamics=replace(
    config.dynamics, clock='physical-clock'))
model = HDPGPC(physical_config).fit(observations, grid=observation_grid)
```

Every standard experiment parser also accepts:

```sh
python examples/flutter_autoencoder_clustering.py --records 0950 --dynamics-clock physical-clock --resume
python examples/xjtu_sy_pca_clustering.py --records Bearing1_1 --dynamics-clock physical-clock --resume
```

Use `--dynamics-clock cluster-clock` to request the default explicitly. Existing
recipes without a clock field retain their behavior. Default serialization
omits this new default field so the pinned reference recipe/hash remains valid;
physical-clock is recorded explicitly. The CLI settings, saved configuration,
and logs identify the selected clock.
The XJTU-SY Fourier/Haar experiments explicitly default to physical-clock;
their `--dynamics-clock cluster-clock` option still selects legacy evolution.
Other defaults, including the core configuration and XJTU PCA/AE, remain unchanged.

## State evolution

In cluster-clock mode, component k evolves between its retained observations:

```
z[k, j] = A[k] z[k, j-1] + process_noise
```

Here j counts observations retained by that component. If global observations
1, 2 and 8 are retained, the last transition is one application of A.

In physical-clock mode, each component has a state at every global sampling tick:

```
z[k, t] = A[k] z[k, t-1] + process_noise
observation[t] = C[state[t]] z[state[t], t] + observation_noise
```

An unassigned component receives a prediction-only step. Its mean propagates
with A and its covariance with `A P A.T + Gamma`; it receives no measurement.
Over a gap of d sampling intervals the equivalent propagation is:

```
mean_new = A**d mean_old
P_new = A**d P_old (A**d).T + sum(A**i Gamma (A**i).T, i=0,...,d-1)
```

These are matrix powers. Thus a return from global observation 2 to 8 propagates
six transitions. The initial prior is the state at global tick zero, including
for a component first observed later. No cluster's state resets merely because
another cluster is assigned an observation.

This remains a set of **parallel component processes** with a switching
observation assignment. It is not a single common mechanical state whose
transition matrix switches with the active regime; that is a different switching
state-space formulation.

## Sampling and ODE interpretation

Physical-clock assumes a **regular, shared input sampling grid**. One application
of A represents one supplied observation interval. The `grid` argument remains
the within-observation GP/feature coordinate; it is not a physical timestamp.
There is no automatic timestamp interpolation or continuous-time generator fit.
An arbitrary irregular-time ODE model is outside this option's contract.

For uniform sampling of a linear ODE `dz/dt = F z`, A can represent
`exp(F * sampling_interval)` when the latent representation supports that model.
It is a finite-time map, not the ODE derivative matrix itself. This interpretation
requires validation of the representation, predictions, and model uncertainty.

Flutter delay windows advance by about 0.1 seconds with the defaults. XJTU-SY
observations advance by one acquisition minute; their raw vibration bursts are
not phase-synchronized. Physical-clock therefore does not make XJTU's latent
transition a fast vibration-mode estimate. Dataset selection such as `--step 2`
changes the interval represented by a fitted A to two acquisition minutes.
Fixed dynamics/noise values must be supplied in the corresponding interval's
units. Irregular expression sampling remains irregular physical time even when
rows are processed on a uniform model clock.

## Filtering, learning and structural moves

The physical trajectory owner is `hdpgpc.dynamics._clock`. It keeps propagated
states separately from retained measurement records. Existing emission histories,
observation policies and alignment interfaces remain observation-indexed views.
Full trajectories support backward smoothing through gaps and include uncertainty
from unobserved evolution.

Positive responsibilities weight measurement conditioning and observation
statistics. Missing measurements contribute no observation mass, measurement-noise
statistics, or fabricated zero-valued data. Their latent transitions can contribute
expected sufficient statistics after smoothing from actual observations. Trailing
prediction-only states are not used as additional parameter-learning evidence.
Online prediction-only commits do not learn parameters.

The physical process prior applies at every tick, independently of assignment.
Its state-prior/entropy terms are counted once per active component in the global
objective. They are not multiplied by that component's assignment responsibilities.
Observation terms remain responsibility-weighted. Consequently physical-clock can
change learning and model selection even when soft assignments retain a small
positive mass at every row. Transition diagnostics remain available for structural
proposal screening; assignment-message refinement uses measurement evidence.

Bayesian sequential, batch and natural-gradient updates, the original LDS update
option, and ML paths use the physical trajectory's statistics. Fixed/shared
parameter policies retain ownership. Independent MFA factors have no persistent
state evolution, so they retain their exact conditional-factor implementation.
Static zero-process-noise dynamics also retain their stationary interpretation.

All inference schedules (`online`, `online_fast`, and `offline`), legacy/offline
split-merge strategies, hard/soft assignments, HMM/IID assignments and the existing
left-to-right topology are available subject to their original validation rules.
GP, fixed affine and MFA observations retain their supported noise/policy choices;
DTW and GP-flow input conditioning retain their existing restrictions. This option
does not enable combinations already rejected by the package (for example,
conjugate-batch learning with uncertain GP-flow input warps).

## Diagnostics and practical limits

`model_latent_trajectories` returns every sampling tick in physical-clock mode,
with `observation_indices`, an `observed` mask and the clock name. Its default
cluster-clock arrays keep their existing format. Latent trajectory plots show the
whole smoothed physical path, marking unobserved points separately. They use
future information and are not causal forecasts.

`model.posterior(cluster)` reports the latest propagated global index in physical
mode, including when another cluster supplied the last measurement. Its fields
and shapes are unchanged. In cluster mode it continues to report the last
observation retained by that cluster. `predictive()` remains the current observation
distribution rather than adding another forecast step.

For physical-clock histories, `posterior(cluster, observation_index=t)` and
`predictive(cluster, observation_index=t)` read a saved global tick without
advancing the model. Predictions use the final fitted observation parameters.
Explicit indices are rejected for cluster-clock and outside the saved history.

The shared record experiments (including MIT-BIH) additionally save
`clusters_output_0_early_stop.png` when plotting physical-clock runs, and
`latent_trajectories_early_stop.png` when `--trajectory-plots` is enabled.
Multiple outputs receive separate numbered files. The trajectory starts at
observation 0 and ends at that cluster’s last final MAP assignment, retaining
unobserved gaps before that endpoint. The gallery evaluates each cluster’s
distribution at its own endpoint, rather than at the end of the record. The
original full-record plots are also saved. `--no-plots` suppresses both versions.
These are read-only views of the fitted smoothed posterior, not separate fits or
causal early-stopping estimates. Other observations can still have informed the
final fitted parameters. For direct diagnostic calls, pass `early_stop=True` to
`plot_latent_trajectories` or `plot_clustering_summary`; this option requires
physical-clock dynamics.

Flutter/bearing diagnostics distinguish retained observations from total trajectory
steps and report propagation-only counts. Conversion to growth per second is
still withheld if provided display timestamps are irregular. Spectral radii remain
conditional properties of the fitted representation, not failure certificates.

Physical-clock stores a state/covariance per component and global tick and may
require more smoothing work than cluster-clock. Long gaps with expanding A can
produce large uncertainty. None of these changes guarantee better segmentation;
compare clocks with the same decoder, priors and structural-search settings.
Legacy private checkpoint compaction preserves the physical trajectory rather
than resetting its clock origin, so those checkpoints also retain full histories.

Validation includes closed-form propagation, joint-Gaussian missing-data smoothing
and marginal likelihood, observation/statistic counts, global-prior bookkeeping,
accepted recurring-state splits, cloning/conversion, public inference schedules,
update policies, topology, multi-output, static/ML, shared parameters, input warps,
and the unchanged pinned legacy reference.
