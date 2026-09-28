# Small runnable tutorials

All examples use deterministic synthetic waveforms; no downloads are required.
Install the package first. The recipes provide explicit small-data parameters,
not changed constructor defaults. Their five-point basis is a pedagogical fixture,
not a recommendation for real recordings.

## Offline dynamic clustering

```python
from hdpgpc import HDPGPC
from hdpgpc.examples import configuration, waveforms

model = HDPGPC(configuration('offline')).fit(waveforms())
print(model.labels_)       # [0 0 1 1 0 1]
print(model.n_clusters_)   # 2 represented clusters
prediction = model.predictive(0)
latent = model.posterior(0)[0]
```

`fit` creates a fresh model and repeats the existing offline local/global schedule.
`predictive` queries the component's current observation mean and native covariance;
it does not forecast the next observation. `posterior` copies the latest stored
filtered/smoothed state without running another smoother.

## Online clustering

```python
from hdpgpc import HDPGPC
from hdpgpc.examples import configuration, waveforms

model = HDPGPC(configuration('online'))
for waveform in waveforms()[:4]:
    result = model.update(waveform)
    print(result.assigned_cluster, result.new_cluster)
print(model.labels_)       # [0 0 0 0] for this dynamic online recipe
```

Each update commits one observation. The first cluster exists before the first
observation; `new_cluster` indicates allocation growth during the update.
`responsibilities` is the retained hard vector and `assignment_probabilities` is
`None`; the implementation does not retain calibrated soft probabilities. Optional
`include_prediction=True, include_posterior=True` request independent snapshots.
An offline fit cannot be continued with online updates.

## Static dynamics

```python
from hdpgpc import HDPGPC
from hdpgpc.examples import configuration, waveforms
model = HDPGPC(configuration('static')).fit(waveforms())
print(model.labels_)
```

This selects the actual supported static formulation, preserving its distinct
initialization and updates. It does not emulate static behavior using a different
Kalman implementation.

## GP temporal alignment

```sh
hdpgpc-reference --mode warping --output warping.json
```

This fits the offline dynamic recipe with alignment enabled. Whether model
alignment executes depends on the existing schedule and assignments. To expose
a nonidentity warp independently, the same command additionally aligns two shifted
24-point Gaussian curves with `hdpgpc.alignment.GaussianProcessWarp`, 50 optimization
steps and the existing objective/defaults. It records offsets, aligned values,
score and actual iteration count. Those extra demonstration settings do not alter
the model's configured alignment iterations. Warping can dominate runtime.
Online warping is not a supported tutorial path.

## Multiple outputs

```python
from hdpgpc import HDPGPC
from hdpgpc.examples import configuration, waveforms
X = waveforms(outputs=2)   # (6 observations, 5 points, 2 outputs)
model = HDPGPC(configuration('multi_output')).fit(X)
print(model.labels_)
states = model.posterior(0)  # one latent snapshot per output
```

The supplied recipe uses the existing offline output weighting. It neither
averages channels before fitting nor gives each channel an independent assignment.
See [configuration](configuration.md) and [API contracts](public_api.md) for
weighting choices, array shapes and supported variations.

Each workflow also has a standalone script in `examples/minimal_*.py` and an
installed command mode (`online`, `offline`, `static`, `warping`, `multi_output`).
For different data, construct a recipe with `x_basis` set to the observation domain
(or use `dataclasses.replace` on the frozen configuration); pass `grid=`
explicitly if observed coordinates differ from the configured basis. Save
`config.to_dict()` or `config.to_yaml()` with each experiment.
