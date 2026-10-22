# Migrating to 0.3

0.3 is a breaking pre-1 release. The legacy implementation and compatibility API
are removed. Keep a 0.2 environment to read old fitted-model pickles; recreate
experiments from data and typed configuration in 0.3.

| Removed interface | Supported replacement |
| --- | --- |
| `GPI_HDP(x_basis, **options)` | `HDPGPC(HDPGPCConfig(...))` |
| whole-model `include_sample(x_train, y, ...)` | `model.update(y, grid=x_train)` with an online recipe |
| whole-model `include_batch(x_trains, y_trains, ...)` | `model.fit(y_trains, grid=x_trains)` with an offline recipe |
| `GPI_model` | `hdpgpc.emissions.GaussianProcessEmission` |
| active `amtgp_warping_system.Warping_system` | `hdpgpc.alignment.GaussianProcessWarp` |
| `from_legacy_args`, `from_legacy_kwargs`, `to_legacy_kwargs` | compose typed configuration sections |
| classification and label reload | `OfflineInference.classify_batch` / `OfflineInference.fit_labels` |

The old first argument was a coordinate grid. Public `X` and `x` now denote
waveform values; `grid` supplies coordinates. `fit` starts fresh; repeated `update`
calls extend an online model. Use the original experiment's parameter values,
not the toy tutorial hyperparameters.

```python
from hdpgpc import HDPGPC
from hdpgpc.examples import configuration, waveforms
model = HDPGPC(configuration('offline')).fit(waveforms())
print(model.labels_)
```

Advanced workflows use typed configuration and explicit engines:

```python
import numpy as np
from hdpgpc.examples import configuration, waveforms
from hdpgpc.inference import OfflineInference
config = configuration('offline')
state = config.build_model()
engine = OfflineInference(state)
y = waveforms()
x = np.repeat(config.x_basis[None], len(y), axis=0)
engine.fit(x, y, it_limit=3, warp=False)
labels = engine.classify_batch(x, y, learning=False)
```

`fit_labels(x, y, labels, M, warp=False)` retains label-conditioned fitting.
`classify_batch(..., learning=True)` retains the original learning branch. Advanced
engines preserve original tensor inputs/returns, mutation behavior and limitations;
they do not promise the copied-result contract of `HDPGPC`. Treat their state as
opaque; its class and mutable fields are private.

Saved configuration dictionaries remain schema version 1. Change
`alignment.policy: legacy` to `alignment.policy: gp` explicitly when importing
an older recipe. The scientific default is still the same GP alignment. `none`
selects identity. Removed policies/constructors raise errors; no fallback exists.

Inference recipes now have one `inference.warp` flag for both schedules, with
`False` as the default. To migrate a saved recipe containing both old flags,
keep its offline `warp` value for an offline schedule or use the old online flag's
value as `warp` for an online schedule, then remove the extra field. The loader
rejects the retired field so that a conflicting pair cannot be interpreted
silently. Call `OnlineInference.update(..., warp=True)` or set
`InferenceConfig(schedule='online', warp=True)` to enable online alignment.

The inactive alternate warper, side-effect scripts, historical study notebooks
and old pickle paths are retired. Maintained examples and eight tutorials replace
the old programs; originals remain in the separate unchanged 0.2 export.
Dataset/plot utilities are private in `datasets._research` and `diagnostics._plots`.
See [architecture](architecture.md) and [validation](migration_0_3/report.md).
