# Small configuration examples

These six recipes use a five-point grid and explicit toy-data hyperparameters.
Their numbers are examples, **not new package defaults**. Files use JSON syntax,
which is a YAML 1.2 subset, so no additional parser dependency is needed.

| File | Formulation / schedule |
| --- | --- |
| `static.yaml` | Static emissions, offline updates, no requested warping |
| `dynamic.yaml` | Current LDS dynamics, offline updates |
| `dynamic_warping.yaml` | LDS dynamics and GP warping, offline updates |
| `online.yaml` | Standard online, explicit `with_warp=False` |
| `offline.yaml` | Repeated offline updates, three-iteration limit |
| `multi_output.yaml` | Two outputs, existing SNR weighting, offline updates |

```python
from pathlib import Path
import numpy as np
from hdpgpc.config import HDPGPCConfig

config = HDPGPCConfig.from_yaml(Path('examples/configurations/offline.yaml').read_text())
model = config.build_model()
x = config.x_basis
a = np.array([0., .4, 1., .4, 0.])
b = np.array([0., -.6, -1.2, -.6, 0.])
y = np.stack([a, a + .01, b, b - .01, a - .015, b + .015])[:, :, None]
xs = np.repeat(x[None], len(y), axis=0)
config.inference.run(model, xs, y)
```

For the online example, call `config.inference.run(model, x, sample)` once per
observation. Multi-output data must have shape `(observations, grid_points, 2)`.
To select fast online, use `dataclasses.replace(config.inference, schedule='online_fast')`.
Examples explicitly opt into seed 42, one numerical thread and float64. Production
defaults leave existing process RNG/thread settings untouched.

For real experiments, set the typed sections using the experiment's actual basis
and hyperparameters; do not substitute the toy values. See
[configuration.md](../../docs/configuration.md). Plotting/data paths are not model configuration.
