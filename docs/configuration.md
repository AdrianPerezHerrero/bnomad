# Typed scientific configuration

Compose `HDPGPCConfig` from typed sections, or load a supplied recipe. Recipes are
data, not fitted models. Flat constructor compatibility adapters are removed.

```python
from dataclasses import replace
from hdpgpc import HDPGPC, HDPGPCConfig
from hdpgpc.examples import configuration, waveforms
config = configuration('dynamic')
config = replace(config, inference=replace(config.inference, it_limit=3))
restored = HDPGPCConfig.from_yaml(config.to_yaml())
model = HDPGPC(restored).fit(waveforms())
```

| Section | Type | Controls |
| --- | --- | --- |
| `hdp` | `HDPConfig` | `M`, concentration preset, HMM dependence, maximum models |
| `emission` | `GPEmissionConfig` | kernels, lengthscale/output/noise initialization and bounds, inducing points |
| `dynamics` | `DynamicsConfig` | static/dynamic mode, process noise, ML/Bayesian updates, annealing, estimation limits |
| `alignment` | `AlignmentConfig` | `gp`/`dtw_gp`/`gp_flow`/`none`, basis, noise/bounds, recursion, warp mode |
| `inference` | `InferenceConfig` | online/fast/offline schedule and separate call controls |
| `multi_output` | `MultiOutputConfig` | output count, GP fitting sharing, SNR, output reduction |
| `numerics` | `NumericalConfig` | device and opt-in seed/thread/dtype/determinism |

Root fields are `x_basis` and `verbose`. Scientific defaults retain the characterized
values: `M=None`, dynamic mode, Bayesian updates, annealing, SNR and balanced HDP.
Bare `HDPGPCConfig()` is an incomplete serializable recipe. Construction requires
a basis, initial observation/process noise and lengthscale initialization/bounds
for automatic kernels. Static construction also still requires process noise;
this preserved assumption is not silently corrected.

Alignment policy defaults to `gp`, replacing the serialized token `legacy` with
the same scientific choice. Change that token explicitly in older recipes.
`none` disables alignment even if requested by a call. `InferenceConfig.warp`
defaults to `False` for both online and offline schedules; set it to `True` to
request alignment with either schedule.
`gp_flow` enables an opt-in variational monotonic GP-flow warp posterior; see
[GP-flow alignment](gp_flow_alignment.md) for its objective, controls and limits.
`dtw_gp` estimates a cheaper monotone point warp with unrestricted DTW and a smooth
GP displacement prior; see [DTW–GP alignment](dtw_gp_alignment.md).

For either online schedule, `verbose=True` prints the baseline objective and
the birth and absorb proposals for each subsequent sample. Each proposal shows
its emission, latent, HDP, and LDS terms, total ELBO, change from the baseline,
and the signed warp objective contribution when available. The emission term
already includes that negative contribution. Candidate cluster numbers refer to the order
before the selected state is reordered. If the birth emission gate fails, the
proposal comparisons are diagnostic and do not affect the assignment.

`schedule=None` selects offline for public `fit` and standard online for `update`.
When `DynamicsConfig.free_deg_MNIV` is omitted (`None`), model construction
uses `5` for offline inference and `20` for `online` or `online_fast`. An
explicit integer, including `5`, overrides this choice in either schedule.
Existing configuration files that contain `free_deg_MNIV: 5` therefore keep
the prior strength they request.
The one-observation MNIW mean update weights the new estimate by
`1/(free_deg_MNIV - 1)`: `1/4` with `5`, versus `1/19` with `20`.
The prior also slows process and observation covariance updates. This applies
to Bayesian LDS parameter learning; latent-state filtering can still adapt
to each observation. For advanced direct construction with no configured
schedule, pass `config.build_model(schedule='online')` to select the online
default; otherwise direct construction assumes offline.

Explicit choices are `online`, `online_fast`, `offline`. `InferenceConfig.run`
dispatches one observation or batch to the native engine. `config.build_model()`
constructs opaque native state for advanced engine workflows; normal usage should
use `HDPGPC(config)`.

The codec supports dictionaries, JSON-form YAML, optional block YAML (PyYAML),
tagged array/tensor shapes and dtypes, supported composite sklearn kernels,
tuples, infinities and repeated references to supported mutable parameter objects.
It never loads pickle or arbitrary classes. Schema version remains 1. Recipes
are not hashable. Optimizer histories, fitted caches and arbitrary custom
components are not serialized configuration data.

Validation rejects unknown modes/sections, invalid counts, incompatible schedule
controls, per-cluster length mismatches, reversed bounds, nonfinite grids,
zero-output reductions, unavailable requested CUDA and missing initialization.
It does not clip parameters, symmetrize covariances, change jitter or repair an
algorithm. Known online warp, multi-output and forced-reuse failures remain
characterized even when a recipe is structurally valid.

Numerical overrides default to `None`. Native model import precedes explicit
overrides so it cannot overwrite a requested seed. Explicit seed controls Python,
NumPy, PyTorch and Pyro; it does not clear Pyro's parameter store. Only characterized
float64 is selectable. Threads and deterministic algorithms are process-wide.

The six YAML examples cover static, dynamic, dynamic plus warping, online, offline
and multi-output. Their tiny-data values are examples, not new defaults. Tests
round-trip recipes and compare serialized construction against all frozen
inference scenarios without changing the expected numbers.
