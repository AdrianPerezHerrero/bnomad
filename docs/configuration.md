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
`InferenceConfig.hard_assignments` defaults to `True`. Set it to `False` to
retain normalized state and transition responsibilities in online or offline
inference. GP filtering and Bayesian LDS sufficient statistics then use each
observation's fractional responsibility. Cluster creation and reported labels
still use a MAP choice. Soft assignments require `bayesian_params=True` and
can cost substantially more because an observation may update several GPs.
Offline emission fitting drops only fractional weights whose combined mass is
at most `1e-8` of that cluster's responsibility total. A cluster with less
than `1e-8` total mass is skipped during that fitting pass. This avoids
numerical updates from negligible softmax tails.
The offline HDP linear term counts transitions between beats once; it is not
multiplied by the waveform point count. Birth and baseline scores each use a
full transition-parameter update from their own expected counts.
`gp_flow` enables an opt-in variational monotonic GP-flow warp posterior; see
[GP-flow alignment](gp_flow_alignment.md) for its objective, controls and limits.
`dtw_gp` estimates a cheaper monotone point warp with unrestricted DTW and a smooth
GP displacement prior; see [DTW–GP alignment](dtw_gp_alignment.md).

For either online schedule, `verbose=True` prints the baseline objective and
the birth and absorb proposals for each subsequent sample. Each proposal shows
its emission, latent, HDP, and LDS terms, total ELBO, change from the baseline,
and the signed warp objective contribution when available. The emission term
already includes that negative contribution. Candidate cluster numbers refer to the order
before the selected state is reordered. With hard assignments, if the birth
emission gate fails, the proposal comparisons are diagnostic and do not affect
the assignment.

`schedule=None` selects offline for public `fit` and standard online for `update`.
When `DynamicsConfig.free_deg_MNIV` is omitted (`None`), model construction
uses `5` for offline inference and `20` for `online` or `online_fast`. An
explicit integer, including `5`, overrides this choice in either schedule.
Existing configuration files that contain `free_deg_MNIV: 5` therefore keep
the prior strength they request.
`free_deg_MNIV` controls the inverse-Wishart prior strength and does not by
itself set a fixed learning rate for the MNIW matrix means.
For optional prior-scale calibration before construction,
`hdpgpc.dynamics.estimate_lds_noise_priors(observations)` estimates scalar
`ini_sigma` and `ini_gamma` from random contiguous observation blocks. Its
outputs are covariance diagonal variances, not standard deviations. Its
identity random-walk pilot assumes `A=C=I` and independent temporal process
and observation noise. It uses the variance and lag-one covariance of beat
differences, and raises an error if these moments cannot identify positive
noise scales. Pass `estimate.observation` to `GPEmissionConfig.ini_sigma` and
`estimate.process` to `DynamicsConfig.ini_gamma`; the function never changes
an existing model or its prior during inference. The full-record example
provides `--calibrate-lds-noise-priors` for offline experiments. The estimate
can be biased by regime changes, nonidentity `A`/`C`, or atypical blocks, and
does not guarantee a smaller LDS KL.
`DynamicsConfig.lds_update` selects `conjugate_sequential` (the package default),
`conjugate_batch`, `natural_gradient`, or `legacy`. The former `conjugate` name
is now `conjugate_sequential`. The sequential policy updates the MNIW posterior
after each assigned observation using the current smoothed state covariance
and adjacent-state cross-covariance as well as the state means. Previously
those covariance statistics were discarded. `conjugate_batch` is available
only offline:
it filters and smooths the whole cluster, sums responsibility-weighted state
and observation moments, and updates the LDS posterior once from its original
prior per cluster fit. The next offline refit uses the last LDS estimate to
filter but starts the MNIW posterior from the original prior again, so it does
not count the same record twice. The batch posterior can differ from the
sequential result because latent states are inferred under different evolving
parameters. `conjugate_batch` requires Bayesian LDS parameters.
It requires observations on the LDS basis grid; GP-flow input warps are not
supported by this conjugate update. Output-aligned waveform warps retain the
common grid and can use it.

`DynamicsConfig.lds_scoring` independently selects `final` (the default) or
`historical`. In `final` mode, the observation ELBO uses the cluster's last
`C` and `Sigma` for every beat, and the latent transition density uses its
last `A` and `Gamma`. In `historical` mode, each score uses the LDS parameter
history associated with that beat, subject to the existing estimation limit.
Both modes score the saved beat-specific smoothed state mean and covariance;
the chain entropy in both modes uses the historical transitions that produced
that smoothed path. This setting does not change filtering, parameter updates,
or the LDS posterior KL. Historical scoring is still an in-sample score, not a
pre-observation predictive score. The full-record example exposes the choice
through `LDS_SCORING` and `--lds-scoring`.

The legacy policy retains the earlier nonconjugate mean and covariance rules
for comparison. The natural-gradient
policy uses `lds_step_size` in `(0, 1]` (default `0.2`) to temper each new
observation's MNIW sufficient statistics and degrees-of-freedom increment:
`eta_new = eta_old + lds_step_size * (eta_conjugate - eta_old)` in natural
parameters. At step size `1`, it equals the sequential conjugate update; below `1`, it
adapts more slowly and is a tempered posterior rather than the exact posterior
for all observations. Static inverse-Wishart observation noise uses the same
tempering; its legacy policy is the existing static update. LDS parameter
damping does not stop the latent Kalman filter from responding to each sample.
For advanced direct construction with no configured schedule, pass
`config.build_model(schedule='online')` to select the online default;
otherwise direct construction assumes offline.

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
