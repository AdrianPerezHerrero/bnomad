# Native architecture in 0.3

Only one implementation path exists. `HDPGPC` validates inputs and copies results;
a typed recipe initializes native `ModelState`; explicit engines execute the
scientific schedules. No class inherits from, wraps, instantiates or dynamically
imports a retired implementation class.

| Area | Actual responsibility | Public surface |
| --- | --- | --- |
| `model` | Input/result adaptation, initialization, mutable state, cluster lifecycle, representation conversion | `HDPGPC`, `UpdateResult`, `Prediction`, `LatentState` |
| `config` | Typed options, validation and data-only dictionary/YAML codec | `HDPGPCConfig`, eight section dataclasses |
| `inference` | Separate standard/fast online, offline, classification and label-conditioned schedules; HMM messages and objective aggregation | `InferenceEngine`, `OnlineInference`, `OfflineInference` |
| `hdp` | Sticks, transitions, concentration presets, expected logs, entropy, attributed bnpy optimizer | `HDPVariational` |
| `emissions` | GP/output trajectories, priors, prediction, fitting, GPyTorch models and SNR weights | `EmissionModel`, `GaussianProcessEmission` |
| `dynamics` | Static/LDS initialization, filtering, smoothing, learning and conjugate distributions | `DynamicsModel`, `StaticDynamics`, `LinearDynamics` |
| `alignment` | Identity/AMTGP policies, grids, optimization, prior and caches | `NoAlignment`, `GPAlignment`, `GaussianProcessWarp` |
| `numerics` | Shared low-level normalization | private |
| `datasets` | Research dataset utilities | dataset-specific |
| `preprocessing` | Generic waveform selection, optional scaling, common-grid preparation | `prepare_sequences` |
| `diagnostics` | Label and prior summaries; LDS and warp inspection plots | reporting and plotting helpers |
| `examples` | Small deterministic recipes, inputs and reference experiment | `configuration`, `waveforms`, reference command |

## Dependency direction

```text
HDPGPC -> typed configuration -> ModelState initialization
       -> OnlineInference.update / update_fast
       -> OfflineInference.fit / classify_batch / fit_labels
engines -> cluster lifecycle, HMM messages, objective scoring, output weights
        -> HDP variational component
        -> GP emission -> GP prior/prediction and training
                       -> static/LDS dynamics
        -> alignment policy -> native AMTGP or identity
scientific components -> numerical helpers and numerical libraries
```

Engines share state, not an inference loop. HDP does not orchestrate emissions,
dynamics or alignment. Dynamics does not import the main model or emission class.
Alignment policy owns enable/disable decisions; the warper owns its optimizer/cache.

`model._state.ModelState` has typed initialization and private component dispatch.
Algorithm operations are direct bindings to functions defined in their owners,
with no second copy of their bodies:

- `model._clusters`: defaults, component construction/copying, birth, pruning and reordering.
- `inference._messages`: local responsibilities and HMM recurrences.
- `inference._scoring`: emission/LDS/HDP objective aggregation and candidate scores.
- `emissions._weights`: SNR, output reduction and joint scores.
- `model._conversion`: array/tensor/device conversion and private checkpoint writing.
- `inference._online`, `_offline`: distinct schedules and private helpers.

Direct bindings preserve self identity, array aliasing, mutation order and trace
boundaries. Private HDP/alignment dispatch on the state is used by current engines;
it is not an exported compatibility API and contains no HDP/warp mathematics.
Native state has no old constructor, `include_sample`, `include_batch` or `from_config`.

`GaussianProcessEmission` owns trajectory/cache state. `emissions._prior` owns
covariance and prediction operations; `_training` owns fitting objectives,
optimizers and stopping loops; `_torch_models` owns the three active GPyTorch
training classes. Unused alternate warp models were retired, not duplicated.

Root `__all__` contains exactly `HDPGPC`, `HDPGPCConfig`, `UpdateResult`, `Prediction`
and `LatentState`. Underscore modules and mutable fields are private. Namespace
imports remain lazy. Scientific-state initialization preserves import-time
float64/seed settings, followed by explicit numerical configuration. There is no
generic facade forwarding of private state.

The flat scientific state and its aliases intentionally remain. This migration
does not redesign optimizer policies, caches, covariance formulas, precision or
convergence rules. Historical 0.2 design documents are labeled as evidence;
their old paths are not current import advice.
