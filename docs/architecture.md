# Native architecture in 0.3

Only one implementation path exists. `HDPGPC` validates inputs and copies results;
a typed recipe initializes native `ModelState`; explicit engines execute the
scientific schedules. No class inherits from, wraps, instantiates or dynamically
imports a retired implementation class.

| Area | Actual responsibility | Public surface |
| --- | --- | --- |
| `model` | Input/result adaptation, initialization, mutable state, cluster lifecycle, representation conversion | `HDPGPC`, `UpdateResult`, `Prediction`, `LatentState` |
| `config` | Typed options, validation and data-only dictionary/YAML codec | `HDPGPCConfig`, eight section dataclasses |
| `inference` | Separate standard/fast online, selectable legacy/split–merge offline, classification and label-conditioned schedules; HMM messages and objective aggregation | `InferenceEngine`, `OnlineInference`, `OfflineInference` |
| `hdp` | Sticks, transitions, concentration presets, expected logs, entropy, attributed bnpy optimizer | `HDPVariational` |
| `emissions` | GP/output trajectories, observation policies and histories, fixed interfaces, affine decoders, PCA/linear-AE calibration, Fourier/wavelet bases, prediction, fitting and SNR weights | `EmissionModel`, `GaussianProcessEmission`, `FixedObservationModel`, `AffineGaussianObservation`, calibration/basis helpers |
| `dynamics` | Static/LDS initialization, filtering, smoothing, learning and conjugate distributions | `DynamicsModel`, `StaticDynamics`, `LinearDynamics` |
| `alignment` | Identity/AMTGP policies, grids, optimization, prior and caches | `NoAlignment`, `GPAlignment`, `GaussianProcessWarp` |
| `numerics` | Shared low-level normalization | private |
| `datasets` | Research dataset utilities | dataset-specific |
| `preprocessing` | Generic waveform selection, optional scaling, common-grid preparation | `prepare_sequences` |
| `diagnostics` | Label and prior summaries; LDS and warp inspection plots | reporting and plotting helpers |
| `experiments` | Common CLI, typed recipe construction, dataset execution, run identity and artifact output | `ExperimentRun`, `save_results`, record/UCR workflows |
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
                       -> fixed affine observation decoder
        -> alignment policy -> native AMTGP or identity
scientific components -> numerical helpers and numerical libraries
```

Engines share state, not an inference loop. HDP does not orchestrate emissions,
dynamics or alignment. Dynamics algorithms do not import the main model or GP emission class.
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
- `inference._split_merge`: opt-in transactional group split/merge search,
  reusing native messages, emission fitting, shared policies and objective terms.

Direct bindings preserve self identity, array aliasing, mutation order and trace
boundaries. Private HDP/alignment dispatch on the state is used by current engines;
it is not an exported compatibility API and contains no HDP/warp mathematics.
Native state has no old constructor, `include_sample`, `include_batch` or `from_config`.

`GaussianProcessEmission` owns trajectory/cache state. `emissions._prior` owns
covariance and prediction operations; `_training` owns fitting objectives,
optimizers and stopping loops; `_torch_models` owns the three active GPyTorch
training classes. Unused alternate warp models were retired, not duplicated.

`emissions._fixed_observation` owns the fixed observation contract and exact
affine Gaussian decoder. `_pca` calibrates a decoder and its priors; `_fourier`
and `_wavelet` construct fixed synthesis bases. These describe the mapping from
latent states to observations. Kalman recurrences, smoothing and conjugate
parameter calculations stay in `dynamics`. The old observation modules and
exports under `dynamics` have been removed. Package initialization, examples
and diagnostic scripts import observation objects from `emissions`.

`emissions._observation` owns shared C/Sigma policies and their observation
histories; `dynamics._transition` owns A/Gamma policies. Both reuse the joint
conjugate-distribution calculations in `dynamics`. These distributions stay
there because they describe LDS parameter posteriors, not a particular encoder.
`emissions._autoencoder` owns the reusable tied linear autoencoder and its
calibration result. Dataset loaders, including the canonical UCR ZIP reader,
belong to `datasets`; optional WFDB is imported only when reading WFDB records.

The eight dataset experiment scripts are matching entry points. Generic stages
live in `experiments.records` and `experiments.ucr`, using the same public model
API and common reporting. Neither inference nor observation modules import
these workflows or example files. The annotated cluster gallery now lives in
`diagnostics._cluster_plots`; `tutorial_utils.plot_record_results` remains an
alias for existing notebooks. See [experiment workflows](experiment_workflows.md).

Root `__all__` contains exactly `HDPGPC`, `HDPGPCConfig`, `UpdateResult`, `Prediction`
and `LatentState`. Underscore modules and mutable fields are private. Namespace
imports remain lazy. Scientific-state initialization preserves import-time
float64/seed settings, followed by explicit numerical configuration. There is no
generic facade forwarding of private state.

The flat scientific state and its aliases intentionally remain. This migration
does not redesign optimizer policies, caches, covariance formulas, precision or
convergence rules. Historical 0.2 design documents are labeled as evidence;
their old paths are not current import advice.
