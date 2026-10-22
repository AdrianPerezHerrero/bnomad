# Native architecture in 0.3

Only one implementation path exists. `HDPGPC` validates inputs and copies results;
a typed recipe initializes native `ModelState`; explicit engines execute the
scientific schedules. No class inherits from, wraps, instantiates or dynamically
imports a retired implementation class.

| Area | Actual responsibility | Public surface |
| --- | --- | --- |
| `model` | Input/result adaptation, initialization, mutable state, cluster lifecycle, representation conversion | `HDPGPC`, `HDPMFA`, `UpdateResult`, `Prediction`, `LatentState` |
| `config` | Typed options, validation and data-only dictionary/YAML codec | `HDPGPCConfig`, `HDPMFAConfig`, shared section dataclasses |
| `inference` | Separate standard/fast online, selectable legacy/splitâ€“merge offline, classification and label-conditioned schedules; HMM messages, DP/HDP-HMM-MFA CAVI and objective aggregation | `InferenceEngine`, `OnlineInference`, `OfflineInference`, `MFAOfflineInference` |
| `hdp` | Sticks, transitions, concentration presets, expected logs, entropy, attributed bnpy optimizer | `HDPVariational` |
| `emissions` | GP/output trajectories, standalone Bayesian FA emission, observation policies and histories, fixed interfaces, affine decoders, PCA/linear-AE calibration, Fourier/wavelet bases, prediction, fitting and SNR weights | `EmissionModel`, `GaussianProcessEmission`, `FactorAnalysisEmission`, `FixedObservationModel`, `AffineGaussianObservation`, calibration/basis helpers |
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
`dynamics/_independent.py` owns compatible fixed-zero-transition batch
trajectory operations; `AffineGaussianObservation` owns batched conditioning
and likelihoods. Its private pass-local conditioner reuses observation factors
for batch LDS filtering; final affine scores also batch nonzero-A trajectories.
`inference/_messages.py` dispatches independent assignments
directly. [Fast-path conditions](independent_fast_paths.md) document fallbacks.
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

`HDPMFA` and `HDPMFAConfig` are additional explicit lazy root imports; the
established root star-import list above is preserved. DP-MFA owns its state in
`model._mfa_state`, its schedule/objective in `inference._mfa`, and its typed
recipe in `config._mfa`. It reuses `HDPVariational` and the verbatim independent
assignment core extracted from `_messages`. It supplies no dummy GP settings
or LDS trajectories. IID assignments retain their original mathematics.
HMM assignments reuse the native recurrences in `inference._hmm_messages`,
through boundary-aware `inference._mfa_assignments`. `hdp._markov` adapts the
native HDP bound/optimizer to K active source rows and a tail destination;
it does not change legacy GP scoring. `model._mfa_lifecycle` isolates and
objective-gates optional births/pruning, using generic marginal-routing and
partition keys in `inference._partitions` (the GP split/merge engine shares
that same key helper). See [DP-MFA](dp_mfa.md) and [HDP-HMM-MFA](hmm_mfa.md).

Typed MFA result snapshots live in `model._mfa_results`; data-only schema checks
and atomic file replacement live in `model._mfa_serialization`. The facade's
public methods are annotated and exported for type checkers without eager
namespace imports. `experiments.mfa` owns synthetic comparisons and provenance;
its independent finite-MFA EM comparator is in `experiments._mfa_reference`,
outside the Bayesian inference and emission implementations. See
[MFA validation](mfa_validation.md).

`emissions._factor_analysis` implements independent local factors and learnable
Gaussian centroid/loading rows. It has no HDP, GP, LDS or model-state dependency.
Its small posterior/statistics/moment result classes and emission are exposed
lazily from `emissions`. The existing GP-specific `EmissionModel` protocol is
unchanged; DP-MFA directly consumes the FA component's named operations. See
[the standalone FA component](factor_analysis_emission.md) for milestone scope.

The flat scientific state and its aliases intentionally remain. This migration
does not redesign optimizer policies, caches, covariance formulas, precision or
convergence rules. Historical 0.2 design documents are labeled as evidence;
their old paths are not current import advice.
