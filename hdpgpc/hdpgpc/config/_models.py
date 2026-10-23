"""Typed scientific configuration. Numerical defaults retain their research meanings."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields
import json
from typing import Literal, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from numpy import ndarray
    from sklearn.gaussian_process.kernels import Kernel

ScalarOption = Union[float, list[float], 'ndarray', None]
BoolOption = Union[bool, list[bool], 'ndarray']
Bounds = Union[tuple[float, float], list[tuple[float, float]], 'ndarray', None]


@dataclass(frozen=True)
class HDPConfig:
    """Concentration preset, initial truncation and dependence switch.

    HMM presets retain joint concentration bundles. With hmm_switch=False,
    gamma is 0.001/0.1/1/100 for less/balanced/more/max respectively.
    balanced_sticky and more_sticky use their base concentrations with kappa=10
    (inactive without HMM); their independent mixture gammas are 0.1 and 1.
    more_extra_sticky uses more concentrations with kappa=100, also inactive
    without HMM, where its mixture gamma remains 1.
    less_extra_sticky uses less concentrations with kappa=100; its independent
    mixture gamma remains 0.001.
    """
    M: int | None = None
    hdp_hyp: Literal['less', 'less_extra_sticky', 'balanced', 'balanced_sticky', 'more', 'more_sticky',
                     'more_extra_sticky', 'max'] = 'balanced'
    hmm_switch: bool = True
    max_models: int | None = None


@dataclass(frozen=True)
class GPEmissionConfig:
    """Kernel/prior values; lists and arrays retain legacy per-cluster semantics."""
    kernels: list[Kernel] | None = None
    fit_gp: bool = True
    ini_lengthscale: ScalarOption = None
    bound_lengthscale: Bounds = None
    ini_sigma: ScalarOption = None
    ini_outputscale: ScalarOption = None
    bound_sigma: Bounds = (1e-10, 1e10)
    inducing_points: BoolOption = False
    check_var: bool = False


@dataclass(frozen=True)
class MatrixPolicyConfig:
    """LDS map prior/ownership. Shrinkage uses a stronger I-centred MNIW prior."""
    mode: Literal['cluster', 'shrinkage', 'shared', 'fixed'] = 'cluster'
    value: ndarray | None = None
    strength: float = 1.0


@dataclass(frozen=True)
class NoisePolicyConfig:
    """LDS covariance posterior/ownership and optional fixed value."""
    mode: Literal['cluster', 'cluster_diagonal', 'shared',
                  'shared_diagonal', 'fixed'] = 'cluster'
    value: ndarray | None = None
    strength: float = 1.0


@dataclass(frozen=True)
class ObservationConfig:
    """Observation map policy. Ordinary fixed matrices use the LDS grid dimension.

    ``cluster`` retains the original per-cluster MNIW update. ``shared`` learns
    one MNIW posterior per output from all cluster responsibilities. ``fixed``
    holds the supplied map and noise covariance, defaulting to identity C and
    the emission's initial observation covariance. A calibrated PCA observation
    uses a rectangular fixed C and a separate waveform grid; Sigma can remain
    fixed or use one of the learned noise policies. A direct fixed affine
    decoder uses the generic `fixed_*` fields; the older `pca_*` fields remain
    readable for existing recipes. ``family='factor_analysis'`` uses the same
    affine grids but learns the intercept, rectangular C and diagonal Sigma.
    """
    scope: Literal['cluster', 'shared', 'fixed'] = 'cluster'
    C: ndarray | None = None
    Sigma: ndarray | None = None
    C_policy: MatrixPolicyConfig | None = None
    Sigma_policy: NoisePolicyConfig | None = None
    # Direct rectangular PCA observation y = pca_mean + C f + epsilon.
    # Set by with_pca_emission after independent minibatch calibration.
    pca_grid: ndarray | None = None
    pca_mean: ndarray | None = None
    pca_variances: ndarray | None = None
    # Generic fixed affine decoder metadata; PCA fields remain readable aliases.
    fixed_grid: ndarray | None = None
    fixed_offset: ndarray | None = None
    fixed_latent_covariance: ndarray | None = None
    # Factor analysis learns the affine decoder rather than freezing it.
    family: Literal['linear', 'factor_analysis'] = 'linear'
    factor_mean_precision: float = 1.0
    factor_loading_precision: float = 1.0


@dataclass(frozen=True)
class DynamicsConfig:
    """Static/LDS controls; omitted MNIW degrees of freedom follow the schedule."""
    model_type: Literal['static', 'dynamic'] | list[Literal['static', 'dynamic']] | ndarray = 'dynamic'
    ini_gamma: ScalarOption = None
    A_policy: MatrixPolicyConfig | None = None
    Gamma_policy: NoisePolicyConfig | None = None
    bound_gamma: Bounds = (1e-1, 1e2)
    annealing: BoolOption = True
    bayesian_params: bool = True
    estimation_limit: int | list[int | None] | ndarray | None = None
    # None selects a schedule-aware prior: 5 offline, 20 online.
    # An integer is always used exactly as supplied.
    free_deg_MNIV: int | None = None
    lds_update: Literal['conjugate_sequential', 'conjugate_batch',
                        'natural_gradient', 'legacy'] = 'conjugate_sequential'
    lds_scoring: Literal['final', 'historical'] = 'final'
    # Fraction of each new batch's sufficient statistics in natural_gradient mode.
    lds_step_size: float = 0.2
    reest_conditions: list[int] = field(default_factory=lambda: [1, 20, 5])


@dataclass(frozen=True)
class AlignmentConfig:
    """Policy selection is separate from each inference call's warp request."""
    policy: Literal['gp', 'gp_flow', 'dtw_gp', 'none'] = 'gp'
    x_basis_warp: ndarray | list[ndarray] | None = None
    noise_warp: float = 0.05
    bound_noise_warp: Bounds = (1e-10, 1e10)
    recursive_warp: BoolOption = False
    warp_updating: BoolOption = False
    method_compute_warp: Literal['standard', 'greedy', 'greedy_bound'] = 'greedy'
    mode_warp: Literal['balanced', 'fine', 'rough'] = 'rough'
    flow_inducing: int = 10
    flow_features: int = 64
    flow_samples: int = 4
    flow_steps: int = 10
    flow_train_iter: int = 30
    flow_lengthscale: float = 0.2
    flow_amplitude: float = 0.1
    flow_lr: float = 0.03
    dtw_points: int = 24
    dtw_inducing: int = 10
    dtw_lengthscale: float = 0.2
    dtw_amplitude: float = 0.005
    dtw_anchor_noise: float = 0.0005


@dataclass(frozen=True)
class InferenceConfig:
    """Construction options and explicit dispatch to distinct existing schedules.

    schedule=None means select at the call site, as legacy experiments do.
    One warp flag controls alignment for either inference schedule.
    offline_strategy selects the batch optimizer independently of LDS updates.
    """
    batch: int | None = None
    reestimate_initial_params: bool = False
    n_explore_steps: int = 10
    offline_representer_selection: Literal['sorted', 'fixed'] = 'sorted'
    schedule: Literal['online', 'online_fast', 'offline'] | None = None
    hard_assignments: bool = True
    warp: bool = False
    force_model: int | None = None
    minibatch: int = 0
    classify: bool = False
    it_limit: int | None = None
    # Divide LDS KL by the latent state dimension in proposal scoring.
    # Raw ELBO reporting remains unscaled.
    birth_lds_kl_per_dimension: bool = True
    offline_strategy: Literal['legacy', 'split_merge'] = 'legacy'
    split_merge_refine_steps: int = 3
    split_merge_stability_patience: int = 3
    split_merge_score_rtol: float = 1e-3

    def run(self, model, x, y):
        """Run one existing entry point; never combine or reorder schedules."""
        from ._validation import validate_inference
        validate_inference(self)
        from ..inference import OnlineInference, OfflineInference
        if self.schedule == 'offline':
            return OfflineInference(model).fit(x, y, self.it_limit, self.warp)
        if self.schedule in ('online', 'online_fast'):
            engine = OnlineInference(model)
            method = engine.update if self.schedule == 'online' else engine.update_fast
            return method(x, y, self.warp, self.force_model, self.minibatch, self.classify)
        raise ValueError('Select an inference schedule before calling run()')


@dataclass(frozen=True)
class MultiOutputConfig:
    """Output count, existing weighting, shared fitting and output reduction."""
    n_outputs: int = 1
    share_gp: bool = False
    use_snr: bool = True
    reduce_outputs: bool = False
    reduce_outputs_ratio: float = 1.0


@dataclass(frozen=True)
class NumericalConfig:
    """Legacy CUDA flag plus opt-in process-wide reproducibility controls.

    None does not reset an existing RNG, thread count, dtype or determinism flag.
    Jitter/solver settings remain internal constants rather than new variation.
    """
    cuda: bool = False
    seed: int | None = None
    num_threads: int | None = None
    dtype: Literal['float64'] | None = None
    deterministic: bool | None = None

    def apply(self):
        from ._validation import validate_numerics
        validate_numerics(self)
        if all(v is None for v in (self.seed, self.num_threads, self.dtype, self.deterministic)):
            return
        import torch
        if self.seed is not None:
            import random
            import numpy as np
            import pyro
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            pyro.set_rng_seed(self.seed)
        if self.num_threads is not None:
            torch.set_num_threads(self.num_threads)
        if self.dtype is not None:
            torch.set_default_dtype(torch.float64)
        if self.deterministic is not None:
            torch.use_deterministic_algorithms(self.deterministic)


SECTIONS = {'hdp': HDPConfig, 'emission': GPEmissionConfig, 'observation': ObservationConfig,
            'dynamics': DynamicsConfig,
            'alignment': AlignmentConfig, 'inference': InferenceConfig,
            'multi_output': MultiOutputConfig, 'numerics': NumericalConfig}
NON_CONSTRUCTOR = {'alignment': {'policy', 'flow_inducing', 'flow_features',
                                 'flow_samples', 'flow_steps', 'flow_train_iter',
                                 'flow_lengthscale', 'flow_amplitude', 'flow_lr',
                                 'dtw_points', 'dtw_inducing',
                                 'dtw_lengthscale', 'dtw_amplitude', 'dtw_anchor_noise'},
                   'inference': {'schedule', 'warp', 'force_model', 'minibatch', 'classify', 'it_limit'},
                   'numerics': {'seed', 'num_threads', 'dtype', 'deterministic'},
                   'observation': {'C_policy', 'Sigma_policy',
                                   'family', 'factor_mean_precision', 'factor_loading_precision',
                                   'pca_grid', 'pca_mean', 'pca_variances',
                                   'fixed_grid', 'fixed_offset', 'fixed_latent_covariance'},
                   'dynamics': {'A_policy', 'Gamma_policy'}}


def resolved_lds_policies(config):
    """Translate the older paired observation scope into independent policies."""
    obs, dynamics = config.observation, config.dynamics
    A = dynamics.A_policy or MatrixPolicyConfig()
    Gamma = dynamics.Gamma_policy or NoisePolicyConfig()
    if obs.C_policy is not None or obs.Sigma_policy is not None:
        C = obs.C_policy or MatrixPolicyConfig()
        Sigma = obs.Sigma_policy or NoisePolicyConfig()
    else:
        C = MatrixPolicyConfig(mode=obs.scope, value=obs.C)
        Sigma = NoisePolicyConfig(mode=obs.scope, value=obs.Sigma)
    return {'A': A, 'Gamma': Gamma, 'C': C, 'Sigma': Sigma}


@dataclass(frozen=True, eq=False)
class HDPGPCConfig:
    """Serializable construction recipe, not a fitted model or checkpoint.

    Compose typed sections for an experiment. build_model creates native state.
    Missing scientific
    initialization values stay None until supplied, and fail clearly at build.
    """
    x_basis: ndarray | list[ndarray] | None = None
    verbose: Literal['silent', 'progress', 'debug'] | bool = False
    hdp: HDPConfig = field(default_factory=HDPConfig)
    emission: GPEmissionConfig = field(default_factory=GPEmissionConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    multi_output: MultiOutputConfig = field(default_factory=MultiOutputConfig)
    numerics: NumericalConfig = field(default_factory=NumericalConfig)

    def __post_init__(self):
        self.validate()

    def __eq__(self, other):
        if not isinstance(other, HDPGPCConfig):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    __hash__ = None

    def validate(self, *, for_build=False):
        from ._validation import validate
        validate(self, for_build=for_build)
        return self




    def build_model(self, *, schedule=None):
        """Construct native state; pass a schedule for an unselected online call."""
        self.validate(for_build=True)
        from ..model._state import ModelState
        self.numerics.apply()
        return ModelState(self, schedule=schedule)

    def to_dict(self):
        from ._codec import encode
        self.validate()
        memo = {}
        return {'schema_version': 1, 'x_basis': encode(self.x_basis, memo), 'verbose': encode(self.verbose, memo),
                **{name: {f.name: encode(getattr(getattr(self, name), f.name), memo) for f in fields(typ)}
                   for name, typ in SECTIONS.items()}}

    @classmethod
    def from_dict(cls, data):
        from ._codec import decode
        if not isinstance(data, dict) or type(data.get('schema_version')) is not int or data['schema_version'] != 1:
            raise ValueError('Expected configuration schema_version=1')
        unknown = set(data) - {'schema_version', 'x_basis', 'verbose'} - set(SECTIONS)
        if unknown:
            raise ValueError('Unknown configuration sections: ' + ', '.join(sorted(unknown)))
        memo = {}
        options = {k: decode(data[k], memo) for k in ('x_basis', 'verbose') if k in data}
        for name, typ in SECTIONS.items():
            section = data.get(name, {})
            if name == 'inference' and isinstance(section, dict):
                old_key = 'birth_lds_kl_per_sqrt_point'
                new_key = 'birth_lds_kl_per_dimension'
                if old_key in section:
                    section = dict(section)
                    old_value = section.pop(old_key)
                    if new_key in section and section[new_key] != old_value:
                        raise ValueError('Conflicting LDS dimension normalization settings')
                    section.setdefault(new_key, old_value)
            if not isinstance(section, dict) or set(section) - {f.name for f in fields(typ)}:
                raise ValueError('Unknown or malformed configuration section: ' + name)
            options[name] = typ(**{f.name: decode(section[f.name], memo) for f in fields(typ) if f.name in section})
        return cls(**options)

    def to_yaml(self):
        """Emit stable JSON syntax, a YAML 1.2 subset, without a YAML dependency."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + '\n'

    @classmethod
    def from_yaml(cls, text):
        """Read emitted YAML; general block YAML requires optional PyYAML."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            try:
                import yaml
            except ImportError as error:
                raise ImportError('Block YAML requires PyYAML; emitted JSON-form YAML needs no extra dependency') from error
            data = yaml.safe_load(text)
        return cls.from_dict(data)


def _initialization_options(config, *, schedule=None):
    """Private expansion into algorithm locals; preserve cross-field aliasing."""
    config.validate(for_build=True)
    if schedule is None:
        schedule = config.inference.schedule or 'offline'
    if schedule not in ('offline', 'online', 'online_fast'):
        raise ValueError('Unknown inference schedule')
    if config.dynamics.lds_update == 'conjugate_batch' and schedule != 'offline':
        raise ValueError('conjugate_batch LDS updates require offline inference')
    if config.inference.offline_strategy == 'split_merge' and schedule != 'offline':
        raise ValueError('split_merge strategy requires offline inference')
    result = {'x_basis': config.x_basis, 'verbose': config.verbose}
    for name in SECTIONS:
        section = getattr(config, name)
        result.update({f.name: getattr(section, f.name) for f in fields(section)
                       if f.name not in NON_CONSTRUCTOR.get(name, ())})
    if result['free_deg_MNIV'] is None:
        result['free_deg_MNIV'] = 5 if schedule == 'offline' else 20
    from ..alignment import GPAlignment, NoAlignment
    if config.alignment.policy == 'none':
        result['alignment'] = NoAlignment()
    elif config.alignment.policy == 'gp_flow':
        from ..alignment._flow_integration import GPFlowAlignment
        result['alignment'] = GPFlowAlignment(config.alignment)
    elif config.alignment.policy == 'dtw_gp':
        from ..alignment._dtw_integration import DTWGPAlignment
        result['alignment'] = DTWGPAlignment(config.alignment)
    else:
        result['alignment'] = GPAlignment()
    return deepcopy(result)
