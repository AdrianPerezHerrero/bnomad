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
    """Existing concentration preset, initial truncation and dependence switch."""
    M: int | None = None
    hdp_hyp: Literal['less', 'balanced', 'more'] = 'balanced'
    hmm_switch: bool = True
    max_models: int | None = None


@dataclass(frozen=True)
class GPEmissionConfig:
    """Kernel/prior values; lists and arrays retain legacy per-cluster semantics."""
    kernels: list[Kernel] | None = None
    ini_lengthscale: ScalarOption = None
    bound_lengthscale: Bounds = None
    ini_sigma: ScalarOption = None
    ini_outputscale: ScalarOption = None
    bound_sigma: Bounds = (1e-10, 1e10)
    inducing_points: BoolOption = False
    check_var: bool = False


@dataclass(frozen=True)
class DynamicsConfig:
    """Supported static/LDS formulation and existing parameter-learning controls."""
    model_type: Literal['static', 'dynamic'] | list[Literal['static', 'dynamic']] | ndarray = 'dynamic'
    ini_gamma: ScalarOption = None
    bound_gamma: Bounds = (1e-1, 1e2)
    annealing: BoolOption = True
    bayesian_params: bool = True
    estimation_limit: int | list[int | None] | ndarray | None = None
    free_deg_MNIV: int = 5
    reest_conditions: list[int] = field(default_factory=lambda: [1, 20, 5])


@dataclass(frozen=True)
class AlignmentConfig:
    """Policy selection is separate from each inference call's warp request."""
    policy: Literal['gp', 'none'] = 'gp'
    x_basis_warp: ndarray | list[ndarray] | None = None
    noise_warp: float = 0.05
    bound_noise_warp: Bounds = (1e-10, 1e10)
    recursive_warp: BoolOption = False
    warp_updating: BoolOption = False
    method_compute_warp: Literal['standard', 'greedy', 'greedy_bound'] = 'greedy'
    mode_warp: Literal['balanced', 'fine', 'rough'] = 'rough'


@dataclass(frozen=True)
class InferenceConfig:
    """Construction options and explicit dispatch to distinct existing schedules.

    schedule=None means select at the call site, as legacy experiments do.
    Online and offline warp defaults deliberately differ.
    """
    batch: int | None = None
    reestimate_initial_params: bool = False
    n_explore_steps: int = 10
    schedule: Literal['online', 'online_fast', 'offline'] | None = None
    with_warp: bool = True
    warp: bool = False
    force_model: int | None = None
    minibatch: int = 0
    classify: bool = False
    it_limit: int | None = None

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
            return method(x, y, self.with_warp, self.force_model, self.minibatch, self.classify)
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


SECTIONS = {'hdp': HDPConfig, 'emission': GPEmissionConfig, 'dynamics': DynamicsConfig,
            'alignment': AlignmentConfig, 'inference': InferenceConfig,
            'multi_output': MultiOutputConfig, 'numerics': NumericalConfig}
NON_CONSTRUCTOR = {'alignment': {'policy'},
                   'inference': {'schedule', 'with_warp', 'warp', 'force_model', 'minibatch', 'classify', 'it_limit'},
                   'numerics': {'seed', 'num_threads', 'dtype', 'deterministic'}}


@dataclass(frozen=True, eq=False)
class HDPGPCConfig:
    """Serializable construction recipe, not a fitted model or checkpoint.

    Compose typed sections for an experiment. build_model creates native state.
    Missing scientific
    initialization values stay None until supplied, and fail clearly at build.
    """
    x_basis: ndarray | list[ndarray] | None = None
    verbose: bool = False
    hdp: HDPConfig = field(default_factory=HDPConfig)
    emission: GPEmissionConfig = field(default_factory=GPEmissionConfig)
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




    def build_model(self):
        """Construct private native state for advanced engine use."""
        self.validate(for_build=True)
        from ..model._state import ModelState
        self.numerics.apply()
        return ModelState(self)

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


def _initialization_options(config):
    """Private expansion into algorithm locals; preserve cross-field aliasing."""
    config.validate(for_build=True)
    result = {'x_basis': config.x_basis, 'verbose': config.verbose}
    for name in SECTIONS:
        section = getattr(config, name)
        result.update({f.name: getattr(section, f.name) for f in fields(section)
                       if f.name not in NON_CONSTRUCTOR.get(name, ())})
    from ..alignment import GPAlignment, NoAlignment
    result['alignment'] = NoAlignment() if config.alignment.policy == 'none' else GPAlignment()
    return deepcopy(result)
