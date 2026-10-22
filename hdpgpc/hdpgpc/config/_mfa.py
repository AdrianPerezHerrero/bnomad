"""Typed recipe for variational DP/HDP-HMM mixtures of factor analyzers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import math
import json
from numbers import Real
from typing import Any, Literal, Mapping

from ._models import HDPConfig, NumericalConfig
from ._validation import boolean, integer, validate_numerics
from ._verbosity import normalize_verbosity


@dataclass(frozen=True, eq=False)
class HDPMFAConfig:
    """Offline DP/HDP-HMM-MFA with optional objective-gated birth/pruning.

    hdp.M is the variational component support size (not the factor rank).
    hdp.max_models, if set, bounds adaptive births. Priors
    broadcast as in FactorAnalysisEmission. Noise is one shared diagonal
    variance vector. NumericalConfig supplies the existing seed/device controls.
    """
    n_factors: int = 2
    hdp: HDPConfig = field(default_factory=lambda: HDPConfig(M=8, hmm_switch=False))
    numerics: NumericalConfig = field(default_factory=NumericalConfig)
    loading_prior_mean: Any = 0.
    loading_prior_precision: Any = 1.
    noise_initial: Any = 1.
    noise_floor: float = 1e-8
    learn_noise: bool = True
    max_iter: int = 200
    tol: float = 1e-6
    chunk_size: int = 1024
    verbose: Literal['silent', 'progress', 'debug'] | bool = 'silent'
    component_strategy: Literal['fixed', 'birth_prune'] = 'fixed'
    n_explore_steps: int = 2
    proposal_refine_steps: int = 3
    prune_threshold: float = 1.

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> HDPMFAConfig:
        if type(self.hdp) is not HDPConfig or type(self.numerics) is not NumericalConfig:
            raise TypeError('hdp and numerics must be HDPConfig and NumericalConfig')
        boolean(self.hdp.hmm_switch, 'hmm_switch')
        integer(self.hdp.M, 'hdp.M', 1)
        integer(self.hdp.max_models, 'hdp.max_models', 1, optional=True)
        if self.hdp.max_models is not None and self.hdp.M > self.hdp.max_models:
            raise ValueError('hdp.M cannot exceed hdp.max_models')
        if self.hdp.hdp_hyp not in ('less', 'balanced', 'balanced_sticky', 'more',
                                     'more_sticky', 'more_extra_sticky', 'max'):
            raise ValueError('Unknown HDP concentration preset')
        for name in ('n_factors', 'max_iter', 'chunk_size', 'n_explore_steps', 'proposal_refine_steps'):
            integer(getattr(self, name), name, 1)
        for name in ('noise_floor', 'tol', 'prune_threshold'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        boolean(self.learn_noise, 'learn_noise')
        if self.component_strategy not in ('fixed', 'birth_prune'):
            raise ValueError("component_strategy must be 'fixed' or 'birth_prune'")
        validate_numerics(self.numerics)
        normalize_verbosity(self.verbose)
        return self

    def to_dict(self) -> dict[str, Any]:
        from ._codec import encode
        values = {item.name: getattr(self, item.name) for item in fields(self)}
        values['hdp'], values['numerics'] = asdict(self.hdp), asdict(self.numerics)
        return {'type': 'hdpmfa_config', 'version': 1, 'values': encode(values)}

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> HDPMFAConfig:
        from ._codec import decode
        if (not isinstance(state, Mapping) or state.get('type') != 'hdpmfa_config'
                or type(state.get('version')) is not int or state.get('version') != 1):
            raise ValueError('Unsupported DP-MFA configuration type or version')
        try:
            if set(state) != {'type', 'version', 'values'}:
                raise ValueError('Unknown configuration fields')
            values = decode(state['values'])
            values['hdp'] = HDPConfig(**values['hdp'])
            values['numerics'] = NumericalConfig(**values['numerics'])
            return cls(**values)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'Invalid HDPMFA configuration: {error}') from error

    def to_yaml(self) -> str:
        """Use the existing JSON-form YAML convention, without extra dependencies."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + '\n'

    @classmethod
    def from_yaml(cls, text: str) -> HDPMFAConfig:
        """Read emitted YAML; block YAML requires the optional PyYAML extra."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            try:
                import yaml
            except ImportError as error:
                raise ImportError('Block YAML requires PyYAML; emitted JSON-form YAML does not') from error
            data = yaml.safe_load(text)
        return cls.from_dict(data)

    def __eq__(self, other):
        if not isinstance(other, HDPMFAConfig):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    __hash__ = None
