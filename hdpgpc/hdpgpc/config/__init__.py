"""Typed scientific configuration and stable data-only serialization."""

from ._models import (HDPGPCConfig, HDPConfig, GPEmissionConfig, ObservationConfig, DynamicsConfig,
                      MatrixPolicyConfig, NoisePolicyConfig,
                      AlignmentConfig, InferenceConfig, MultiOutputConfig,
                      NumericalConfig)

__all__ = ('HDPGPCConfig', 'HDPConfig', 'GPEmissionConfig', 'ObservationConfig', 'DynamicsConfig',
           'MatrixPolicyConfig', 'NoisePolicyConfig',
           'LDS_PRESETS', 'with_lds_preset', 'with_pca_emission', 'with_fixed_observation_model', 'with_mfa_emission',
           'AlignmentConfig', 'InferenceConfig', 'MultiOutputConfig', 'NumericalConfig')


def __getattr__(name):
    if name in ('LDS_PRESETS', 'with_lds_preset', 'with_pca_emission', 'with_fixed_observation_model', 'with_mfa_emission'):
        from . import _presets
        return getattr(_presets, name)
    raise AttributeError(name)
