"""Named LDS policy bundles built from the existing typed configuration fields."""

from dataclasses import replace
import math
from numbers import Real

import numpy as np

from ._models import (HDPGPCConfig, MatrixPolicyConfig, NoisePolicyConfig,
                      resolved_lds_policies)


LDS_PRESETS = (
    'all_free', 'plastic', 'stable', 'only_plastic', 'only_stable',
    'shared_emission', 'shared_latent_process', 'all_shrunk',
    'all_fixed', 'random_walk', 'shared_all', 'diagonal_noise',
    'static_mixture',
)


_MODES = {
    # A, Gamma, C, Sigma. Noise shrinkage changes prior concentration,
    # not its covariance scale.
    'all_free': ('cluster', 'cluster', 'cluster', 'cluster'),
    'plastic': ('cluster', 'cluster', 'shrinkage', 'cluster'),
    'stable': ('shrinkage', 'cluster', 'cluster', 'cluster'),
    'only_plastic': ('cluster', 'cluster', 'fixed', 'fixed'),
    'only_stable': ('fixed', 'fixed', 'cluster', 'cluster'),
    'shared_emission': ('cluster', 'cluster', 'shared', 'shared'),
    'shared_latent_process': ('shared', 'shared', 'cluster', 'cluster'),
    'all_shrunk': ('shrinkage', 'cluster', 'shrinkage', 'cluster'),
    'all_fixed': ('fixed', 'fixed', 'fixed', 'fixed'),
    'random_walk': ('fixed', 'fixed', 'fixed', 'fixed'),
    'shared_all': ('shared', 'shared', 'shared', 'shared'),
    'diagonal_noise': ('cluster', 'cluster_diagonal',
                       'cluster', 'cluster_diagonal'),
}


def with_lds_preset(config: HDPGPCConfig, preset: str, *,
                    shrink_strength: float = 10.0) -> HDPGPCConfig:
    """Return a new recipe using one named bundle of LDS parameter policies.

    Existing custom matrix/covariance values are retained when compatible.
    Shrinkage always centers matrix priors on I; covariance shrinkage centers
    on the configured initial noise scale (or its custom prior value).
    ``static_mixture`` also selects static dynamics and independent cluster
    assignments via ``hmm_switch=False``.
    """
    if not isinstance(config, HDPGPCConfig):
        raise TypeError('config must be HDPGPCConfig')
    if config.observation.pca_grid is not None:
        raise ValueError('PCA fixes C and selects free A/Gamma; LDS presets do not apply')
    if preset not in LDS_PRESETS:
        raise ValueError('Unknown LDS preset: ' + str(preset))
    if (isinstance(shrink_strength, bool) or not isinstance(shrink_strength, Real)
            or not math.isfinite(shrink_strength) or shrink_strength <= 1):
        raise ValueError('shrink_strength must be finite and > 1')

    old = resolved_lds_policies(config)
    if preset == 'static_mixture':
        C = MatrixPolicyConfig(mode='fixed', value=old['C'].value)
        Sigma = NoisePolicyConfig(mode='cluster', value=old['Sigma'].value)
        observation = replace(config.observation, scope='cluster', C=None,
                              Sigma=None, C_policy=C, Sigma_policy=Sigma)
        dynamics = replace(config.dynamics, model_type='static',
                           A_policy=None, Gamma_policy=None)
        hdp = replace(config.hdp, hmm_switch=False)
        return replace(config, observation=observation, dynamics=dynamics, hdp=hdp)

    A_mode, Gamma_mode, C_mode, Sigma_mode = _MODES[preset]

    def matrix(name, mode):
        value = old[name].value
        if mode == 'shrinkage' or (preset == 'random_walk' and name in ('A', 'C')):
            value = None
        return MatrixPolicyConfig(mode=mode, value=value,
                                  strength=shrink_strength if mode == 'shrinkage' else 1.0)

    def noise(name, mode):
        strength = (shrink_strength if (preset in ('stable', 'all_shrunk') and name == 'Gamma')
                    or (preset in ('plastic', 'all_shrunk') and name == 'Sigma') else 1.0)
        return NoisePolicyConfig(mode=mode, value=old[name].value, strength=strength)

    observation = replace(config.observation, scope='cluster', C=None, Sigma=None,
                          C_policy=matrix('C', C_mode), Sigma_policy=noise('Sigma', Sigma_mode))
    dynamics = replace(config.dynamics, model_type='dynamic',
                       A_policy=matrix('A', A_mode), Gamma_policy=noise('Gamma', Gamma_mode),
                       lds_scoring=('final' if any(mode.startswith('shared') for mode in
                                                   (A_mode, Gamma_mode, C_mode, Sigma_mode))
                                    else config.dynamics.lds_scoring))
    hdp = replace(config.hdp, hmm_switch=True)
    return replace(config, observation=observation, dynamics=dynamics, hdp=hdp)


def with_pca_emission(config: HDPGPCConfig, estimate, *,
                      sigma_policy: NoisePolicyConfig | None = None) -> HDPGPCConfig:
    """Install a calibrated rectangular PCA decoder with a fixed C.

    Sigma defaults to fixed calibrated residual noise. A supplied noise policy
    may learn full or diagonal Sigma per cluster or across all clusters; its
    optional value replaces the calibrated covariance as the prior mean.
    A and Gamma remain learned per cluster. The latent LDS dimension is the
    PCA component count; the observation grid retains the waveform dimension.
    GP kernel fitting and warping are excluded because component indices are
    not waveform coordinates.
    """
    from ..dynamics._pca import PCAObservationEstimate

    if not isinstance(config, HDPGPCConfig) or not isinstance(estimate, PCAObservationEstimate):
        raise TypeError('expected HDPGPCConfig and PCAObservationEstimate')
    if config.multi_output.n_outputs != 1:
        raise ValueError('PCA observation currently requires one output channel')
    if config.inference.warp or config.numerics.cuda:
        raise ValueError('PCA observation currently requires warp=False and CPU inference')
    if sigma_policy is None:
        sigma_policy = NoisePolicyConfig(mode='fixed')
    if not isinstance(sigma_policy, NoisePolicyConfig):
        raise TypeError('sigma_policy must be a NoisePolicyConfig')
    if sigma_policy.mode not in ('fixed', 'cluster', 'cluster_diagonal',
                                 'shared', 'shared_diagonal'):
        raise ValueError('PCA Sigma policy must be fixed, cluster, cluster_diagonal, shared, or shared_diagonal')
    dimension = estimate.components.shape[1]
    sigma_value = (estimate.residual_covariance.copy() if sigma_policy.value is None
                   else np.asarray(sigma_policy.value).copy())
    observation = replace(
        config.observation, scope='cluster', C=None, Sigma=None,
        C_policy=MatrixPolicyConfig(mode='fixed', value=estimate.components.copy()),
        Sigma_policy=replace(sigma_policy, value=sigma_value),
        pca_grid=estimate.grid.copy(), pca_mean=estimate.mean.copy(),
        pca_variances=estimate.latent_variances.copy(),
    )
    dynamics = replace(
        config.dynamics, model_type='dynamic', bayesian_params=True,
        A_policy=MatrixPolicyConfig(mode='cluster'),
        Gamma_policy=NoisePolicyConfig(mode='cluster'), lds_scoring='final',
        lds_update=('conjugate_sequential' if (
            config.dynamics.lds_update == 'legacy' or
            (config.dynamics.lds_update == 'conjugate_batch' and
             config.inference.schedule != 'offline'))
                    else config.dynamics.lds_update),
    )
    emission = replace(config.emission, inducing_points=True, fit_gp=False)
    return replace(config, x_basis=np.arange(dimension, dtype=np.float64)[:, None],
                   observation=observation, dynamics=dynamics, emission=emission,
                   multi_output=replace(config.multi_output, use_snr=False))
