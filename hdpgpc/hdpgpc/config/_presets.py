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
    if (config.observation.pca_grid is not None or
            config.observation.fixed_grid is not None):
        raise ValueError('Fixed decoder selects C and A/Gamma; LDS presets do not apply')
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
    from ..emissions._pca import PCAObservationEstimate

    if not isinstance(config, HDPGPCConfig) or not isinstance(estimate, PCAObservationEstimate):
        raise TypeError('expected HDPGPCConfig and PCAObservationEstimate')
    from ..emissions._fixed_observation import AffineGaussianObservation

    model = AffineGaussianObservation(
        C=estimate.components, Sigma=estimate.residual_covariance,
        observation_grid=estimate.grid, offset=estimate.mean,
        latent_covariance=np.diag(estimate.latent_variances),
    )
    return with_fixed_observation_model(config, model, sigma_policy=sigma_policy)



def with_fixed_observation_model(config: HDPGPCConfig, model, *,
                                 sigma_policy: NoisePolicyConfig | None = None) -> HDPGPCConfig:
    """Install an exact fixed affine Gaussian decoder in the existing LDS.

    The full `FixedObservationModel` contract also specifies what an external
    nonlinear decoder must provide. The current conjugate LDS engine accepts
    only `AffineGaussianObservation`; a nonlinear implementation needs its own
    posterior and ELBO integration before it can be used for fitting.
    """
    from ..emissions._fixed_observation import AffineGaussianObservation

    if not isinstance(config, HDPGPCConfig) or not isinstance(model, AffineGaussianObservation):
        raise TypeError('Current LDS inference requires HDPGPCConfig and AffineGaussianObservation')
    if config.multi_output.n_outputs != 1 or config.inference.warp or config.numerics.cuda:
        raise ValueError('Fixed affine decoder requires one output, warp=False and CPU inference')
    sigma_policy = NoisePolicyConfig(mode='fixed') if sigma_policy is None else sigma_policy
    if not isinstance(sigma_policy, NoisePolicyConfig) or sigma_policy.mode not in (
            'fixed', 'cluster', 'cluster_diagonal', 'shared', 'shared_diagonal'):
        raise ValueError('Unsupported fixed affine Sigma policy')
    sigma = (model.Sigma.copy() if sigma_policy.value is None
             else np.asarray(sigma_policy.value, dtype=np.float64).copy())
    observation = replace(
        config.observation, scope='cluster', family='linear', C=None, Sigma=None,
        C_policy=MatrixPolicyConfig(mode='fixed', value=model.C.copy()),
        Sigma_policy=replace(sigma_policy, value=sigma),
        pca_grid=None, pca_mean=None, pca_variances=None,
        fixed_grid=model.grid.copy(), fixed_offset=model.offset.copy(),
        fixed_latent_covariance=model.latent_covariance.copy(),
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
    return replace(
        config, x_basis=np.arange(model.latent_dim, dtype=np.float64)[:, None],
        observation=observation, dynamics=dynamics,
        emission=replace(config.emission, inducing_points=True, fit_gp=False),
        multi_output=replace(config.multi_output, use_snr=False),
    )


def with_mfa_emission(config: HDPGPCConfig, initialization, *, dynamic=False,
                      mean_precision=1., loading_precision=1., shared=False) -> HDPGPCConfig:
    """Learn affine cluster emissions using native HDPGPC inference.

    ``initialization`` is an AffineGaussianObservation (for example a calibrated
    PCA decoder). It initializes the prior, not a frozen projection. Each
    cluster learns its intercept, rectangular loading matrix and diagonal noise.
    With shared=True, loadings and diagonal noise are pooled while intercepts
    stay cluster-specific. This mode selects conjugate_batch and requires
    offline inference; its joint posterior counts the shared prior only once.
    Static defaults fix A=0, Gamma=I, prior state covariance I and disable HMM
    assignments, except that explicit left_to_right topology retains its required
    HMM switch. ``dynamic=True`` learns cluster A/Gamma and retains the input
    HMM switch. Existing inference strategy and KL normalization are retained.
    """
    from ..emissions import AffineGaussianObservation
    if type(shared) is not bool:
        raise TypeError('shared must be boolean')
    if type(dynamic) is not bool:
        raise ValueError('dynamic must be boolean')
    if not isinstance(initialization, AffineGaussianObservation):
        raise TypeError('initialization must be AffineGaussianObservation')
    initial = replace(initialization, Sigma=np.diag(np.diag(initialization.Sigma)),
                      latent_covariance=np.eye(initialization.latent_dim))
    base = with_fixed_observation_model(config, initial)
    r = initial.latent_dim
    return replace(base,
        observation=replace(base.observation, family='factor_analysis',
            C_policy=MatrixPolicyConfig(mode='shared' if shared else 'cluster', value=initial.C.copy()),
            Sigma_policy=NoisePolicyConfig(mode='shared_diagonal' if shared else 'cluster_diagonal', value=initial.Sigma.copy()),
            factor_mean_precision=mean_precision, factor_loading_precision=loading_precision),
        dynamics=replace(base.dynamics,
            lds_update='conjugate_batch' if shared else base.dynamics.lds_update,
            A_policy=MatrixPolicyConfig(mode='cluster') if dynamic else
                     MatrixPolicyConfig(mode='fixed', value=np.zeros((r, r))),
            Gamma_policy=NoisePolicyConfig(mode='cluster') if dynamic else
                         NoisePolicyConfig(mode='fixed', value=np.eye(r)),
            annealing=False, lds_scoring='final'),
        hdp=(base.hdp if dynamic or base.hdp.transition_topology == 'left_to_right'
             else replace(base.hdp, hmm_switch=False)))
