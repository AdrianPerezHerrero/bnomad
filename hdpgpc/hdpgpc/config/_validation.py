"""New-API validation; never clip parameters or repair legacy algorithms."""

import math
from numbers import Integral, Real


def integer(value, name, minimum=0, optional=False):
    if optional and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def boolean(value, name):
    if type(value).__name__ not in ('bool', 'bool_'):
        raise ValueError(name + ' must be boolean')


def validate_inference(c):
    if c.schedule not in (None, 'online', 'online_fast', 'offline'):
        raise ValueError('Unknown inference schedule')
    for name in ('warp', 'classify', 'reestimate_initial_params',
                 'hard_assignments', 'birth_lds_kl_per_dimension'):
        boolean(getattr(c, name), name)
    for name in ('batch', 'force_model'):
        integer(getattr(c, name), name, optional=True)
    integer(c.minibatch, 'minibatch')
    integer(c.n_explore_steps, 'n_explore_steps')
    if c.offline_representer_selection not in ('sorted', 'fixed'):
        raise ValueError('offline_representer_selection must be sorted or fixed')
    if c.offline_strategy not in ('legacy', 'split_merge'):
        raise ValueError('offline_strategy must be legacy or split_merge')
    integer(c.split_merge_refine_steps, 'split_merge_refine_steps', 1)
    integer(c.split_merge_stability_patience, 'split_merge_stability_patience', 1)
    if (isinstance(c.split_merge_score_rtol, bool) or not isinstance(c.split_merge_score_rtol, Real)
            or not math.isfinite(c.split_merge_score_rtol) or c.split_merge_score_rtol < 0):
        raise ValueError('split_merge_score_rtol must be finite and nonnegative')
    if c.offline_strategy == 'split_merge' and c.schedule in ('online', 'online_fast'):
        raise ValueError('split_merge strategy requires offline inference')
    integer(c.it_limit, 'it_limit', 1, optional=True)
    if c.schedule == 'offline' and (c.force_model is not None or c.minibatch != 0 or c.classify):
        raise ValueError('force_model/minibatch/classify apply only to online inference')
    if c.schedule in ('online', 'online_fast') and c.it_limit is not None:
        raise ValueError('it_limit applies only to offline inference')


def validate_numerics(n):
    boolean(n.cuda, 'cuda')
    integer(n.seed, 'seed', optional=True)
    if n.seed is not None and n.seed >= 2**32:
        raise ValueError('seed must fit NumPy uint32')
    integer(n.num_threads, 'num_threads', 1, optional=True)
    if n.dtype not in (None, 'float64'):
        raise ValueError('Only the existing float64 formulation is supported')
    if n.deterministic is not None:
        boolean(n.deterministic, 'deterministic')


def validate(c, *, for_build=False):
    from ._models import SECTIONS, MatrixPolicyConfig, NoisePolicyConfig, resolved_lds_policies
    for name, typ in SECTIONS.items():
        if type(getattr(c, name)) is not typ:
            raise TypeError(name + ' must be ' + typ.__name__)
    import numpy as np
    h, e, obs, d, a, i, o, n = (getattr(c, k) for k in SECTIONS)
    if (obs.C_policy is not None or obs.Sigma_policy is not None) and obs.scope != 'cluster':
        raise ValueError('Use independent C_policy/Sigma_policy or legacy observation.scope, not both')
    if (obs.C_policy is not None or obs.Sigma_policy is not None) and (obs.C is not None or obs.Sigma is not None):
        raise ValueError('Custom C/Sigma values belong in their independent policies')
    policies = resolved_lds_policies(c)
    if obs.family not in ('linear', 'factor_analysis'):
        raise ValueError('Unknown observation family')
    factor_analysis = obs.family == 'factor_analysis'
    for name in ('factor_mean_precision', 'factor_loading_precision'):
        value = getattr(obs, name)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
            raise ValueError(name + ' must be finite and positive')
    pca_fields = (obs.pca_grid, obs.pca_mean, obs.pca_variances)
    fixed_fields = (obs.fixed_grid, obs.fixed_offset, obs.fixed_latent_covariance)
    if any(value is not None for value in pca_fields) and any(
            value is not None for value in fixed_fields):
        raise ValueError('Use either legacy PCA fields or fixed observation fields')
    pca_observation = any(value is not None for value in pca_fields + fixed_fields)
    direct_observation_dimension = None
    if factor_analysis and not pca_observation:
        raise ValueError('Factor analysis requires an affine feature grid and initialization; use with_mfa_emission')
    if pca_observation:
        direct_fields = fixed_fields if any(value is not None for value in fixed_fields) else pca_fields
        if any(value is None for value in direct_fields):
            raise ValueError('Fixed observation requires grid, offset and latent covariance')
        grid = np.asarray(direct_fields[0])
        direct_observation_dimension = grid.shape[0] if grid.ndim else 0
        mean = np.asarray(direct_fields[1])
        variances = np.asarray(direct_fields[2])
        decoder = np.asarray(policies['C'].value)
        noise = np.asarray(policies['Sigma'].value)
        latent_valid = decoder.ndim == 2 and (
            (variances.shape == (decoder.shape[1],)
             and np.all(variances > 0)) or
            (variances.shape == (decoder.shape[1], decoder.shape[1])
             and np.all(np.isfinite(variances))
             and np.allclose(variances, variances.T)
             and np.linalg.eigvalsh(variances).min() > 0))
        if (grid.ndim != 2 or grid.shape[1] != 1 or mean.shape != (len(grid),)
                or decoder.ndim != 2 or decoder.shape[0] != len(grid)
                or not latent_valid
                or noise.shape != (len(grid), len(grid))
                or not all(np.isfinite(value).all() for value in
                           (grid, mean, variances, decoder, noise))):
            raise ValueError('Fixed observation dimensions/values must match the waveform and latent grids')
        if factor_analysis and (policies['C'].mode != 'cluster'
                or policies['Sigma'].mode != 'cluster_diagonal'
                or d.lds_scoring != 'final' or d.lds_update == 'legacy'):
            raise ValueError('Factor analysis requires cluster C, cluster_diagonal Sigma, final scoring and a conjugate/natural-gradient LDS update')
        if ((policies['C'].mode != 'fixed' and not factor_analysis) or policies['Sigma'].mode not in (
                    'fixed', 'cluster', 'cluster_diagonal', 'shared', 'shared_diagonal')
                or (policies['A'].mode, policies['Gamma'].mode) not in (
                    ('cluster', 'cluster'), ('fixed', 'cluster'),
                    ('fixed', 'fixed'))
                or d.model_type != 'dynamic' or o.n_outputs != 1 or i.warp or n.cuda
                 or e.fit_gp or not e.inducing_points or i.reestimate_initial_params):
            raise ValueError('Affine observation requires a supported C/Sigma family, cluster A/Gamma or fixed A with cluster/fixed Gamma, one output, '
                             'fixed GP kernel, inducing mode, CPU and no warp/recalibration')
    for name in ('A', 'C'):
        policy = policies[name]
        if type(policy) is not MatrixPolicyConfig or policy.mode not in ('cluster', 'shrinkage', 'shared', 'fixed'):
            raise ValueError(f'{name} policy requires a valid MatrixPolicyConfig')
        if not isinstance(policy.strength, Real) or not math.isfinite(policy.strength) or policy.strength < 1:
            raise ValueError(f'{name} prior strength must be finite and >= 1')
        if policy.mode != 'shrinkage' and policy.strength != 1:
            raise ValueError(f'{name} prior strength applies only to shrinkage mode')
        if policy.mode == 'shrinkage' and policy.strength <= 1:
            raise ValueError(f'{name} shrinkage requires strength > 1')
        if policy.mode == 'shrinkage' and policy.value is not None:
            raise ValueError(f'{name} shrinkage is centered on identity; omit the custom value')
    for name in ('Gamma', 'Sigma'):
        policy = policies[name]
        if type(policy) is not NoisePolicyConfig or policy.mode not in (
                'cluster', 'cluster_diagonal', 'shared', 'shared_diagonal', 'fixed'):
            raise ValueError(f'{name} policy requires a valid NoisePolicyConfig')
        if (not isinstance(policy.strength, Real) or not math.isfinite(policy.strength)
                or policy.strength < 1):
            raise ValueError(f'{name} prior strength must be finite and >= 1')
        if policy.mode == 'fixed' and policy.strength != 1:
            raise ValueError(f'{name} fixed noise has no prior strength')
    for name, policy in policies.items():
        if policy.value is not None:
            matrix = np.asarray(policy.value)
            if (matrix.ndim != 2 or not np.all(np.isfinite(matrix))
                    or (matrix.shape[0] != matrix.shape[1]
                        and not (pca_observation and name == 'C'))):
                raise ValueError(f'{name} policy value must be a finite square matrix'
                                 + (' (except rectangular PCA C)' if pca_observation else ''))
            if name in ('Gamma', 'Sigma') and (not np.allclose(matrix, matrix.T)
                    or not np.all(np.linalg.eigvalsh(matrix) > 0)):
                raise ValueError(f'{name} policy covariance must be positive definite')
            if name in ('Gamma', 'Sigma') and 'diagonal' in policy.mode and not np.allclose(matrix, np.diag(np.diag(matrix))):
                raise ValueError(f'{name} diagonal policy value must be diagonal')
    for matrix_name, noise_name in (('A', 'Gamma'), ('C', 'Sigma')):
        matrix_mode, noise_mode = policies[matrix_name].mode, policies[noise_name].mode
        matrix_scope = 'fixed' if matrix_mode == 'fixed' else ('shared' if matrix_mode == 'shared' else 'cluster')
        noise_scope = 'fixed' if noise_mode == 'fixed' else ('shared' if noise_mode.startswith('shared') else 'cluster')
        if {matrix_scope, noise_scope} == {'shared', 'cluster'}:
            raise ValueError(f'{matrix_name}/{noise_name}: mixed shared and cluster learned scopes lack a joint MNIW posterior')
    custom_policies = any(getattr(c.observation, key) is not None for key in ('C_policy', 'Sigma_policy')) or any(
        getattr(c.dynamics, key) is not None for key in ('A_policy', 'Gamma_policy'))
    if custom_policies and not d.bayesian_params:
        raise ValueError('LDS parameter policies require bayesian_params=True')
    if custom_policies and d.lds_update == 'legacy':
        raise ValueError('Independent LDS policies require a conjugate or natural-gradient update')
    if custom_policies and d.lds_scoring == 'historical' and any(p.mode.startswith('shared') for p in policies.values()):
        raise ValueError('Shared LDS parameters require final scoring')
    if custom_policies and i.reestimate_initial_params:
        raise ValueError('LDS policies require priors set before fitting')
    if obs.scope not in ('cluster', 'shared', 'fixed'):
        raise ValueError('observation.scope must be cluster, shared, or fixed')
    if obs.scope == 'cluster' and (obs.C is not None or obs.Sigma is not None):
        raise ValueError('observation.C/Sigma require shared or fixed scope')
    for name in ('C', 'Sigma'):
        matrix = getattr(obs, name)
        if matrix is not None:
            array = np.asarray(matrix)
            if array.ndim != 2 or array.shape[0] != array.shape[1] or not np.all(np.isfinite(array)):
                raise ValueError(f'observation.{name} must be a finite square matrix')
            if name == 'Sigma' and (not np.allclose(array, array.T)
                                     or not np.all(np.linalg.eigvalsh(array) > 0)):
                raise ValueError('observation.Sigma must be symmetric positive definite')
    if obs.scope != 'cluster' and not d.bayesian_params:
        raise ValueError('shared/fixed observation parameters require bayesian_params=True')
    if obs.scope != 'cluster' and d.lds_scoring == 'historical':
        raise ValueError('shared/fixed observation parameters require final LDS scoring')
    if obs.scope != 'cluster' and i.reestimate_initial_params:
        raise ValueError('shared/fixed observation parameters require priors set before fitting')
    if (obs.scope == 'shared' or any(p.mode.startswith('shared') for p in policies.values())) and i.warp and a.policy == 'gp_flow':
        raise ValueError('shared LDS learning requires a common LDS grid')
    integer(h.M, 'M', 1, optional=True)
    if h.transition_topology not in ('ergodic', 'left_to_right'):
        raise ValueError('transition_topology must be ergodic or left_to_right')
    if h.transition_topology == 'left_to_right':
        if not h.hmm_switch:
            raise ValueError('left_to_right requires hmm_switch=True')
        if i.offline_strategy != 'split_merge' or i.schedule in ('online', 'online_fast'):
            raise ValueError('left_to_right currently requires offline_strategy=split_merge and offline inference')
        if n.cuda:
            raise ValueError('left_to_right global-weight quadrature currently requires CPU inference')
    M = 1 if h.M is None else h.M
    integer(h.max_models, 'max_models', 1, optional=True)
    if h.max_models is not None and h.max_models < M:
        raise ValueError('max_models cannot be smaller than initial M')
    if h.hdp_hyp not in ('less', 'less_extra_sticky', 'balanced', 'balanced_sticky', 'more', 'more_sticky',
                         'more_extra_sticky', 'max'):
        raise ValueError('hdp_hyp must be less, less_extra_sticky, balanced, balanced_sticky, more, more_sticky, more_extra_sticky or max')
    integer(o.n_outputs, 'n_outputs', 1)
    if (obs.scope != 'cluster' or any(p.mode.startswith('shared') for p in policies.values())) and o.n_outputs != 1:
        raise ValueError('shared/fixed observation parameters currently require one output')
    if (obs.scope != 'cluster' or any(p.mode.startswith('shared') for p in policies.values())) and n.cuda:
        raise ValueError('shared/fixed observation parameters currently require CPU inference')
    from ._verbosity import normalize_verbosity
    normalize_verbosity(c.verbose)
    for obj, names in ((h, ('hmm_switch',)),
                       (e, ('check_var', 'fit_gp')), (d, ('bayesian_params',)),
                       (o, ('share_gp', 'use_snr', 'reduce_outputs')), (n, ('cuda',))):
        for name in names:
            boolean(getattr(obj, name), name)
    if not isinstance(o.reduce_outputs_ratio, Real) or not 0 < o.reduce_outputs_ratio <= 1:
        raise ValueError('reduce_outputs_ratio must be in (0, 1]')
    if o.reduce_outputs and int(np.rint(o.n_outputs * o.reduce_outputs_ratio)) < 1:
        raise ValueError('Output reduction would leave zero outputs')
    validate_inference(i)
    if not i.hard_assignments and not d.bayesian_params:
        raise ValueError('soft assignments require bayesian_params=True for weighted LDS updates')
    if a.policy not in ('gp', 'gp_flow', 'dtw_gp', 'none'):
        raise ValueError('Unknown alignment policy')
    for name in ('flow_inducing', 'flow_features', 'flow_samples', 'flow_steps'):
        integer(getattr(a, name), name, 1)
    integer(a.flow_train_iter, 'flow_train_iter', 0)
    for name in ('flow_lengthscale', 'flow_amplitude', 'flow_lr'):
        value = getattr(a, name)
        if not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
            raise ValueError(name + ' must be finite and positive')
    for name in ('dtw_points', 'dtw_inducing'):
        integer(getattr(a, name), name, 2)
    for name in ('dtw_lengthscale', 'dtw_amplitude', 'dtw_anchor_noise'):
        value = getattr(a, name)
        if not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
            raise ValueError(name + ' must be finite and positive')
    if a.method_compute_warp not in ('standard', 'greedy', 'greedy_bound'):
        raise ValueError('Unknown alignment method')
    if a.mode_warp not in ('balanced', 'fine', 'rough'):
        raise ValueError('Unknown alignment mode')
    if not isinstance(a.noise_warp, Real) or not math.isfinite(a.noise_warp) or a.noise_warp <= 0:
        raise ValueError('noise_warp must be finite and positive')
    # NoAlignment deliberately overrides True request flags: that is valid.
    validate_numerics(n)
    integer(d.free_deg_MNIV, 'free_deg_MNIV', 1, optional=True)
    if d.lds_update not in ('conjugate_sequential', 'conjugate_batch',
                            'natural_gradient', 'legacy'):
        raise ValueError('lds_update must be conjugate_sequential, conjugate_batch, natural_gradient, or legacy')
    if d.lds_scoring not in ('final', 'historical'):
        raise ValueError('lds_scoring must be final or historical')
    if d.lds_update == 'conjugate_batch' and not d.bayesian_params:
        raise ValueError('conjugate_batch requires Bayesian LDS parameters')
    if (d.lds_update == 'conjugate_batch'
            and c.inference.schedule in ('online', 'online_fast')):
        raise ValueError('conjugate_batch LDS updates require offline inference')
    if (d.lds_update == 'conjugate_batch' and c.inference.warp
            and a.policy == 'gp_flow'):
        raise ValueError('conjugate_batch does not support GP-flow input warps')
    if (isinstance(d.lds_step_size, bool) or not isinstance(d.lds_step_size, Real)
            or not math.isfinite(d.lds_step_size) or not 0 < d.lds_step_size <= 1):
        raise ValueError('lds_step_size must be finite and in (0, 1]')
    if not isinstance(d.reest_conditions, (list, tuple)) or len(d.reest_conditions) != 3:
        raise ValueError('reest_conditions must contain min_samples, max_samples, divisor')
    for v in d.reest_conditions:
        integer(v, 'reest_conditions entry', 1)
    if d.reest_conditions[0] > d.reest_conditions[1]:
        raise ValueError('Minimum re-estimation samples exceeds maximum')

    def clusters(value, name):
        if isinstance(value, (list, np.ndarray)):
            if len(value) < M:
                raise ValueError(name + ' has fewer entries than initial M')
            return list(value)
        return [value]

    for value in clusters(d.model_type, 'model_type'):
        if value not in ('static', 'dynamic'):
            raise ValueError('model_type must be static or dynamic')
        if value == 'static' and (d.A_policy is not None or d.Gamma_policy is not None):
            raise ValueError('A/Gamma policies require a dynamic LDS; the static model has no transition')
    for obj, names in ((e, ('inducing_points',)), (d, ('annealing',)),
                       (a, ('recursive_warp', 'warp_updating'))):
        for name in names:
            for value in clusters(getattr(obj, name), name):
                boolean(value, name)
    for value in clusters(d.estimation_limit, 'estimation_limit'):
        integer(value, 'estimation_limit', 1, optional=True)
    for obj, names in ((e, ('ini_lengthscale', 'ini_sigma', 'ini_outputscale')), (d, ('ini_gamma',))):
        for name in names:
            for value in clusters(getattr(obj, name), name):
                if value is None:
                    continue
                array = np.asarray(value)
                if array.dtype.kind not in 'iuf' or not np.all(np.isfinite(array)):
                    raise ValueError(name + ' must contain finite numerical values')
                if array.ndim == 0 and value <= 0:
                    raise ValueError(name + ' must be positive')
    for obj, names in ((e, ('bound_lengthscale', 'bound_sigma')), (d, ('bound_gamma',)), (a, ('bound_noise_warp',))):
        for name in names:
            for value in clusters(getattr(obj, name), name):
                if value is None:
                    continue
                bounds = np.asarray(value)
                if bounds.shape != (2,) or bounds.dtype.kind not in 'iuf' or not np.all(np.isfinite(bounds)):
                    raise ValueError(name + ' must be a bound pair (or per-cluster pairs)')
                if bounds[0] <= 0 or bounds[1] < bounds[0]:
                    raise ValueError(name + ' requires 0 < lower <= upper')
    if e.kernels is not None:
        from sklearn.gaussian_process.kernels import Kernel
        if not isinstance(e.kernels, (list, tuple)) or len(e.kernels) < M or not all(isinstance(k, Kernel) for k in e.kernels):
            raise ValueError('kernels must contain at least M sklearn kernels')
        from ._codec import encode
        for kernel in e.kernels:
            encode(kernel)  # reject nonserializable custom classes/callables explicitly
    for value, name in ((c.x_basis, 'x_basis'), (a.x_basis_warp, 'x_basis_warp')):
        if value is None:
            continue
        grids = value if isinstance(value, list) else [value]
        if isinstance(value, list) and len(value) < M:
            raise ValueError(name + ' has fewer grids than initial M')
        for grid in grids:
            if not isinstance(grid, np.ndarray) or grid.ndim not in (1, 2) or grid.size == 0:
                raise ValueError(name + ' requires nonempty NumPy grids (or a list of grids)')
            if grid.ndim == 2 and grid.shape[1] != 1:
                raise ValueError(name + ' must be one-dimensional coordinates')
            if not np.all(np.isfinite(grid)):
                raise ValueError(name + ' must contain finite coordinates')
    if for_build:
        if n.cuda:
            import torch
            if not torch.cuda.is_available():
                raise ValueError('CUDA requested but unavailable on this host')
        if c.x_basis is None:
            raise ValueError('x_basis is required to build a model')
        grids_all = c.x_basis if isinstance(c.x_basis, list) else [c.x_basis]
        if custom_policies:
            if any(p.mode.startswith('shared') for p in policies.values()) and any(
                    not np.array_equal(grids_all[0], grid) for grid in grids_all[1:]):
                raise ValueError('Shared LDS parameters require a common LDS grid')
            dimension = len(grids_all[0])
            for name, policy in policies.items():
                required_shape = ((direct_observation_dimension, dimension) if pca_observation and name == 'C'
                                  else (direct_observation_dimension, direct_observation_dimension)
                                  if pca_observation and name == 'Sigma'
                                  else (dimension, dimension))
                if policy.value is not None and np.asarray(policy.value).shape != required_shape:
                    raise ValueError(f'{name} policy value must match dimension {required_shape}')
        if obs.scope != 'cluster':
            grids = c.x_basis if isinstance(c.x_basis, list) else [c.x_basis]
            if any(not np.array_equal(grids[0], grid) for grid in grids[1:]):
                raise ValueError('shared/fixed observation parameters require a common LDS grid')
            dimension = len(grids[0])
            for name in ('C', 'Sigma'):
                matrix = getattr(obs, name)
                if matrix is not None and np.asarray(matrix).shape != (dimension, dimension):
                    raise ValueError(f'observation.{name} must match the LDS grid dimension {dimension}')
        required = [(e.ini_sigma, 'ini_sigma'), (d.ini_gamma, 'ini_gamma'),
                    (e.bound_sigma, 'bound_sigma'), (d.bound_gamma, 'bound_gamma'),
                    (a.bound_noise_warp, 'bound_noise_warp')]
        if e.kernels is None:
            required += [(e.ini_lengthscale, 'ini_lengthscale'), (e.bound_lengthscale, 'bound_lengthscale')]
        for value, name in required:
            if any(v is None for v in clusters(value, name)[:M]):
                raise ValueError(name + ' must be supplied; its legacy default remains None')
