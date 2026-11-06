"""Read-only transition diagnostics shared by fixed-affine sequence experiments."""
import numpy as np
from scipy.stats import invwishart


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.array(value, dtype=float, copy=True)


def transition_radius_samples(posterior, count, seed):
    """Sample rho(A) from the fitted transition parameter posterior.

    Gamma ~ IW(n0+r+1, n0*scale); A|Gamma has row covariance Gamma and
    column covariance precision^-1. Diagonal noise uses independent inverse-gamma
    rows, and fixed noise uses its supplied covariance. Constant-input posteriors
    sample [A,b] jointly and report the radius of A, excluding the intercept.
    These are conditional model diagnostics,
    not a physical-instability probability or model-selection uncertainty.
    """
    if count < 1:
        return np.empty(0)
    fixed_A = getattr(posterior, 'fixed_transition', None)
    if fixed_A is not None:
        return np.repeat(max(abs(np.linalg.eigvals(_array(fixed_A)))), count)
    if getattr(posterior, 'fixed_matrix', False):
        return np.repeat(max(abs(np.linalg.eigvals(_array(posterior.m_mean)))), count)
    from ..dynamics._distributions import matrix_normal_inv_wishart
    from ..dynamics._parameter_posterior import ConstrainedPairPosterior
    if not isinstance(posterior, (matrix_normal_inv_wishart, ConstrainedPairPosterior)):
        raise ValueError('Stability sampling requires a supported transition parameter posterior')
    mean, precision, scale = map(_array, (posterior.m_mean, posterior.m_r_cov, posterior.scale))
    dimension = len(mean)
    rng = np.random.default_rng(seed)
    degrees = float(posterior.n0)
    noise_family = getattr(posterior, 'noise', 'full')
    if noise_family == 'fixed':
        covariances = np.broadcast_to(scale, (count, dimension, dimension))
    elif noise_family == 'diagonal':
        variances = 1. / rng.gamma(degrees / 2 + 1,
            scale=1. / (degrees * np.diag(scale) / 2), size=(count, dimension))
        covariances = np.zeros((count, dimension, dimension))
        covariances[:, np.arange(dimension), np.arange(dimension)] = variances
    else:
        covariances = invwishart.rvs(df=degrees+dimension+1, scale=degrees*scale,
                                   size=count, random_state=rng).reshape(count, dimension, dimension)
    chol = np.linalg.cholesky(precision)
    result = []
    for covariance in covariances:
        noise = rng.normal(size=mean.shape)
        # noise @ L_precision^-1 gives the required column covariance.
        right = np.linalg.solve(chol.T, noise.T).T
        A = mean+np.linalg.cholesky(covariance) @ right
        if getattr(posterior, 'constant_input', False):
            A = A[:, :-1]
        result.append(float(max(abs(np.linalg.eigvals(A)))))
    return np.asarray(result)


def affine_dynamics_diagnostics(model, paths, times, *, samples, seed, interpretation):
    labels = np.asarray(model.labels_, dtype=int)
    clusters = []
    for cluster, path in paths.items():
        indices = path['indices']
        owned = labels[indices] == cluster
        gp = model._backend.gpmodels[0][cluster]
        A = _array(gp.A[-1])
        eigenvalues = np.linalg.eigvals(A)
        radius = float(max(abs(eigenvalues)))
        differences = np.diff(times[indices])
        regular = bool(len(differences) and np.all(np.diff(indices) == 1)
                       and np.allclose(differences, np.median(differences), rtol=.01, atol=2e-6))
        step = float(np.median(differences)) if regular else None
        rows = np.flatnonzero(labels == cluster)
        blocks = int(1+np.sum(np.diff(rows) != 1)) if len(rows) else 0
        record = dict(cluster=int(cluster), map_windows=int(owned.sum()), map_blocks=blocks,
            retained_windows=len(gp.indexes), trajectory_windows=len(indices),
            dynamics_clock=getattr(gp,'dynamics_clock','cluster-clock'),
            propagation_only_windows=len(indices)-len(gp.indexes),
            regular_retained_clock=regular, step_seconds=step,
            latent_spectral_radius=radius,
            point_growth_per_second=float(np.log(radius)/step) if step and radius > 0 else None,
            eigenvalues_real=eigenvalues.real, eigenvalues_imag=eigenvalues.imag,
            interpretation=('Global regular sampling-clock evolution; not continuous-time ODE or failure identification'
                if getattr(gp,'dynamics_clock','cluster-clock')=='physical-clock' else interpretation))
        posterior = gp.internal_params
        record['transition_posterior'] = dict(family=type(posterior).__name__,
            mean=_array(posterior.m_mean), column_precision=_array(posterior.m_r_cov),
            scale=_array(posterior.scale), excess_degrees=float(posterior.n0))
        try:
            radii = transition_radius_samples(posterior, samples, seed+int(cluster))
            if len(radii):
                record.update(variational_probability_rho_gt_one=float(np.mean(radii > 1)),
                    rho_quantiles=np.quantile(radii, [.025, .5, .975]), posterior_samples=len(radii))
        except (ValueError, np.linalg.LinAlgError) as error:
            record['posterior_sampling_note'] = str(error)
        clusters.append(record)
    return clusters
