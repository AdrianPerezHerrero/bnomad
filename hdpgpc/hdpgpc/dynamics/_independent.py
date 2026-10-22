"""Exact batch operations for fixed-zero transitions and affine observations.

Parameter learning still belongs to _trajectory. These operations only replace
independent filtering/scoring work and retain the native list-shaped histories.
"""
import torch


def compatible(emission):
    from ..emissions._fixed_observation import AffineGaussianObservation

    policies = getattr(emission, 'lds_pair_policies', None)
    return (emission.lds_update == 'conjugate_batch' and emission.lds_scoring == 'final'
            and not emission.fit_gp
            and type(getattr(emission, 'fixed_observation_model', None)) is AffineGaussianObservation
            and policies is not None and policies['transition'][0].mode == 'fixed'
            and policies['observation'][0].mode == 'fixed'
            and not emission.A[-1].requires_grad
            and torch.count_nonzero(emission.A[-1]).item() == 0
            and torch.count_nonzero(emission.Gamma[-1]).item() > 0)


def full_pass(emission, grids, observations, resp, active):
    """Batch a fresh fixed-parameter E-step; return False for scalar fallback."""
    if (not compatible(emission) or emission.N != 0 or emission.indexes
            or len(emission.f_star_sm) != 1 or len(emission.cov_f_sm) != 1
            or active.numel() == 0):
        return False
    reference = emission.direct_observation_grid
    if not torch.equal(grids, reference[None].expand_as(grids)):
        return False
    model = emission.fixed_observation_model
    values = observations[active]
    weights = resp[active]
    zero = torch.zeros_like(emission.f_star_sm[0])
    initial = emission.cov_f_sm[0]
    gamma, sigma = emission.Gamma[-1], emission.Sigma[-1]
    # Scalar filtering identifies the initial prior by exact covariance equality.
    first_prior = initial if torch.equal(initial, emission.gp.direct_initial_covariance) else gamma
    first_mean, first_cov = model.condition_batch(zero, first_prior, values[:1],
                                                 responsibilities=weights[:1], noise=sigma)
    if len(active) > 1:
        means, covariances = model.condition_batch(zero, gamma, values[1:],
                                                  responsibilities=weights[1:], noise=sigma)
        means = torch.cat((first_mean, means))
        covariances = torch.cat((first_cov, covariances))
        if not torch.equal(gamma, emission.gp.direct_initial_covariance):
            # Preserve the unusual native sentinel even if it recurs mid-path.
            recurring = (covariances[:-1] == emission.gp.direct_initial_covariance).all(dim=(1, 2))
            if bool(recurring.any()):
                return False
    else:
        means, covariances = first_mean, first_cov
    indices = active.tolist()
    emission.indexes.extend(indices)
    emission.x_train.extend(grids[index] for index in indices)
    emission.y_train.extend(values.unbind(0))
    emission.f_star.extend(means.unbind(0))
    emission.f_star_sm.extend(means.unbind(0))
    emission.cov_f.extend(covariances.unbind(0))
    emission.cov_f_sm.extend(covariances.unbind(0))
    emission.N += len(indices)
    from ._trajectory import hold_lds_params
    for _ in indices:
        hold_lds_params(emission)
    # RTS gains are exactly zero: filtered and smoothed moments coincide.
    emission.bayesian_batch_params(resp)
    return True


def observation_scores(emission, grids, observations, *, no_first=False):
    """Batch the native all-observation state lookup and Gaussian likelihood."""
    reference = emission.direct_observation_grid
    if not torch.equal(grids, reference[None].expand_as(grids)):
        return None
    indices = torch.as_tensor(emission.indexes, dtype=torch.long, device=grids.device)
    rows = torch.arange(len(grids), device=grids.device)
    states = torch.searchsorted(indices, rows, right=True)
    means = torch.stack(emission.f_star_sm)[states]
    covariances = torch.stack(emission.cov_f_sm)[states]
    sigma = emission.Sigma[-1]
    model = emission.fixed_observation_model
    scores = model.expected_log_likelihood_batch(observations, means, covariances, noise=sigma)
    if not no_first:
        row = int(indices[0])
        first_noise = sigma + torch.diag(sigma).mean() * .01 * torch.eye(
            len(sigma), dtype=sigma.dtype, device=sigma.device)
        scores[row:row + 1] = model.expected_log_likelihood_batch(
            observations[row:row + 1], means[row:row + 1], covariances[row:row + 1], noise=first_noise)
    return scores


def latent_scores(emission, count, h_ini=1.):
    """Independent Gaussian latent factors; the first state keeps its prior."""
    if len(emission.indexes) == 1:
        result = emission.f_star_sm[0].new_zeros(count)
        result[emission.indexes[0]] = emission.log_lat_error(0, h_ini)
        return result
    means = torch.stack(emission.f_star_sm[1:]).squeeze(-1)
    covariances = torch.stack(emission.cov_f_sm[1:])
    # Match emission._chol_spd's ordinary jitter, repairing failed rows with it.
    eye = emission._eye_cached(means.shape[1], means.device, means.dtype)
    symmetric = .5 * (covariances + covariances.mT)
    scale = symmetric.diagonal(dim1=-2, dim2=-1).abs().mean(-1).clamp_min(torch.finfo(means.dtype).eps)
    factors, status = torch.linalg.cholesky_ex(symmetric + 1e-8 * scale[:, None, None] * eye)
    if bool(status.any()):
        factors = factors.clone()
        for row in torch.where(status != 0)[0].tolist():
            factors[row] = emission._chol_spd(covariances[row])
    logdet = 2 * factors.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    factor = emission._chol_spd(emission.Gamma[-1])
    precision = torch.cholesky_inverse(factor)
    quadratic = torch.einsum('ni,ij,nj->n', means, precision, means)
    trace = torch.einsum('ij,nji->n', precision, covariances)
    scores = .5 * (means.shape[1] + logdet - 2 * factor.diag().log().sum() - quadratic - trace)
    # Retain the scalar first-state convention, including h_ini and prior mean.
    scores[0] = emission.log_lat_error(0, h_ini)
    result = means.new_zeros(count)
    result[torch.as_tensor(emission.indexes, device=means.device)] = scores
    return result
