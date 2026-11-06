"""Conjugate affine latent transitions without augmenting the evolving state.

The extra regression column is the deterministic input one. Its coefficient is
a parameter, never an LDS state with a singular process covariance. Existing
zero-input priors are returned unchanged when the extension is disabled.
"""
from types import SimpleNamespace

import torch

from ._distributions import matrix_normal_inv_wishart
from ._parameter_posterior import ConstrainedPairPosterior, make_pair_prior


def _weights(mass, columns, reference, weights):
    if columns < 1 or not float(mass) > 0:
        raise ValueError('Affine transition updates require positive transition mass')
    roots = (reference.new_full((1, columns), (float(mass) / columns) ** .5)
             if weights is None else torch.as_tensor(
                 weights, dtype=reference.dtype, device=reference.device).reshape(1, -1))
    if (roots.shape[1] != columns or not torch.isfinite(roots).all()
            or bool((roots < 0).any()) or not torch.allclose(
                roots.square().sum(), torch.as_tensor(mass, dtype=roots.dtype,
                                                      device=roots.device))):
        raise ValueError('Intercept weights must preserve the transition responsibility mass')
    return roots


def augment_transition_statistics(mass, y, x, yy, xx, yx, *, intercept_weights=None):
    """Append a constant-input predictor to responsibility-scaled moments."""
    roots = _weights(mass, x.shape[1], x, intercept_weights)
    augmented = torch.cat((x, roots), dim=0)
    covariance = xx.new_zeros((len(xx) + 1, len(xx) + 1))
    covariance[:-1, :-1] = xx
    cross = yx.new_zeros((yx.shape[0], yx.shape[1] + 1))
    cross[:, :-1] = yx
    return augmented, covariance, cross


def make_transition_prior(A, Gamma, degrees, matrix_policy=None, noise_policy=None, *,
                          constant_input=False, intercept=None, intercept_strength=1.):
    """Build the original prior, or its optional affine regression extension."""
    if not constant_input:
        if matrix_policy is not None and noise_policy is not None:
            return make_pair_prior(A, Gamma, degrees, matrix_policy, noise_policy)
        A = torch.as_tensor(A)
        return matrix_normal_inv_wishart(A, torch.eye(
            A.shape[1], dtype=A.dtype, device=A.device), degrees, Gamma)
    matrix_policy = matrix_policy or SimpleNamespace(mode='cluster', strength=1.)
    noise_policy = noise_policy or SimpleNamespace(mode='cluster', strength=1.)
    A = torch.as_tensor(A, dtype=torch.float64)
    Gamma = torch.as_tensor(Gamma, dtype=A.dtype, device=A.device)
    strength = float(intercept_strength)
    if not 0 < strength < float('inf'):
        raise ValueError('intercept_strength must be positive and finite')
    b = A.new_zeros((len(A), 1)) if intercept is None else torch.as_tensor(
        intercept, dtype=A.dtype, device=A.device).reshape(-1, 1)
    if b.shape != (len(A), 1) or not torch.isfinite(b).all():
        raise ValueError('Transition intercept must match the latent dimension and be finite')
    precision = torch.eye(A.shape[1] + 1, dtype=A.dtype, device=A.device)
    if matrix_policy.mode == 'shrinkage':
        precision[:-1, :-1] *= matrix_policy.strength
    precision[-1, -1] = strength
    noise = ('fixed' if noise_policy.mode == 'fixed' else
             'diagonal' if 'diagonal' in noise_policy.mode else 'full')
    if noise == 'diagonal':
        Gamma = torch.diag(torch.diag(Gamma))
    return AffineTransitionPosterior(torch.cat((A, b), dim=1), precision,
        degrees * noise_policy.strength, Gamma,
        fixed_transition=A if matrix_policy.mode == 'fixed' else None, noise=noise)


class AffineTransitionPosterior(ConstrainedPairPosterior):
    """Posterior for [A,b] with optional fixed A and covariance constraints.

    Full noise gives a matrix-normal/inverse-Wishart family. Diagonal noise
    gives independent normal/inverse-gamma rows. Fixed A leaves a conjugate
    Gaussian coefficient for b, rather than accidentally fixing b as well.
    """
    constant_input = True
    transition_intercept = True

    def __init__(self, mean, precision, n0, scale, *, fixed_transition=None, noise='full'):
        super().__init__(mean, precision, n0, scale, noise=noise)
        self.fixed_transition = (None if fixed_transition is None else
            torch.as_tensor(fixed_transition, dtype=self.m_mean.dtype,
                            device=self.m_mean.device).clone())

    def get_mean(self):
        return self.m_mean[:, :-1]

    def get_intercept(self):
        return self.m_mean[:, -1:]

    def _free_distribution(self):
        if self.fixed_transition is None:
            return self
        return ConstrainedPairPosterior(self.m_mean[:, -1:], self.m_r_cov[-1:, -1:],
                                        self.n0, self.scale, noise=self.noise)

    def posterior(self, n_k, y1, y2, cov, cov_, cov_cross,
                  sse_matrix=None, annealing=False, *, intercept_weights=None):
        if sse_matrix is not None:
            y1, y2 = sse_matrix @ y1, sse_matrix @ y2
            cov = sse_matrix @ cov @ sse_matrix.T
            cov_ = sse_matrix @ cov_ @ sse_matrix.T
            cov_cross = sse_matrix @ cov_cross @ sse_matrix.T
        if self.fixed_transition is None:
            augmented, xx, yx = augment_transition_statistics(
                n_k, y1, y2, cov, cov_, cov_cross, intercept_weights=intercept_weights)
            updated = ConstrainedPairPosterior.posterior(
                self, n_k, y1, augmented, cov, xx, yx, annealing=annealing)
            mean, precision = updated.m_mean, updated.m_r_cov
        else:
            A = self.fixed_transition
            roots = _weights(n_k, y2.shape[1], y2, intercept_weights)
            residual = y1 - A @ y2
            residual_cov = cov - A @ cov_cross.T - cov_cross @ A.T + A @ cov_ @ A.T
            updated = self._free_distribution().posterior(
                n_k, residual, roots, residual_cov, roots.new_zeros((1, 1)),
                residual.new_zeros((len(A), 1)), annealing=annealing)
            mean = torch.cat((A, updated.m_mean), dim=1)
            precision = self.m_r_cov.clone()
            precision[-1:, -1:] = updated.m_r_cov
        return type(self)(mean, precision, updated.n0, updated.scale,
                          fixed_transition=self.fixed_transition, noise=self.noise)

    def posterior_natural_gradient(self, n_k, y1, y2, cov, cov_, cov_cross,
                                   sse_matrix=None, annealing=False, step_size=.2, *,
                                   intercept_weights=None):
        if not 0 < step_size <= 1:
            raise ValueError('step_size must be in (0, 1]')
        root = step_size ** .5
        roots = _weights(n_k, y2.shape[1], y2, intercept_weights)
        return self.posterior(step_size * n_k, root * y1, root * y2,
            step_size * cov, step_size * cov_, step_size * cov_cross,
            sse_matrix=sse_matrix, annealing=annealing, intercept_weights=root * roots)

    def posterior_legacy(self, *args, **kwargs):
        """Use the valid affine conjugate update for the legacy schedule.

        The historical square-matrix heuristic has no rectangular constant-input
        counterpart. Feature-off calls continue to use that original heuristic.
        """
        return self.posterior(*args, **kwargs)

    def kl_divergence(self, prior):
        posterior_free, prior_free = self._free_distribution(), prior._free_distribution()
        if self.noise == 'full':
            return matrix_normal_inv_wishart.kl_divergence(posterior_free, prior_free)
        return ConstrainedPairPosterior.kl_divergence(posterior_free, prior_free)

    def log_likelihood_MNIW(self, matrix, covariance, n0=None):
        matrix = torch.as_tensor(matrix, dtype=self.m_mean.dtype, device=self.m_mean.device)
        if matrix.shape == self.get_mean().shape:
            matrix = torch.cat((matrix, self.get_intercept()), dim=1)
        distribution = self._free_distribution()
        if self.fixed_transition is not None:
            matrix = matrix[:, -1:]
        return matrix_normal_inv_wishart.log_likelihood_MNIW(
            distribution, matrix, covariance, n0)

    def _convert(self, target):
        super()._convert(target)
        if self.fixed_transition is not None:
            value = self.fixed_transition
            self.fixed_transition = (value.detach().cpu().numpy() if target == 'numpy'
                and isinstance(value, torch.Tensor) else value if target == 'numpy'
                else torch.as_tensor(value).to(target))


def _positive(matrix):
    matrix = .5 * (matrix + matrix.T)
    minimum = torch.linalg.eigvalsh(matrix).min()
    floor = matrix.new_tensor(1e-8)
    return matrix + torch.eye(len(matrix), dtype=matrix.dtype, device=matrix.device) * (
        floor - minimum).clamp_min(0.)


def fit_transition_moments(A, Gamma, mass, y, x, yy, xx, yx, *, intercept=None,
                           intercept_weights=None, matrix_policy=None, noise_policy=None):
    """Affine ML update from expected sufficient statistics, respecting constraints."""
    fixed_A = matrix_policy is not None and matrix_policy.mode == 'fixed'
    if fixed_A:
        roots = _weights(mass, x.shape[1], x, intercept_weights)
        b = ((y - A @ x) @ roots.T) / mass
        coefficients = torch.cat((A, b), dim=1)
    augmented, latent_covariance, cross = augment_transition_statistics(
        mass, y, x, yy, xx, yx, intercept_weights=intercept_weights)
    if not fixed_A:
        sxx = augmented @ augmented.T + latent_covariance
        syx = y @ augmented.T + cross
        coefficients = torch.linalg.solve(_positive(sxx), syx.T).T
    error = y - coefficients @ augmented
    residual = (error @ error.T + yy - coefficients @ cross.T
                - cross @ coefficients.T + coefficients @ latent_covariance @ coefficients.T)
    if noise_policy is not None and noise_policy.mode == 'fixed':
        noise = Gamma
    else:
        noise = residual / mass
        if noise_policy is not None and 'diagonal' in noise_policy.mode:
            noise = torch.diag(torch.diag(noise))
        noise = _positive(noise)
    return coefficients[:, :-1], coefficients[:, -1:], noise
