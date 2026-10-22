"""Conjugate LDS updates when a matrix or its covariance is constrained.

The unrestricted full/full case continues to use the original MNIW class.
For diagonal row noise, each row has its own normal-inverse-gamma posterior.
"""

import torch

from ._distributions import _inverse_wishart_kl
from ._distributions import matrix_normal_inv_wishart


def make_pair_prior(mean, scale, degrees, matrix_policy, noise_policy):
    """Create the exact conjugate family selected for one LDS parameter pair."""
    mean = torch.as_tensor(mean, dtype=torch.float64)
    scale = torch.as_tensor(scale, dtype=mean.dtype, device=mean.device)
    if 'diagonal' in noise_policy.mode:
        scale = torch.diag(torch.diag(scale))
    precision = torch.eye(mean.shape[1], dtype=mean.dtype, device=mean.device)
    if matrix_policy.mode == 'shrinkage':
        precision = matrix_policy.strength * precision
    degrees = degrees * noise_policy.strength
    if matrix_policy.mode != 'fixed' and noise_policy.mode in ('cluster', 'shared'):
        return matrix_normal_inv_wishart(mean, precision, degrees, scale)
    return ConstrainedPairPosterior(
        mean, precision, degrees, scale,
        fixed_matrix=matrix_policy.mode == 'fixed',
        noise=('fixed' if noise_policy.mode == 'fixed' else
               'diagonal' if 'diagonal' in noise_policy.mode else 'full'))


class ConstrainedPairPosterior:
    def __init__(self, mean, precision, n0, scale, *, fixed_matrix=False,
                 noise='full'):
        self.m_mean = torch.as_tensor(mean).clone()
        self.m_r_cov = torch.as_tensor(precision, dtype=self.m_mean.dtype,
                                     device=self.m_mean.device).clone()
        self.n0 = n0
        self.scale = torch.as_tensor(scale, dtype=self.m_mean.dtype,
                                     device=self.m_mean.device).clone()
        self.fixed_matrix = fixed_matrix
        self.noise = noise  # full, diagonal, or fixed

    def get_mean(self):
        return self.m_mean

    def get_scale(self, final=False):
        return self.scale

    def set_scale(self, scale):
        self.scale = torch.as_tensor(scale, dtype=self.scale.dtype,
                                     device=self.scale.device)

    def posterior(self, n_k, y1, y2, cov, cov_, cov_cross,
                  sse_matrix=None, annealing=False):
        if self.fixed_matrix and self.noise == 'fixed':
            return self
        if sse_matrix is not None:
            y1, y2 = sse_matrix @ y1, sse_matrix @ y2
            cov = sse_matrix @ cov @ sse_matrix.T
            cov_ = sse_matrix @ cov_ @ sse_matrix.T
            cov_cross = sse_matrix @ cov_cross @ sse_matrix.T
        s_xx = y2 @ y2.T + cov_
        s_yx = y1 @ y2.T + cov_cross
        if self.fixed_matrix:
            mean, precision = self.m_mean, self.m_r_cov
        else:
            precision = self.m_r_cov + s_xx
            cross = self.m_mean @ self.m_r_cov + s_yx
            mean = torch.cholesky_solve(cross.T, torch.linalg.cholesky(
                0.5 * (precision + precision.T))).T
        if self.noise == 'fixed':
            scale = self.scale
        else:
            error = y1 - mean @ y2
            residual = (error @ error.T + cov - mean @ cov_cross.T
                        - cov_cross @ mean.T + mean @ cov_ @ mean.T)
            if not self.fixed_matrix:
                delta = mean - self.m_mean
                residual = residual + delta @ self.m_r_cov @ delta.T
            residual = 0.5 * (residual + residual.T)
            scale = (self.n0 * self.scale + residual) / (self.n0 + n_k)
            if self.noise == 'diagonal':
                scale = torch.diag(torch.diag(scale))
        return ConstrainedPairPosterior(
            mean, precision, self.n0 + n_k, scale,
            fixed_matrix=self.fixed_matrix, noise=self.noise)

    def posterior_natural_gradient(self, n_k, y1, y2, cov, cov_, cov_cross,
                                   sse_matrix=None, annealing=False, step_size=0.2):
        root = step_size ** 0.5
        return self.posterior(step_size * n_k, root * y1, root * y2,
                              step_size * cov, step_size * cov_,
                              step_size * cov_cross, sse_matrix, annealing)

    def kl_divergence(self, prior):
        zero = self.scale.new_zeros(())
        if self.noise == 'full':
            d = self.scale.shape[0]
            nu_q, nu_p = self.n0 + d + 1, prior.n0 + d + 1
            return _inverse_wishart_kl(nu_q, self.n0 * self.scale,
                                       nu_p, prior.n0 * prior.scale)
        if self.noise == 'diagonal':
            aq, ap = self.n0 / 2 + 1, prior.n0 / 2 + 1
            bq = self.n0 * torch.diag(self.scale) / 2
            bp = prior.n0 * torch.diag(prior.scale) / 2
            kl = (ap * torch.log(bq / bp) + torch.lgamma(bq.new_tensor(ap))
                  - torch.lgamma(bq.new_tensor(aq))
                  + (aq - ap) * torch.digamma(bq.new_tensor(aq))
                  + aq * (bp / bq - 1)).sum()
            if self.fixed_matrix:
                return kl
            expected_precision = aq / bq
        else:
            kl = zero
            expected_precision = None
        if self.fixed_matrix:
            return kl
        d, p = self.m_mean.shape
        chol_q = torch.linalg.cholesky(self.m_r_cov)
        chol_p = torch.linalg.cholesky(prior.m_r_cov)
        logdet = 2 * (torch.log(torch.diag(chol_q)).sum()
                      - torch.log(torch.diag(chol_p)).sum())
        trace = torch.trace(torch.cholesky_solve(prior.m_r_cov, chol_q))
        delta = self.m_mean - prior.m_mean
        row_quad = delta @ prior.m_r_cov @ delta.T
        mean_term = ((expected_precision * torch.diag(row_quad)).sum()
                     if expected_precision is not None else
                     torch.trace(torch.linalg.solve(self.scale, row_quad)))
        return kl + 0.5 * (d * (logdet + trace - p)
                           + mean_term)

    def _convert(self, target):
        for name in ('m_mean', 'm_r_cov', 'scale'):
            value = getattr(self, name)
            if target == 'numpy':
                if isinstance(value, torch.Tensor):
                    setattr(self, name, value.detach().cpu().numpy())
            else:
                setattr(self, name, torch.as_tensor(value).to(target))

    def to_numpy(self):
        self._convert('numpy')

    def to_torch(self):
        self._convert('cpu')

    def to_cpu(self):
        self._convert('cpu')

    def to_cuda(self):
        self._convert('cuda')
