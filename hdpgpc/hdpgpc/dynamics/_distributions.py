"""Conjugate distributions for LDS parameter updates; native pickle paths."""

import numpy as np
import torch


def _spd_cholesky(matrix):
    return torch.linalg.cholesky(0.5 * (matrix + matrix.T))


def _logdet_cholesky(chol):
    return 2.0 * torch.log(torch.diagonal(chol)).sum()


def _inverse_wishart_kl(nu_q, psi_q, nu_p, psi_p):
    """KL(IW(nu_q, psi_q) || IW(nu_p, psi_p)), with scale matrices psi."""
    d = psi_q.shape[0]
    nu_q = torch.as_tensor(nu_q, dtype=psi_q.dtype, device=psi_q.device)
    nu_p = torch.as_tensor(nu_p, dtype=psi_q.dtype, device=psi_q.device)
    psi_p = torch.as_tensor(psi_p, dtype=psi_q.dtype, device=psi_q.device)
    L_q = _spd_cholesky(psi_q)
    L_p = _spd_cholesky(psi_p)
    logdet_q = _logdet_cholesky(L_q)
    logdet_p = _logdet_cholesky(L_p)
    log_two = torch.log(psi_q.new_tensor(2.0))
    indices = torch.arange(1, d + 1, dtype=psi_q.dtype, device=psi_q.device)
    expected_logdet = (logdet_q - d * log_two
                       - torch.digamma((nu_q + 1.0 - indices) / 2.0).sum())
    log_normalizer_q = (0.5 * nu_q * (logdet_q - d * log_two)
                        - torch.special.multigammaln(nu_q / 2.0, d))
    log_normalizer_p = (0.5 * nu_p * (logdet_p - d * log_two)
                        - torch.special.multigammaln(nu_p / 2.0, d))
    trace_scale = torch.trace(torch.cholesky_solve(psi_p, L_q))
    return (log_normalizer_q - log_normalizer_p
            - 0.5 * (nu_q - nu_p) * expected_logdet
            + 0.5 * nu_q * (trace_scale - d))


class matrix_normal_inv_wishart():
    """MNIW prior for an LDS matrix and its row covariance.

    ``m_r_cov`` is the matrix-normal column precision. ``scale`` is the
    inverse-Wishart covariance mean; ``n0`` is its degrees of freedom in
    excess of ``row_dimension + 1``. This keeps the configured small prior
    strengths valid when the observation vector has many points.
    """
    def __init__(self, m_mean, m_r_cov, n0, scale):
        if type(m_mean) is torch.Tensor:
            self.m_mean = m_mean
        else:
            self.m_mean = torch.from_numpy(m_mean)
        if type(m_r_cov) is torch.Tensor:
            self.m_r_cov = m_r_cov
        else:
            self.m_r_cov = torch.from_numpy(m_r_cov)
        self.n0 = n0
        self.scale = scale
        if type(scale) is torch.Tensor:
            self.set_scale(scale)
        else:
            self.set_scale(scale)
        self.b = 0.7

    def posterior(self, n_k, y1, y2, cov, cov_, cov_cross, sse_matrix=None, annealing=False):
        new_n0 = self.n0 + n_k

        device = self.scale.device
        dtype = self.scale.dtype
        d = self.scale.shape[0]
        id_d = torch.eye(d, device=device, dtype=dtype)

        if sse_matrix is None:
            sse_matrix = id_d

        observations = sse_matrix @ y1
        latents = sse_matrix @ y2
        projected_cov = sse_matrix @ cov @ sse_matrix.T
        projected_cross = sse_matrix @ cov_cross @ sse_matrix.T
        projected_latent_cov = sse_matrix @ cov_ @ sse_matrix.T
        s_xx = latents @ latents.T + projected_latent_cov
        s_yx = observations @ latents.T + projected_cross

        prior_precision = self.m_r_cov
        new_precision = prior_precision + s_xx
        weighted_cross = self.m_mean @ prior_precision + s_yx
        chol = torch.linalg.cholesky(0.5 * (new_precision + new_precision.T))
        new_mean = torch.cholesky_solve(weighted_cross.T, chol).T

        # Equivalent to Psi_0 + S_yy + M_0 Lambda_0 M_0^T
        # - M_n Lambda_n M_n^T. Form the data residual directly so a
        # fractional observation does not lose PSD to cancellation.
        delta = new_mean - self.m_mean
        data_residual = observations - new_mean @ latents
        residual = (data_residual @ data_residual.T
                    + projected_cov
                    - new_mean @ projected_cross.T
                    - projected_cross @ new_mean.T
                    + new_mean @ projected_latent_cov @ new_mean.T
                    + delta @ prior_precision @ delta.T)
        new_scale = (self.n0 * self.scale + residual) / new_n0
        new_scale = 0.5 * (new_scale + new_scale.T)
        return matrix_normal_inv_wishart(new_mean, new_precision, new_n0, new_scale)

    def posterior_natural_gradient(self, n_k, y1, y2, cov, cov_, cov_cross,
                                   sse_matrix=None, annealing=False, step_size=0.2):
        """Damp a conjugate update in MNIW natural-parameter space.

        The full posterior adds (S_xx, S_yx, S_yy, n_k) to the canonical
        parameters. A step of ``step_size`` adds that fraction of all four
        statistics. Scaling observations by its square root and covariance
        statistics by the step gives exactly those increments while retaining
        the stable residual-form covariance calculation in ``posterior``.
        """
        if not 0 < step_size <= 1:
            raise ValueError('step_size must be in (0, 1]')
        root = step_size ** 0.5
        return self.posterior(
            step_size * n_k, root * y1, root * y2,
            step_size * cov, step_size * cov_, step_size * cov_cross,
            sse_matrix=sse_matrix, annealing=annealing,
        )

    def posterior_legacy(self, n_k, y1, y2, cov, cov_, cov_cross,
                         sse_matrix=None, annealing=False):
        """Historical pre-conjugate LDS update, retained for comparison only."""
        new_n0 = self.n0 + n_k
        device, dtype = self.scale.device, self.scale.dtype
        d = self.scale.shape[0]
        id_d = torch.eye(d, device=device, dtype=dtype)
        if sse_matrix is None:
            sse_matrix = id_d

        precision_as_scale = 0.5 * (self.m_r_cov + self.m_r_cov.T)
        jitter = 1e-2 * torch.mean(torch.diag(self.scale).abs()).clamp_min(torch.finfo(dtype).eps)
        L_scale = torch.linalg.cholesky(precision_as_scale + jitter * id_d)
        scale_inv = torch.cholesky_solve(id_d, L_scale)

        y2p = sse_matrix @ y2
        y1p = sse_matrix @ y1
        exp_f_f_ = y2p @ y2p.T + sse_matrix @ cov_ @ sse_matrix.T
        exp_ff_ = y1p @ y2p.T + sse_matrix @ cov_cross @ sse_matrix.T
        S__ = exp_f_f_ + scale_inv
        S_ = exp_ff_ + self.m_mean @ scale_inv
        L_S__ = torch.linalg.cholesky(0.5 * (S__ + S__.T) + 1e-8 * id_d)
        part_mean = torch.cholesky_solve(S_.T, L_S__).T

        if n_k == 1:
            new_mean = ((self.n0 - 2) * self.m_mean + part_mean) / (new_n0 - 2)
            error = y1p - y2p
        else:
            new_mean = part_mean
            error = y1 - new_mean @ y2
        new_scale = ((self.n0 - 2) * self.scale + error @ error.T) / (new_n0 - 2)
        return matrix_normal_inv_wishart(new_mean, S__, new_n0, new_scale)

    def log_likelihood_MNIW(self, M, Sigma, n0):
        """Compute MNIW log-likelihood with cheaper SPD solves."""
        d = M.shape[0]
        device = self.scale.device
        dtype = self.scale.dtype
        id_d = torch.eye(d, device=device, dtype=dtype)

        L_sig = torch.linalg.cholesky(0.5 * (Sigma + Sigma.T) + 1e-8 * id_d)
        D = M - self.m_mean

        sig_inv_D = torch.cholesky_solve(D, L_sig)
        mean_lik = -0.5 * torch.sum((D @ self.m_r_cov) * sig_inv_D)

        sig_inv_scale = torch.cholesky_solve(self.scale, L_sig)
        scale_lik = -0.5 * torch.trace(sig_inv_scale)

        return mean_lik + scale_lik

    def kl_divergence(self, prior):
        """Analytic KL from this MNIW posterior to ``prior``.

        M | Sigma has row covariance Sigma and column precision m_r_cov;
        Sigma ~ IW(n0 + row_dimension + 1, n0 * scale).
        """
        d, p = self.m_mean.shape
        device, dtype = self.m_mean.device, self.m_mean.dtype
        prior_mean = torch.as_tensor(prior.m_mean, dtype=dtype, device=device)
        prior_precision = torch.as_tensor(prior.m_r_cov, dtype=dtype, device=device)
        posterior_precision = torch.as_tensor(self.m_r_cov, dtype=dtype, device=device)
        prior_scale = torch.as_tensor(prior.scale, dtype=dtype, device=device)
        posterior_scale = torch.as_tensor(self.scale, dtype=dtype, device=device)
        n_q = torch.as_tensor(self.n0, dtype=dtype, device=device)
        n_p = torch.as_tensor(prior.n0, dtype=dtype, device=device)
        psi_q, psi_p = n_q * posterior_scale, n_p * prior_scale
        nu_q, nu_p = n_q + d + 1, n_p + d + 1

        L_q = _spd_cholesky(posterior_precision)
        L_p = _spd_cholesky(prior_precision)
        trace_precision = torch.trace(torch.cholesky_solve(prior_precision, L_q))
        delta = self.m_mean - prior_mean
        mean_quadratic = delta @ prior_precision @ delta.T
        mean_term = nu_q * torch.trace(torch.cholesky_solve(
            mean_quadratic, _spd_cholesky(psi_q)))
        matrix_kl = 0.5 * (d * (_logdet_cholesky(L_q) - _logdet_cholesky(L_p)
                                + trace_precision - p) + mean_term)
        return _inverse_wishart_kl(nu_q, psi_q, nu_p, psi_p) + matrix_kl

    def get_mean(self):
        return self.m_mean

    def get_scale(self, final=False):
        return self.scale


    def set_scale(self, scale):
        if type(self.scale) is torch.Tensor:
            if self.scale.is_cuda:
                self.scale = scale.cuda()
            else:
                self.scale = scale
        else:
            self.scale = torch.from_numpy(scale)

    def cond_to_numpy(self, x):
        if x is not None:
            if type(x) is torch.Tensor:
                x = x.detach().numpy()
        return x

    def cond_to_torch(self, x):
        if x is not None:
            if type(x) is not torch.Tensor:
                x = torch.from_numpy(np.array(x))
                x.requires_grad = False
        return x

    def to_numpy(self):
        self.m_mean = self.cond_to_numpy(self.m_mean)
        self.m_r_cov = self.cond_to_numpy(self.m_r_cov)
        self.scale = self.cond_to_numpy(self.scale)

    def to_torch(self):
        self.m_mean = self.cond_to_torch(self.m_mean)
        self.m_r_cov = self.cond_to_torch(self.m_r_cov)
        self.scale = self.cond_to_torch(self.scale)

    def to_cuda(self):
        self.m_mean = self.m_mean.cuda()
        self.m_r_cov = self.m_r_cov.cuda()
        self.scale = self.scale.cuda()

    def to_cpu(self):
        self.m_mean = self.m_mean.cpu()
        self.m_r_cov = self.m_r_cov.cpu()
        self.scale = self.scale.cpu()


class inv_wishart():
    """ Class to define the Inverse Wishart distribution (IW) for the LDS static parameters.
    """
    def __init__(self, n0, scale, C_fixed):
        if type(C_fixed) is torch.Tensor:
            self.C_fixed = C_fixed
        else:
            self.m_mean = torch.from_numpy(C_fixed)
        if type(scale) is torch.Tensor:
            self.scale = scale
        else:
            self.scale = torch.from_numpy(scale)
        self.n0 = n0
        self.b = 0.7

    def posterior(self, n_k, y1, y2, cov, cov_, cov_cross, sse_matrix=None, annealing=False):
        y_k = y1
        y_k_ = y2
        new_n0 = self.n0 + n_k
        id = torch.eye(self.scale.shape[0])
        if torch.cuda.is_available() and self.scale.is_cuda:
            id = id.cuda()
        if sse_matrix is None:
            sse_matrix = id
        # e = (y_k - torch.matmul(self.C_fixed, y_k_))
        e = (y_k - y_k_)
        e2 =  torch.matmul(e, e.T)
        e2 = torch.linalg.multi_dot([sse_matrix, e2, sse_matrix.T])
        new_scale = ((self.n0 - 2) * self.scale + e2) / (new_n0 - 2)
        return inv_wishart(new_n0, new_scale, self.C_fixed)

    def posterior_natural_gradient(self, n_k, y1, y2, cov, cov_, cov_cross,
                                   sse_matrix=None, annealing=False, step_size=0.2):
        """Temper the static inverse-Wishart data increment by ``step_size``."""
        if not 0 < step_size <= 1:
            raise ValueError('step_size must be in (0, 1]')
        root = step_size ** 0.5
        return self.posterior(
            step_size * n_k, root * y1, root * y2,
            step_size * cov, step_size * cov_, step_size * cov_cross,
            sse_matrix=sse_matrix, annealing=annealing,
        )

    def get_scale(self, final=False):
        # return self.scale / (self.n0 - self.scale.shape[0] - 1)
        return self.scale * self.n0 / (self.n0 - 2)

    def kl_divergence(self, prior):
        """Analytic IW KL using the static posterior's (n0 - 2) scale update."""
        d = self.scale.shape[0]
        device, dtype = self.scale.device, self.scale.dtype
        n_q = torch.as_tensor(self.n0, dtype=dtype, device=device)
        n_p = torch.as_tensor(prior.n0, dtype=dtype, device=device)
        # posterior() accumulates Psi = (n0 - 2) * scale + scatter.
        psi_q = (n_q - 2.0) * self.scale
        psi_p = (n_p - 2.0) * torch.as_tensor(prior.scale, dtype=dtype, device=device)
        return _inverse_wishart_kl(n_q + d - 1, psi_q, n_p + d - 1, psi_p)

    def set_scale(self, scale):
        if type(scale) is torch.Tensor and type(self.scale) is torch.Tensor:
            if self.scale.is_cuda:
                self.scale = scale.cuda()
            else:
                self.scale = scale
        else:
            self.scale = torch.from_numpy(scale)

    def get_C(self):
        return self.C_fixed

    def get_mean(self):
        return self.C_fixed

    def cond_to_numpy(self, x):
        if x is not None:
            if type(x) is torch.Tensor:
                x = x.detach().numpy()
        return x

    def cond_to_torch(self, x):
        if x is not None:
            if type(x) is not torch.Tensor:
                x = torch.from_numpy(np.array(x))
        return x

    def to_numpy(self):
        self.scale = self.cond_to_numpy(self.scale)
        self.C_fixed = self.cond_to_numpy(self.C_fixed)

    def to_torch(self):
        self.scale = self.cond_to_torch(self.scale)
        self.C_fixed = self.cond_to_torch(self.C_fixed)

    def to_cuda(self):
        self.scale = self.scale.cuda()
        self.C_fixed = self.C_fixed.cuda()

    def to_cpu(self):
        self.scale = self.scale.cpu()
        self.C_fixed = self.C_fixed.cpu()
