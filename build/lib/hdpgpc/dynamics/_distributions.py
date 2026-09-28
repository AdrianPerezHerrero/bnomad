"""Conjugate distributions for LDS parameter updates; native pickle paths."""

import numpy as np
import torch


class matrix_normal_inv_wishart():
    """ Class to define the Matrix Normal Inverse Wishart distribution (MNIW) for the LDS parameters.
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
        y_k = y1
        y_k_ = y2
        new_n0 = self.n0 + n_k

        device = self.scale.device
        dtype = self.scale.dtype
        d = self.scale.shape[0]
        id_d = torch.eye(d, device=device, dtype=dtype)

        if sse_matrix is None:
            sse_matrix = id_d

        scale = 0.5 * (self.m_r_cov + self.m_r_cov.T)
        jitter = 1e-2 * torch.mean(torch.diag(self.scale).abs()).clamp_min(torch.finfo(dtype).eps)
        L_scale = torch.linalg.cholesky(scale + jitter * id_d)
        scale_inv = torch.cholesky_solve(id_d, L_scale)

        y2p = sse_matrix @ y_k_
        y1p = sse_matrix @ y_k

        exp_f_f_ = y2p @ y2p.T + sse_matrix @ cov_ @ sse_matrix.T
        exp_ff_ = y1p @ y2p.T + sse_matrix @ cov_cross @ sse_matrix.T
        #exp_ff = y1p @ y1p.T + sse_matrix @ cov @ sse_matrix.T

        S__ = exp_f_f_ + scale_inv
        S_ = exp_ff_ + self.m_mean @ scale_inv
        #S = exp_ff + self.m_mean @ scale_inv @ self.m_mean.T

        L_S__ = torch.linalg.cholesky(0.5 * (S__ + S__.T) + 1e-8 * id_d)
        part_mean = torch.cholesky_solve(S_.T, L_S__).T

        if n_k == 1:
            new_m_mean = ((self.n0 - 2) * self.m_mean + part_mean) / (new_n0 - 2)
            e = y1p - y2p
            e2 = e @ e.T
            new_scale = ((self.n0 - 2) * self.scale + e2) / (new_n0 - 2)
        else:
            new_m_mean = part_mean
            e = y_k - new_m_mean @ y_k_
            e2 = e @ e.T
            new_scale = ((self.n0 - 2) * self.scale + e2) / (new_n0 - 2)

        new_m_r_cov = S__
        return matrix_normal_inv_wishart(new_m_mean, new_m_r_cov, new_n0, new_scale)

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

    def get_mean(self):
        return self.m_mean

    def get_scale(self, final=False):
        if final:
            return self.scale
        else:
            return self.scale * self.n0 / (self.n0 - 2)


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

    def get_scale(self, final=False):
        # return self.scale / (self.n0 - self.scale.shape[0] - 1)
        return self.scale * self.n0 / (self.n0 - 2)

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
