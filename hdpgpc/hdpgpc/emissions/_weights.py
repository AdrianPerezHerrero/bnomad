"""Output weighting, SNR and joint emission scores.

Functions operate on the explicitly supplied mutable inference state.
Numerical expressions and mutation order are preserved from the executable specification.
"""

import numpy as np
import torch
import pandas as pd
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from scipy.special._ufuncs import psi as digamma, gammaln
from torchmetrics.audio import SignalNoiseRatio
from tqdm import trange
import pickle as plk
from ..emissions import _gaussian_process as gp
from ..alignment import _components as _alignment
from ..hdp._variational import HDPVariational
from ..hdp._optimizer import find_optimum_multiple_tries, kvec
from ..numerics._normalization import max_shift_log_scores
from ..hdp._entropy import state_entropy_terms, pair_entropy_terms


def signaltonoise(self, a, axis=0, ddof=0):
    """ Compute signal to noise ratio
    """
    a = np.asanyarray(a)
    # m = a.mean(axis)** 2
    m = 100.0
    # m = a.max(axis)[0] ** 2
    sd = a.std(axis=axis, ddof=ddof) ** 2
    return np.where(sd == 0, 0, m / sd)


def rolling_snr(self, signal, window_size: int):
    """ Compute rolled signal to noise ratio.
    """
    signal_series = pd.Series(signal)
    rolling_mean = signal_series.rolling(window=window_size).mean()[window_size:].mean()
    #rolling_max = signal_series.rolling(window=window_size).max().max() ** 2
    #rolling_min = signal_series.rolling(window=window_size).min().min() ** 2
    rolling_std = signal_series.rolling(window=window_size).std()[window_size:].mean()
    rolling_snr = 10 * np.log10(
        (rolling_mean ** 2) / (rolling_std ** 2))  # .replace(0, np.finfo(float).eps))  # type: ignore
    return rolling_snr


def weight_mean(self, q, snr=None):
    """ Weight the log squared error with the snr computed to join the information on each lead.
    """
    if len(q.shape) > 2:
        if snr is None:
            return torch.einsum('ijk,ik->ij', q, self.snr_norm)
        else:
            snr_ = torch.softmax(torch.max(snr, dim=1)[0], dim=1)
            return torch.einsum('ijk,ik->ij', q, snr_)
    else:
        if snr is None:
            snr_frac = torch.sum(self.snr_norm, dim=0) / torch.sum(self.snr_norm)
            return torch.einsum('ij,j->i', q, snr_frac)
        else:
            snr_ = torch.softmax(torch.max(snr, dim=1)[0], dim=1)
            snr_frac = torch.sum(snr_, dim=0) / torch.sum(snr_)
            return torch.einsum('ij,j->i', q, snr_frac)


def reduce_num_outputs(self, y_trains):
    ratio = self.reduce_outputs_ratio
    num_final_outputs = int(np.rint(y_trains.shape[2] * ratio))
    var = np.var(np.sum(y_trains, axis=1), axis=0)
    final_outputs = np.sort(var.argsort()[::-1][:num_final_outputs])
    if self.verbose:
        print("Performed reduction of outputs based on variance.")
        print("Ratio of reduction: " + str(ratio) + " Final outputs: " + str(final_outputs))
    self.n_outputs = num_final_outputs
    self.wp_sys = [self.wp_sys[ld] for ld in final_outputs]
    self.gpmodels = [self.gpmodels[ld] for ld in final_outputs]
    return y_trains[:, :, final_outputs]


def compute_snr_ini(self, y_trains):
    """ Initial computation of snr
    """
    if self.use_snr:
        n_samples = np.array(y_trains).shape[0]
        n_outputs = np.array(y_trains).shape[2]
        snr = torch.zeros(n_samples, n_outputs)
        snr_comp = SignalNoiseRatio()
        for ld in range(n_outputs):
            snr[:, ld] = torch.tensor(
                [snr_comp(torch.from_numpy(y),
                          torch.mean(torch.from_numpy(y_trains)[:, :, ld], dim=0)) for y in
                 y_trains[:, :, ld]])
        self.snr_norm = torch.softmax(snr, dim=1)
    else:
        self.snr_norm = torch.ones(y_trains.shape[0], y_trains.shape[2])


def compute_snr(self, y_trains, gp):
    """ Iterative computation of snr
    """
    if self.use_snr:
        snr = torch.zeros(y_trains.shape[0])
        snr_comp = SignalNoiseRatio()
        for t in range(y_trains.shape[0]):
            j = np.min([np.max([gp.find_closest_lower(t), 1]), len(gp.f_star_sm) - 1])
            if self.inducing_points:
                snr[t] = snr_comp(y_trains[t], gp.resample_latent_mean(self.x_train[-1], t=j)[0].T[0])
            else:
                snr[t] = snr_comp(y_trains[t], gp.f_star_sm[j].T[0])
        # snr = torch.tensor([0.5] * y_trains.shape[0])
        return snr
    else:
        snr = torch.ones(y_trains.shape[0])
        return snr


def normalize_snr(self, snr):
    """ Normalize snr using softmax
    """
    if self.use_snr:
        return torch.softmax(torch.max(torch.clone(snr), dim=1)[0], dim=1)
    else:
        return torch.softmax(torch.max(torch.clone(snr), dim=1)[0], dim=1)


def compute_joint_xy_q(self, x_trains, y_trains, gpmodels, outputs=(0, 1), jitter=1e-6):
    ld_x, ld_y = outputs
    n_samples = y_trains.shape[0]
    M = len(gpmodels[ld_x])
    q_joint = torch.zeros((n_samples, M), dtype=torch.float64, device=self.device)

    for m in range(M):
        gp_x = gpmodels[ld_x][m]
        gp_y = gpmodels[ld_y][m]

        for i in range(n_samples):
            x_i = x_trains[i]
            yx = y_trains[i, :, [ld_x]]
            yy = y_trains[i, :, [ld_y]]

            mean_x, cov_x = gp_x.observe_last(x_i)
            mean_y, cov_y = gp_y.observe_last(x_i)

            mean_x = mean_x.reshape(-1)
            mean_y = mean_y.reshape(-1)

            rx = yx.reshape(-1) - mean_x
            ry = yy.reshape(-1) - mean_y
            r = torch.cat([rx, ry], dim=0)

            sx = torch.sqrt(torch.clamp(torch.diag(cov_x), min=jitter))
            sy = torch.sqrt(torch.clamp(torch.diag(cov_y), min=jitter))

            rho = torch.tanh(self.rho_xy[m])  # one scalar per cluster
            cross = rho * torch.diag(sx * sy)

            Sigma = torch.block_diag(cov_x, cov_y)
            L = cov_x.shape[0]
            Sigma[:L, L:] = cross
            Sigma[L:, :L] = cross.T
            Sigma = Sigma + jitter * torch.eye(2 * L, device=self.device)

            chol = torch.linalg.cholesky(Sigma)
            alpha = torch.cholesky_solve(r[:, None], chol).squeeze(-1)
            logdet = 2.0 * torch.sum(torch.log(torch.diag(chol)))

            q_joint[i, m] = -0.5 * (
                    torch.dot(r, alpha) + logdet + (2 * L) * np.log(2.0 * np.pi)
            )

    return q_joint

