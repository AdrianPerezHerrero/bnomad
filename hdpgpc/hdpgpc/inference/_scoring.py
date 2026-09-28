"""Emission, LDS and HDP objective aggregation.

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


def compute_q_elbo(self, resp, respPair, q, q_lat, gpmodels, M, new_indexes=None, snr=None, post=False,
                   one_sample=False, verb=True):
    """ Method to compute ELBO terms.
    """
    if one_sample:
        n_points = 1
    else:
        n_points = self.x_basis[0].shape[0]
    # TODO: here is so simple to implement the non-clipped resposability assignation just by multiplying q by the resp term.
    q_bas = torch.sum(q[torch.where(resp == 1.0)]) * self.static_factor
    elbo_latent = torch.sum(q_lat[torch.where(resp == 1.0)]) * self.dynamic_factor
    if post:
        elbo_bas = self.elbo_Linears(resp, respPair, post=post,
                                     one_sample=one_sample) * n_points  # / self.n_outputs
    else:
        elbo_bas = self.elbo_Linears(resp, respPair, one_sample=one_sample) * n_points  # / self.n_outputs
    elbo_bas_LDS = 0
    if snr is None:
        frac = torch.ones(self.n_outputs, device=resp.device()) / self.n_outputs  # / self.M
    elif snr == 'saved':
        frac = torch.sum(self.snr_norm, dim=0)
        frac = frac / torch.sum(frac) * n_points  # * self.n_outputs# / self.M#
    else:
        frac = torch.sum(torch.softmax(torch.max(snr, dim=1)[0], dim=1), dim=0)
        frac = frac / torch.sum(frac) * n_points  # * self.n_outputs# * self.M#
    for i in range(self.n_outputs):
        elbo_bas_LDS = elbo_bas_LDS + self.full_LDS_elbo(gpmodels[i], torch.sum(resp, dim=0),
                                                         one_sample=one_sample) * frac[i]

    # elbo_bas = elbo_bas + elbo_latent
    # elbo_bas = 0
    if verb:
        print("Sum resp_temp: " + str(torch.sum(resp, dim=0).int().numpy()) + " - Total samples: " + str(
            torch.sum(resp).int().numpy()))
        print(
            f"Q_em: {q_bas.item():.2f}, Q_lat: {elbo_latent.item():.2f}, Elbo_linear: {elbo_bas:.2f}, Elbo_LDS: {elbo_bas_LDS.item():.2f}")
    if self.hmm_switch:
        elbo_bas = elbo_bas + elbo_bas_LDS + elbo_latent
    else:
        elbo_bas = elbo_latent
    return q_bas, elbo_bas


def full_LDS_elbo(self, gpmodels, sum_resp, one_sample=False):
    """ Method to accumulate LDS ELBO terms..
    """
    elb = torch.zeros(1)
    M_ = 0
    if one_sample:
        frac = sum_resp / torch.sum(sum_resp)
    else:
        frac = sum_resp / torch.sum(sum_resp)
    for i in sum_resp:
        if i > 0:
            M_ = M_ + 1
    gp_temp = gpmodels if one_sample else gpmodels
    for i, gp in enumerate(gp_temp):
        if sum_resp[i] > 0:
            if sum_resp[i] < self.free_deg_MNIV:
                # elb = elb + gp.return_LDS_param_likelihood(first=True)
                if one_sample:
                    elb = elb + gp.return_LDS_param_likelihood(first=False) * frac[i] * 1.0
                else:
                    elb = elb + gp.return_LDS_param_likelihood(first=False) * frac[i] * 1.0
            else:
                elb = elb + gp.return_LDS_param_likelihood() * frac[i]
    if one_sample:
        return elb  # / M_
    else:
        return elb / M_  # * np.min([np.max([1, M_ - 1]), self.M])


def estimate_new(self, t, gpmodel, x_train, y, h=1.0):
    """
    Auxiliary function, reestimate new variational parameters.

    """
    mean_, cov_, C_, Sigma_ = gpmodel.smoother_weighted(x_train, y, h)
    if len(gpmodel.indexes) == 1:
        q_new = gpmodel.log_sq_error(x_train, y, mean=mean_[-1], cov=cov_[-1], C=C_[-1], Sigma=Sigma_[-1],
                                     i=-1, first=True)
    else:
        q_new = gpmodel.log_sq_error(x_train, y, mean=mean_[-1], cov=cov_[-1], C=C_[-1], Sigma=Sigma_[-1],
                                     i=-1)  # , first=True)
    return q_new


def compute_q(self, x_train=None, y=None, gpmodels=None, mean=None, cov=None, C=None, Sigma=None, liks=None,
              batch=None, ld=None):
    """
    Compute variational parameters related with the observations. (Only for online scheme)
    Parameters
    ----------
    y : array sample of shape (s_samples)

    Returns
    -------
    q : array of shape m with q
    """

    M = self.M
    if x_train is None:
        x_train = self.x_train[-1]
    if y is None:
        y = self.y_w
    if isinstance(y, np.ndarray):
        T = len(y)
    else:
        T = len(y[0])
    if gpmodels is None:
        gpmodels = self.gpmodels
    if mean is None:
        mean = []
        for m in range(M):
            mean.append(gpmodels[m].f_star)
    if cov is None:
        cov = []
        for m in range(M):
            cov.append(gpmodels[m].cov_f)
    if C is None:
        C = []
        for m in range(M):
            C.append(gpmodels[m].C)
    if Sigma is None:
        Sigma = []
        for m in range(M):
            Sigma.append(gpmodels[m].Sigma)
    if type(y) is torch.Tensor:
        q = np.repeat(-np.inf, M)
        for m in range(M):
            q[m] = gpmodels[m].log_sq_error(x_train, y, mean[m][-1], cov[m][-1], C[m][-1], Sigma[m][-1], -1)
    else:
        if len(self.q) > 0:
            if ld is not None:
                q = torch.clone(self.q[-1][:, :, ld])
            else:
                q = torch.clone(self.q[-1])
            if len(q[-1]) < M:
                sh = q.shape
                q_ = torch.zeros((sh[0], sh[1] + 1))
                q_[:, :-1] = q
                q = q_
                for t in range(T - 1):
                    first = True if gpmodels[-1].N == 1 else False
                    q_ = gpmodels[-1].log_sq_error(x_train, y[-1][t], mean[-1][0], cov[-1][0],
                                                   C[-1][0], Sigma[-1][0], 0, first=first)
                    q[t, -1] = q_[0]
            q_ = torch.from_numpy(np.repeat(-np.inf, M))
            for m in range(M):
                first = True if gpmodels[m].N == 1 else False
                q_[m] = gpmodels[m].log_sq_error(x_train, y[m][-1], mean[m][-1], cov[m][-1],
                                                 C[m][-1], Sigma[m][-1], T - 1, first=first)
            q = torch.concatenate([q, q_[np.newaxis, :]])
        else:
            q = torch.zeros((self.T, M, self.n_outputs))
            q_ = torch.from_numpy(np.repeat(-np.inf, M))
            for m in range(M):
                first = True if gpmodels[m].N == 1 else False
                q_[m] = gpmodels[m].log_sq_error(x_train, y[m][-1], mean[m][-1], cov[m][-1],
                                                 C[m][-1], Sigma[m][-1], T - 1, first=first)
            q = q_

    return self.cond_to_torch(q)

