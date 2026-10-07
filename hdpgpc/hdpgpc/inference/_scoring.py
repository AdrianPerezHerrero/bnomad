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


def _lds_point_dimension(self, gpmodels):
    """Dimension of the latent LDS matrices used in this score."""
    for output in gpmodels:
        if output:
            return int(output[0].A[-1].shape[0])
    return len(self.x_basis_ini)


def compute_q_elbo(self, resp, respPair, q, q_lat, gpmodels, M, new_indexes=None, snr=None, post=False,
                   one_sample=False, verb=None, return_components=False, birth_score=True):
    """Compute ELBO terms, optionally tempering the LDS KL for birth scoring.

    ``birth_score=False`` always returns the unmodified ELBO. The optional
    score divides the LDS KL by the LDS state dimension in every schedule.
    This is an experiment, not an evidence lower bound.
    """
    if verb is None:
        verb = self.verbose
    if self.hard_assignments:
        q_bas = torch.sum(q[torch.where(resp == 1.0)]) * self.static_factor
        elbo_latent = torch.sum(q_lat[torch.where(resp == 1.0)]) * self.dynamic_factor
    else:
        weights = resp.to(q.dtype)
        q_bas = torch.where(weights > 0, weights * q, torch.zeros_like(q)).sum() * self.static_factor
        elbo_latent = torch.where(weights > 0, weights * q_lat,
                                  torch.zeros_like(q_lat)).sum() * self.dynamic_factor
    if post:
        elbo_bas = self.elbo_Linears(resp, respPair, post=post,
                                     one_sample=one_sample)
    else:
        elbo_bas = self.elbo_Linears(resp, respPair, one_sample=one_sample)
    # Final scoring counts the final parameter KL once per cluster/output.
    # Historical scoring uses that cluster's mean over scored beat states.
    elbo_bas_LDS = torch.zeros((), dtype=q.dtype, device=q.device)
    cluster_counts = torch.sum(resp, dim=0)
    for output in range(self.n_outputs):
        elbo_bas_LDS = elbo_bas_LDS + self.full_LDS_elbo(
            gpmodels[output], cluster_counts, one_sample=one_sample)
    raw_lds = elbo_bas_LDS
    adjusted_lds = birth_score and self.birth_lds_kl_per_sqrt_point
    if adjusted_lds:
        dimension = _lds_point_dimension(self, gpmodels)
        divisor = dimension
        divisor_name = str(dimension)
        elbo_bas_LDS = raw_lds / divisor

    chain_entropy = torch.zeros_like(q_bas)
    if one_sample and not self.hard_assignments:
        chain_entropy = torch.as_tensor(
            self.calcELBO_NonlinearTerms(
                self.cond_to_numpy(self.cond_cpu(resp)),
                self.cond_to_numpy(self.cond_cpu(respPair))),
            dtype=q_bas.dtype, device=q_bas.device)

    components = None
    if return_components:
        components = {
            'emission': q_bas,
            'latent': elbo_latent,
            'hdp': torch.as_tensor(elbo_bas) if self.hmm_switch else torch.zeros_like(q_bas),
            'lds': elbo_bas_LDS,
            'lds_raw': raw_lds,
            'chain_entropy': chain_entropy,
        }

    # elbo_bas = elbo_bas + elbo_latent
    # elbo_bas = 0
    if verb:
        print("Sum resp_temp: " + str(torch.sum(resp, dim=0).int().numpy()) + " - Total samples: " + str(
            torch.sum(resp).int().numpy()))
        lds_label = (f"BirthScore_LDS: {elbo_bas_LDS.item():.2f} "
                     f"(raw {raw_lds.item():.2f} / {divisor_name}; divisor={divisor:.6g})"
                     if adjusted_lds else
                     f"Elbo_LDS: {elbo_bas_LDS.item():.2f}")
        print(f"Q_em: {q_bas.item():.2f}, Q_lat: {elbo_latent.item():.2f}, "
              f"Elbo_linear: {elbo_bas:.2f}, {lds_label}")
    if self.hmm_switch:
        elbo_bas = elbo_bas + elbo_bas_LDS + elbo_latent
    else:
        elbo_bas = elbo_bas_LDS + elbo_latent
    elbo_bas = elbo_bas + chain_entropy
    if return_components:
        return q_bas, elbo_bas, components
    return q_bas, elbo_bas


def full_LDS_elbo(self, gpmodels, sum_resp, one_sample=False):
    """Sum cluster LDS terms and count a shared observation KL once."""
    elb = sum_resp.new_zeros(())
    for index, gp in enumerate(gpmodels):
        if sum_resp[index] > 0:
            elb = elb + gp.return_LDS_param_likelihood()
    if gpmodels and torch.any(sum_resp > 0):
        policy = getattr(gpmodels[0], 'observation_policy', None)
        if policy is not None and policy.scope == 'shared':
            elb = elb - policy.kl()
    return elb


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

