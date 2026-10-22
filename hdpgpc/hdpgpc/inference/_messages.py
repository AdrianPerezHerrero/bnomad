"""HMM responsibilities and forward/backward message recurrences.

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
from . import _hmm_messages
from ._hmm_messages import _transition_weights, _forward_log_fallback, _backward_log_fallback
from ._independent_assignments import _iid_messages, _shifted_log_evidence


def _safe_exp(self, x, hard=None):
    """Convert log marginals to hard or normalized soft responsibilities."""
    if hard is None:
        hard = getattr(self, 'hard_assignments', True)
    if not hard:
        if x.dim() not in (2, 3):
            raise ValueError('Expected state or transition log marginals')
        flat = x.reshape(x.shape[0], -1)
        finite = torch.isfinite(flat)
        valid = finite.any(dim=1, keepdim=True)
        scores = torch.where(finite, flat, torch.full_like(flat, -torch.inf))
        scores = torch.where(valid, scores, torch.zeros_like(scores))
        probabilities = torch.softmax(scores, dim=1)
        return torch.where(valid, probabilities, torch.zeros_like(probabilities)).reshape_as(x)
    if x.dim() == 2:
        idx = x.argmax(dim=-1, keepdim=True)
        y = torch.zeros_like(x)
        y.scatter_(-1, idx, 1.0)
        return y
    if x.dim() == 3:
        N = x.shape[0]
        x_flat = x.reshape(N, -1)  # (N, K*K)
        idx = x_flat.argmax(dim=-1, keepdim=True)  # (N,1)
        y_flat = torch.zeros_like(x_flat, dtype=torch.float32)
        y_flat.scatter_(1, idx, 1.0)
        y_flat[~torch.isfinite(x_flat).any(dim=1)] = 0.0
        return y_flat.reshape_as(x)


def assignment_messages(self, start, transitions, evidence):
    """Keep the native HMM path; dispatch independent assignments directly."""
    if not self.hmm_switch:
        return _iid_messages(self, evidence)
    alpha, scales = self.forward(start, transitions, evidence)
    beta = self.backward(transitions, evidence, scales)
    return alpha, beta, scales


def variational_local_terms(self, q, transTheta=None, startTheta=None, liks=None,
                            classify=False, hard_assignments=None):
    """
    Compute variational terms of the SLDS for one example. Online approach.

    Parameters
    ----------
    y : vector double
        observation vector (or warped versions).
    liks : vector, optional
        likelihoods of the warpings.
    hard_assignments : bool, optional
        Override the configured choice of one-hot or normalized marginals.
    """
    # Parameters of model
    M = self.M
    q = torch.clone(q)
    # Compute the initial values
    if transTheta is None:
        transTheta = self.transTheta
    if startTheta is None:
        startTheta = self.startTheta
    if liks is None:
        liks = np.zeros(q.shape[1])
    if self.hmm_switch:
        transPi = self.hdp.expected_log_transitions(M, transTheta)
        startPi = self.hdp.expected_log_start(M, startTheta)
    else:
        transPi = self.hdp.iid_log_transitions(q.shape[1])
        startPi = self.hdp.expected_log_iid_weights(q.shape[1])
    self.trans_A = transPi
    liks_ = np.array(liks)[:, np.newaxis]  # * 0.5
    q[-1] = q[-1] + liks_
    if classify:
        for q_aux in q:
            q_aux[-1] = -np.inf
    m_ = np.argmax(self.weight_mean(q)[-1])
    q_, _ = self.LogLik(self.weight_mean(q))
    alpha, beta, margprob = assignment_messages(self, startPi, transPi, q_)
    resp, _ = self.LogLik(torch.log(alpha * beta), axis=1)
    m_resp = np.argmax(self.cond_cpu(resp[-1]))
    respPair, _ = self.LogLik(self.coupled_state_coef(alpha, beta, transPi, q_, margprob), axis=1)
    if m_ != m_resp and self.verbose:
        print("Mismatch between SSE (" + str(m_ + 1) + ") and Resp (" + str(m_resp + 1) + ").")
    if self.verbose and torch.any(torch.isnan(resp)):
        print("Error")
    assignments = self._safe_exp(resp, hard=hard_assignments)
    pair_assignments = self._safe_exp(respPair, hard=hard_assignments)
    if classify:
        return q, assignments, resp, pair_assignments, respPair
    else:
        return assignments, resp, pair_assignments, respPair


def LogLik(self, logSoftEv, axis=1):
    ''' Return element-wise normalized input log likelihood

    Numerically safe, guaranteed not to underflow

    Returns
    --------
    SoftEv : 2D array, size TxK
              equal to logSoftEv, up to prop constant for each row
    lognormC : 1D array, size T
              gives log of the prop constant for each row
    '''
    if torch.is_tensor(logSoftEv) and logSoftEv.ndim == 3 and axis == 1:
        # Pair marginals must retain relative scale across both state axes.
        maxima = torch.amax(logSoftEv, dim=(1, 2))
        shifts = torch.where(torch.isfinite(maxima), maxima, torch.zeros_like(maxima))
        return logSoftEv - shifts[:, None, None], maxima
    return max_shift_log_scores(logSoftEv, axis)


def forward(self, pi=None, trans_A=None, q=None):
    """Delegate to the reusable native HMM recurrence."""
    return _hmm_messages.forward(self, pi, trans_A, q)


def backward(self, trans_A=None, q=None, margprob=None):
    """Delegate to the reusable native HMM recurrence."""
    return _hmm_messages.backward(self, trans_A, q, margprob)


def coupled_state_coef(self, alpha=None, beta=None, trans_A=None, q=None, margprobs=None):
    """Delegate to the reusable native HMM recurrence."""
    return _hmm_messages.coupled_state_coef(self, alpha, beta, trans_A, q, margprobs)


def sum_coupled_state_coef(self, alpha=None, beta=None, trans_A=None, q=None):
    """
    Compute psi coefficients for ease reestimation of parameters on state-model.

    Parameters
    ----------
    alpha : matrix of shape txm with alphas, optional
         The default is compute alphas.
    beta : matrix of shape txm with betas, optional
         The default is compute betas.
    A : matrix like of shape (mxm_samples) transition matrix of the model
        The default is last saved.
    q : matrix like of shape (txm_samples) variational observations of the example t
        The default is last saved.
    Returns
    -------
    psi : array of matrix of shape txmxm
        Psi coefficients.

    """
    if alpha is None:
        alpha = self.forward()
    if beta is None:
        beta = self.backward()
    if trans_A is None:
        trans_A = self.trans_A
    if q is None:
        q = self.q[-1]

    alpha = self.cond_cuda(self.cond_to_torch(alpha))
    beta = self.cond_cuda(self.cond_to_torch(beta))
    trans_A = self.cond_cuda(self.cond_to_torch(trans_A))
    q = self.cond_cuda(self.cond_to_torch(q))

    T = alpha.shape[0]
    M = self.M + 1
    psi = self.cond_cuda(self.cond_to_torch(np.matrix([[-np.inf] * M] * M)))
    for i in range(M - 1):
        for j in range(M):
            psi[i, j] = torch.sum(alpha[:, i]) + trans_A[i, j] + torch.sum(q[:, j]) + torch.sum(beta[:, j])  # - den
        den = torch.logsumexp(psi.view(-1), 0)
        psi = psi - den
    return psi


def compute_h(self, alpha=None, beta=None, time=None):
    """
    Compute variational parameters related with relevance.

    Parameters
    ----------
    alpha : array os shpae m with alphas (forward msg), optional
        The default is compute alphas.
    beta : array of shape m with betas (forward msg), optional
        The default is compute betas.
    t : int time where is wanted to compute h_t(i) for all states
        The default is last.
    Returns
    -------
    h_ : array of shape m with h

    """
    T = alpha.shape[0]
    M = self.M + 1
    if alpha is None:
        alpha = self.forward()
    if beta is None:
        beta = self.backward()
    alpha = self.cond_cuda(self.cond_to_torch(alpha))
    beta = self.cond_cuda(self.cond_to_torch(beta))
    h = self.cond_cuda(self.cond_to_torch(np.matrix([[-np.inf] * M] * T)))
    for t in range(T):
        den = self.cond_cuda(self.cond_to_torch(-np.inf))
        for i in range(M):
            den = torch.logaddexp(den, alpha[t, i] + beta[t, i])
        h_ = self.cond_cuda(self.cond_to_torch(np.array([-np.inf] * M)))
        for i in range(M):
            h_[i] = alpha[t, i] + beta[t, i] - den
        h[t] = h_

    if time is None:
        return self.cond_cuda(self.cond_to_torch(h))
    else:
        return self.cond_cuda(self.cond_to_torch(h[time]))


def baum_welch(self, alpha=None, beta=None, trans_A=None, q=None):
    """
    Re-estimate parameters of HMM on terms of baum-welch algorithm

    Parameters
    ----------
    alpha : matrix of shape txm with alphas, optional
         The default is compute alphas.
    beta : matrix of shape txm with betas, optional
         The default is compute betas.
    A : matrix like of shape (mxm_samples) transition matrix of the model
        The default is last saved.
    q : matrix like of shape (txm_samples) variational observations of the example t
        The default is last saved.

    Returns
    -------
    pi : array of size m, initial distribution vector.

    trans_A : matrix if size mxm, transition matrix


    """
    if self.hmm_switch:
        M = self.M
        T = self.T

        if alpha is None:
            alpha = self.forward()
        if beta is None:
            beta = self.backward()
        if trans_A is None:
            trans_A = self.trans_A
        if q is None:
            q = self.q[-1]

        alpha = self.cond_to_torch(alpha)
        beta = self.cond_to_torch(beta)
        trans_A = self.cond_to_torch(trans_A)
        q = self.cond_to_torch(q)

        h = self.cond_to_torch(self.compute_h(alpha, beta))
        pi_ = self.cond_to_torch(np.ravel(h[0]))
        trans_A_ = self.cond_to_torch(np.matrix([[0.0] * M] * M))
        # Compute numerator and denominator using specific functions as described in Rabiner
        psi = self.cond_to_torch(self.coupled_state_coef(alpha, beta, trans_A, q))
        for i in range(M):
            den = self.cond_to_torch(-np.inf)
            for t in range(T - 1):
                for j in range(M):
                    den = torch.logaddexp(den, psi[t][i, j])
            for j in range(M):
                num = self.cond_to_torch(-np.inf)
                for t in range(T - 1):
                    num = torch.logaddexp(num, psi[t][i, j])
                # Case where zero expected times of transitions are detected
                if num == -np.inf:
                    trans_A_[i, j] = -np.inf
                else:
                    trans_A_[i, j] = num - den
            # To maintain properties of the transition matrix (sum of row = 1)
            aux = self.normalize_log(np.ravel(trans_A_[i]))
            for j in range(M):
                trans_A_[i, j] = aux[j]

        return pi_, trans_A_
    else:
        return self.pi[-1], self.trans_A[-1]


def normalize_log(self, x):
    # Special case when maximum bound has been reached
    bound = 1e-50
    log_bound = np.log(bound)
    if np.max(x) == -np.inf:
        x = np.repeat(log_bound, len(x))
    if not np.isclose(np.max(x), 0):
        aux = np.abs(x) / np.max(np.abs(x))
        aux = -aux + np.ones(len(aux))
        aux = [bound if i == 0 else i for i in aux]
        # aux = np.exp(aux)
        den = np.sum(aux)
        x = np.log(aux / den)
    else:
        ind = np.argmax(x)
        x = np.repeat(log_bound, len(x))
        x[ind] = 0.0
    return x

