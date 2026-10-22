"""Native forward/backward recurrences, independent of GP/LDS imports."""
import numpy as np
import torch
from ._independent_assignments import _iid_messages, _shifted_log_evidence


def _transition_weights(self, trans_A, n_states):
    """Exponentiate supplied expected log transitions with one common shift."""
    if not self.hmm_switch:
        weights = self.hdp.expected_log_iid_weights(n_states)
        weights = torch.exp(weights - weights.max())
        weights = weights.clamp_min(torch.finfo(weights.dtype).tiny)
        return weights.repeat(n_states, 1)
    if trans_A is None or not hasattr(trans_A, 'shape') or trans_A.shape != (n_states, n_states):
        trans_A = self.compute_trans_A(n_states)
    trans_A = self.cond_cuda(self.cond_to_torch(trans_A))
    weights = torch.exp(trans_A - torch.max(trans_A))
    return torch.where(torch.isfinite(trans_A),
                       weights.clamp_min(torch.finfo(weights.dtype).tiny),
                       torch.zeros_like(weights))



def _forward_log_fallback(q, start_logits, transition_weights):
    """Recompute one failed forward pass; ordinary passes never enter here."""
    evidence = _shifted_log_evidence(q)
    log_trans = torch.log(transition_weights.T)
    log_start = start_logits - torch.max(start_logits)
    log_alpha = torch.empty_like(q)
    scales = torch.empty(q.shape[0], device=q.device, dtype=q.dtype)
    for t in range(q.shape[0]):
        if t == 0:
            joint = log_start + evidence[t]
        else:
            joint = evidence[t] + torch.logsumexp(
                log_trans + log_alpha[t - 1][:, None], dim=0,
            )
        normalizer = torch.logsumexp(joint, dim=0)
        if not torch.isfinite(normalizer):
            raise FloatingPointError(
                f'Forward pass has no valid start/transition and emission path '
                f'at observation {t}'
            )
        log_alpha[t] = joint - normalizer
        scales[t] = torch.exp(normalizer)
    return torch.exp(log_alpha), scales



def _backward_log_fallback(q, transition_weights):
    """Recompute one failed backward pass in log space."""
    evidence = _shifted_log_evidence(q)
    log_trans = torch.log(transition_weights)
    log_beta = torch.zeros_like(q)
    for t in range(q.shape[0] - 2, -1, -1):
        joint = log_trans + evidence[t + 1][None, :] + log_beta[t + 1][None, :]
        row = torch.logsumexp(joint, dim=1)
        normalizer = torch.logsumexp(row, dim=0)
        if not torch.isfinite(normalizer):
            raise FloatingPointError(
                f'Backward pass has no valid transition and emission path '
                f'after observation {t}'
            )
        log_beta[t] = row - normalizer
    return torch.exp(log_beta)



def forward(self, pi=None, trans_A=None, q=None):
    """
    Forward-pass messages to compute alphas.

    Parameters
    ----------
    pi : array like of shape (m_samples) vector of initial distribution

    A : matrix like of shape (mxm_samples) transition matrix of the model

    q : matrix like of shape (txm_samples) variational observations of the example t

    Returns
    -------
    alphas : matrix of shape txm with alphas.

    """
    if pi is None and self.hmm_switch:
        pi = self.pi[-1]
    if trans_A is None:
        trans_A = self.trans_A
    if q is None:
        q = self.q[-1]
    if not self.hmm_switch:
        alpha, _, scales = _iid_messages(self, q)
        return alpha, scales
    M = self.M
    T = q.shape[0]
    K = q.shape[1]
    if not self.hmm_switch:
        pi = self.hdp.expected_log_iid_weights(K)
    fmsg = torch.zeros((T, K), device=self.device, dtype=getattr(self, 'message_dtype', None))
    margPrObs = torch.zeros(T, device=self.device, dtype=getattr(self, 'message_dtype', None))
    start_logits = (self.cond_cuda(self.cond_to_torch(pi)) if not self.hmm_switch
                    else self.cond_cuda(self.cond_to_torch(self.compute_trans_pi(K, pi))))
    pi_ = torch.exp(start_logits - torch.max(start_logits))

    def safe_exp(x):
        return torch.exp(x - torch.max(x, dim=1, keepdim=True).values)

    PiTMat = _transition_weights(self, trans_A, K).T
    q_ = q
    q_ = safe_exp(q_)
    if self.fmsg is not None:
        if self.fmsg.shape[1] < K:
            fmsg[:T - 1, :K - 1] = torch.clone(self.fmsg)
        else:
            fmsg[:T - 1, :K] = torch.clone(self.fmsg)
        margPrObs[:T - 1] = torch.clone(self.margPrObs)
        fmsg[-1] = torch.matmul(PiTMat, fmsg[-2]) * q_[-1]
        margPrObs[-1] = torch.sum(fmsg[-1])
        fmsg[-1] /= margPrObs[-1]
    else:
        # Reuse a view of this fresh message row, avoiding repeated indexing
        # and redundant indexed write-back. Keep the original autograd and
        # Tensor-subclass path; only ordinary nongradient tensors use views.
        reuse_rows = all(type(value) is torch.Tensor and not value.requires_grad
                         for value in (pi_, PiTMat, q_))
        for t in range(0, T):
            if reuse_rows:
                row = fmsg[t]
                if t == 0:
                    row.copy_(pi_ * q_[0])
                else:
                    row.copy_(torch.matmul(PiTMat, fmsg[t - 1]) * q_[t])
                margPrObs[t] = torch.sum(row)
            else:
                if t == 0:
                    fmsg[t] = pi_ * q_[0]
                else:
                    fmsg[t] = torch.matmul(PiTMat, fmsg[t - 1]) * q_[t]
                margPrObs[t] = torch.sum(fmsg[t])
            fmsg[t] /= margPrObs[t]
    if not torch.isfinite(fmsg).all():
        return _forward_log_fallback(q, start_logits, PiTMat)
    # if torch.argmax(fmsg[-1]) != torch.argmax(q[-1]) and self.verbose:
    #     print("Miss")
    return fmsg, margPrObs



def backward(self, trans_A=None, q=None, margprob=None):
    """
    Backward-pass messages to compute betas.

    Parameters
    ----------
    A : matrix like of shape (mxm_samples) transition matrix of the model

    q : matrix like of shape (txm_samples) variational observations of the example t

    Returns
    -------
    betas : matrix of shape txm with betas.

    """
    if trans_A is None:
        trans_A = self.trans_A
    if q is None:
        q = self.q[-1]
    if not self.hmm_switch:
        return torch.ones_like(q)
    M = q.shape[1]
    q_ = q

    def safe_exp(x):
        return torch.exp(x - torch.max(x, dim=1, keepdim=True).values)

    PiMat = _transition_weights(self, trans_A, M)
    q_ = safe_exp(q_)
    T = q.shape[0]
    K = M
    bmsg = torch.ones((T, K), device=self.device, dtype=getattr(self, 'message_dtype', None))
    reuse_rows = all(type(value) is torch.Tensor and not value.requires_grad
                     for value in (PiMat, q_))
    for t in range(T - 2, -1, -1):
        if reuse_rows:
            row = bmsg[t]
            row.copy_(torch.matmul(PiMat, bmsg[t + 1] * q_[t + 1]))
            row /= torch.sum(row)
        else:
            bmsg[t] = torch.matmul(PiMat, bmsg[t + 1] * q_[t + 1])
            bmsg[t] /= torch.sum(bmsg[t])
    if not torch.isfinite(bmsg).all():
        return _backward_log_fallback(q, PiMat)
    return bmsg



def coupled_state_coef(self, alpha=None, beta=None, trans_A=None, q=None, margprobs=None):
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
    if not self.hmm_switch:
        if alpha is None:
            alpha, _, _ = _iid_messages(self, self.q[-1] if q is None else q)
        pairs = alpha.new_zeros((len(alpha), alpha.shape[1], alpha.shape[1]))
        pairs[1:] = alpha[:-1, :, None] * alpha[1:, None, :]
        return torch.log(pairs)
    if alpha is None or margprobs is None:
        alpha, margprobs = self.forward()
    if beta is None:
        beta = self.backward()
    if trans_A is None:
        trans_A = self.trans_A
    if q is None:
        q = self.q[-1]

    def safe_exp(x):
        return torch.exp(x - torch.max(x, dim=1, keepdim=True).values)

    alpha = self.cond_cuda(self.cond_to_torch(alpha))
    beta = self.cond_cuda(self.cond_to_torch(beta))
    q = self.cond_cuda(self.cond_to_torch(q))
    PiMat = _transition_weights(self, trans_A, q.shape[1])
    bmsgSoftEv = safe_exp(q)
    bmsgSoftEv *= beta
    respPair = torch.zeros((q.shape[0], q.shape[1], q.shape[1]),
                           device=q.device, dtype=q.dtype)
    respPair[1:] = alpha[:-1][:, :, np.newaxis] *\
                   bmsgSoftEv[1:][:, np.newaxis, :]
    # respPair = alpha[:, :, np.newaxis] * \
    #               bmsgSoftEv[:, np.newaxis, :]
    respPair *= PiMat[np.newaxis, :, :]
    den = torch.sum(respPair, dim=(1, 2))
    # Row zero has no predecessor and is deliberately empty. Other zero rows
    # can arise when separately exponentiated message factors underflow.
    bad = torch.where((den[1:] <= 0) | ~torch.isfinite(den[1:]))[0] + 1
    if bad.numel():
        log_trans = torch.log(PiMat)
        for t in bad.tolist():
            log_pair = (torch.log(alpha[t - 1])[:, None] + log_trans
                        + q[t][None, :] + torch.log(beta[t])[None, :])
            normalizer = torch.logsumexp(log_pair.reshape(-1), dim=0)
            if not torch.isfinite(normalizer):
                raise FloatingPointError(
                    f'Pair message has no valid transition/emission path '
                    f'at observation {t}'
                )
            respPair[t] = torch.exp(log_pair - normalizer)
            den[t] = 1.0
    den[0] = 1.0
    respPair /= den[:, None, None]
    return torch.log(respPair)



