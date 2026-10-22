"""Boundary-aware reuse of the native HMM messages for MFA."""
from types import SimpleNamespace
import math

import torch

from . import _hmm_messages


def _log_messages(evidence, start, transition):
    """Rare underflow fallback, retaining log messages until marginals form."""
    alpha, beta = torch.empty_like(evidence), torch.zeros_like(evidence)
    scales = evidence.new_empty(len(evidence))
    for t in range(len(evidence)):
        joint = evidence[t] + (start if t == 0 else torch.logsumexp(
            alpha[t - 1][:, None] + transition, 0))
        scales[t] = torch.logsumexp(joint, 0)
        if not bool(torch.isfinite(scales[t])):
            raise FloatingPointError(f'HMM has no finite path at observation {t}')
        alpha[t] = joint - scales[t]
    for t in range(len(evidence) - 2, -1, -1):
        row = torch.logsumexp(transition + evidence[t + 1] + beta[t + 1], 1)
        beta[t] = row - torch.logsumexp(row, 0)
    resp = torch.softmax(alpha + beta, 1)
    pair = evidence.new_zeros((len(evidence), evidence.shape[1], evidence.shape[1]))
    for t in range(1, len(evidence)):
        joint = alpha[t - 1][:, None] + transition + evidence[t] + beta[t]
        pair[t] = torch.softmax(joint.reshape(-1), 0).reshape_as(transition)
    return resp, pair, scales


def markov_assignments(state, evidence, lengths, *, start=None, transition=None):
    """Return smoothed singleton/pair marginals and log partition increments.

    Pair row t represents (s[t-1],s[t]); every sequence start is all zeros.
    Log partition increments sum to log Z for the supplied local potentials.
    """
    k, n = state.M, len(evidence)
    if start is None:
        start = torch.digamma(state.startTheta[:k]) - torch.digamma(state.startTheta.sum())
    if transition is None:
        transition = (torch.digamma(state.transTheta[:, :k])
                      - torch.digamma(state.transTheta.sum(1))[:, None])
    resp = torch.empty_like(evidence)
    pair = evidence.new_zeros((n, k, k))
    increments = evidence.new_empty(n)
    # Fresh context per call prevents accidental reuse of cached online messages.
    context = SimpleNamespace(M=k, hmm_switch=True, device=evidence.device,
        message_dtype=evidence.dtype, fmsg=None, cond_cuda=lambda x: x.to(evidence.device),
        cond_to_torch=lambda x: torch.as_tensor(x, dtype=evidence.dtype, device=evidence.device),
        compute_trans_pi=lambda count, pi: pi)
    offset = 0
    for length in lengths:
        if length == 0:
            continue
        end = offset + length
        q = evidence[offset:end]
        try:
            # The legacy kernel clamps underflowed transition weights to tiny.
            # Before that changes a path probability, use exact log messages.
            minimum_log = math.log(torch.finfo(q.dtype).tiny)
            if (bool(((transition - transition.max()) < minimum_log).any())
                    or bool(((start - start.max()) < minimum_log).any())
                    or bool(((q - q.max(1, keepdim=True).values) < minimum_log).any())):
                raise FloatingPointError('Extreme local HMM potentials')
            alpha, scales = _hmm_messages.forward(context, start, transition, q)
            beta = _hmm_messages.backward(context, transition, q, scales)
            product = alpha * beta
            r = product / product.sum(1, keepdim=True)
            p = _hmm_messages.coupled_state_coef(context, alpha, beta, transition, q, scales).exp()
            log_scales = scales.log() + q.max(1).values
            log_scales[0] += start.max()
            log_scales[1:] += transition.max()
            # Underflow can leave separately normalized messages disjoint, or
            # native tiny-weight clipping can corrupt pair marginal consistency.
            if (not bool(torch.isfinite(r).all() and torch.isfinite(log_scales).all())
                    or not torch.allclose(p[1:].sum(1), r[1:], atol=1e-8, rtol=1e-7)
                    or not torch.allclose(p[1:].sum(2), r[:-1], atol=1e-8, rtol=1e-7)):
                raise FloatingPointError('Underflow in native HMM messages')
        except FloatingPointError:
            r, p, log_scales = _log_messages(q, start, transition)
        resp[offset:end], pair[offset:end], increments[offset:end] = r, p, log_scales
        offset = end
    return resp, pair, increments


def initial_pairs(resp, lengths):
    pair = resp.new_zeros((len(resp), resp.shape[1], resp.shape[1]))
    offset = 0
    for length in lengths:
        end = offset + length
        pair[offset + 1:end] = resp[offset:end - 1, :, None] * resp[offset + 1:end, None, :]
        offset = end
    return pair


def chain_entropy(resp, pair, lengths):
    value = resp.new_zeros(())
    offset = 0
    for length in lengths:
        end = offset + length
        value -= torch.special.xlogy(resp[offset], resp[offset]).sum()
        p = pair[offset + 1:end]
        # Conditional entropy H(s_t | s_{t-1}); xlogy(0,0) = 0.
        value -= torch.special.xlogy(p, p).sum()
        previous = p.sum(2)
        value += torch.special.xlogy(previous, previous).sum()
        offset = end
    return value
