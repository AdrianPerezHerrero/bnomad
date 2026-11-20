"""Banded ordered-chain messages and joint MAP decoding."""
from collections import OrderedDict
import weakref

import torch
from ._assignments import from_labels


class PathCache:
    """Fit-local decoded paths; weak references never retain pair histories.

    Tensor versions detect in-place writes, including writes through views.
    Inference-mode tensors lack versions and are deliberately not cached.
    """
    def __init__(self, maxsize=16):
        self.maxsize = maxsize
        self._entries = OrderedDict()

    def get(self, resp, pair):
        try:
            versions = (resp._version, pair._version)
        except (AttributeError, RuntimeError):
            return _posterior_path(resp, pair)
        key = (id(resp), id(pair))
        entry = self._entries.get(key)
        if entry is not None and entry[0]() is resp and entry[1]() is pair and entry[2] == versions:
            self._entries.move_to_end(key)
            return entry[3]
        path = _posterior_path(resp, pair)
        self._entries[key] = (weakref.ref(resp), weakref.ref(pair), versions, path)
        self._entries.move_to_end(key)
        while len(self._entries) > self.maxsize:
            self._entries.popitem(last=False)
        return path


def decode(log_initial, log_transitions, unary):
    """Viterbi path; row marginals must never be decoded independently."""
    length, count = unary.shape
    value = log_initial+unary[0]
    pointers = torch.zeros((length, count), dtype=torch.long, device=unary.device)
    diagonal = log_transitions.diag()
    advance = log_transitions.diag(1)
    for t in range(1, length):
        stay = value+diagonal
        step = torch.cat((value.new_full((1,), -torch.inf), value[:-1]+advance))
        moved = step > stay
        pointers[t] = torch.arange(count, device=unary.device)-moved.long()
        value = torch.maximum(stay, step)+unary[t]
        if bool(torch.isfinite(value).any()):
            value = value-value.max()
    if not bool(torch.isfinite(value).any()):
        raise FloatingPointError('No feasible ordered state path')
    path = pointers.new_empty(length)
    path[-1] = value.argmax()
    for t in range(length-1, 0, -1):
        path[t-1] = pointers[t, path[t]]
    return path


def posterior_path(resp, pair, *, cache=None):
    """Decode the MAP joint path from consistent chain state/pair marginals."""
    return _posterior_path(resp, pair) if cache is None else cache.get(resp, pair)


def _posterior_path(resp, pair):
    length, count = resp.shape
    if length == 1:
        return torch.zeros(1, dtype=torch.long, device=resp.device)
    # A time-inhomogeneous chain has q(s_t|s_{t-1})=xi/gamma.
    value = torch.log(resp[0])
    pointers = torch.zeros((length, count), dtype=torch.long, device=resp.device)
    diagonal = pair.diagonal(dim1=1, dim2=2)
    advance = pair.diagonal(offset=1, dim1=1, dim2=2)
    # One storage scan preserves the old handling of malformed/nonordered input;
    # the valid ordered dynamic program only visits its two predecessor bands.
    banded = (bool(torch.isfinite(resp).all()) and not bool((resp < 0).any())
              and int(torch.count_nonzero(pair[1:])) ==
              int(torch.count_nonzero(diagonal[1:]) + torch.count_nonzero(advance[1:])))
    index = torch.arange(count, device=resp.device)
    if banded:
        log_previous = torch.log(resp[:-1])
        stay_terms = torch.log(diagonal[1:])-log_previous
        step_terms = torch.log(advance[1:])-log_previous[:, :-1]
        stay_terms = torch.where(torch.isfinite(stay_terms), stay_terms, -torch.inf)
        step_terms = torch.where(torch.isfinite(step_terms), step_terms, -torch.inf)
        impossible = value.new_full((1,), -torch.inf)
    for t in range(1, length):
        if banded:
            stay = value+stay_terms[t-1]
            step = torch.cat((impossible, value[:-1]+step_terms[t-1]))
            # Dense max chooses the lowest source index: advance wins ties.
            moved = (step >= stay) & (index > 0)
            value = torch.maximum(stay, step)
            pointers[t] = torch.where(torch.isneginf(value), 0, index-moved.long())
        else:
            log_pair = torch.log(pair[t])
            conditional = log_pair-torch.log(resp[t-1])[:, None]
            conditional = torch.where(torch.isfinite(conditional), conditional,
                                      torch.full_like(conditional, -torch.inf))
            score, source = (value[:, None]+conditional).max(0)
            pointers[t], value = source, score
    if not bool(torch.isfinite(value).any()):
        raise FloatingPointError('Ordered responsibilities contain no feasible joint path')
    path = pointers.new_empty(length)
    path[-1] = value.argmax()
    for t in range(length-1, 0, -1):
        path[t-1] = pointers[t, path[t]]
    if path[0] != 0 or bool(((path[1:]-path[:-1] < 0) | (path[1:]-path[:-1] > 1)).any()):
        raise ValueError('Assignments violate left-to-right topology')
    return path


def marginals(initial, transitions, unary, *, hard=False, decode_path=True):
    """Log-domain banded forward/backward and a valid hard decoding."""
    length, count = unary.shape
    if hard:
        if not bool(torch.isfinite(torch.logsumexp(initial+unary[0], 0))):
            raise FloatingPointError('No feasible ordered state path')
        path = decode(initial, transitions, unary)
        resp, pair = from_labels(path, count, unary)
        return resp, pair, path
    forward = torch.full_like(unary, -torch.inf)
    backward = torch.zeros_like(unary)
    forward[0] = initial+unary[0]
    scales = unary.new_zeros(length)
    scales[0] = torch.logsumexp(forward[0], 0)
    forward[0] -= scales[0]
    stay, advance = transitions.diag(), transitions.diag(1)
    for t in range(1, length):
        moved = torch.cat((unary.new_full((1,), -torch.inf), forward[t-1, :-1]+advance))
        forward[t] = unary[t]+torch.logaddexp(forward[t-1]+stay, moved)
        scales[t] = torch.logsumexp(forward[t], 0)
        forward[t] -= scales[t]
    if not bool(torch.isfinite(scales).all()):
        raise FloatingPointError('No feasible ordered state path')
    for t in range(length-2, -1, -1):
        future = unary[t+1]+backward[t+1]
        moved = torch.cat((advance+future[1:], unary.new_full((1,), -torch.inf)))
        backward[t] = torch.logaddexp(stay+future, moved)-scales[t+1]
    resp = torch.softmax(forward+backward, 1)
    pair = unary.new_zeros((length, count, count))
    index = torch.arange(count, device=unary.device)
    pair[1:, index, index] = torch.exp(forward[:-1]+stay+unary[1:]+backward[1:]-scales[1:, None])
    if count > 1:
        pair[1:, index[:-1], index[1:]] = torch.exp(
            forward[:-1, :-1]+advance+unary[1:, 1:]+backward[1:, 1:]-scales[1:, None])
    pair[1:] /= pair[1:].sum((1, 2))[:, None, None]
    path = decode(initial, transitions, unary) if decode_path else None
    return resp, pair, path
