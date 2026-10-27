"""Banded ordered-chain messages and joint MAP decoding."""
import torch


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


def posterior_path(resp, pair):
    """Decode the MAP joint path from consistent chain state/pair marginals."""
    length, count = resp.shape
    if length == 1:
        return torch.zeros(1, dtype=torch.long, device=resp.device)
    # A time-inhomogeneous chain has q(s_t|s_{t-1})=xi/gamma.
    value = torch.log(resp[0])
    pointers = torch.zeros((length, count), dtype=torch.long, device=resp.device)
    for t in range(1, length):
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


def marginals(initial, transitions, unary, *, hard=False):
    """Log-domain banded forward/backward and a valid hard decoding."""
    length, count = unary.shape
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
    path = decode(initial, transitions, unary)
    if hard:
        resp = torch.nn.functional.one_hot(path, count).to(unary)
        pair.zero_()
        pair[1:] = resp[:-1, :, None]*resp[1:, None, :]
    return resp, pair, path
