"""Commit state assignments and their adjacent-time transition pairs together.

Ordinary hard inference retains its marginal MAP labels. Ordered inference
supplies its already decoded feasible path. Soft HMM pair marginals must come
from forward/backward; they must not be replaced by products of state marginals.
"""
import torch


def hard_pairs(states):
    """Build incoming pairs from committed one-hot states; row zero is empty."""
    pairs = states.new_zeros((len(states), states.shape[1], states.shape[1]))
    pairs[1:] = states[:-1, :, None]*states[1:, None, :]
    return pairs


def from_labels(labels, count, reference):
    """Construct a hard chain on the reference tensor's dtype and device."""
    labels = torch.as_tensor(labels, device=reference.device, dtype=torch.long)
    states = reference.new_zeros((len(labels), count))
    states.scatter_(1, labels[:, None], 1.)
    return states, hard_pairs(states)


def from_log_marginals(owner, log_states, log_pairs=None, *, hard=None):
    """Harden one sequence or retain the existing coherent soft marginals."""
    hard = getattr(owner, 'hard_assignments', True) if hard is None else hard
    states = owner._safe_exp(log_states, hard=hard)
    if hard:
        return states, hard_pairs(states)
    return states, owner._safe_exp(log_pairs, hard=False)


def force_last(states, pairs, state):
    """Force the final state and replace its entire incoming transition row.

    Also preserves the existing soft forced-assignment convention: the previous
    state marginal remains unchanged and the destination becomes deterministic.
    """
    previous = states[-2].clone() if len(states) > 1 else None
    states[-1].zero_()
    states[-1, state] = 1.
    pairs[-1].zero_()
    if previous is not None:
        pairs[-1, :len(previous), state] = previous


def append_hard(states, pairs, state, count):
    """Copy a consistent prefix and append one state/pair without redecoding."""
    result = states.new_zeros((len(states)+1, count))
    result[:-1, :states.shape[1]] = states
    incoming = result.new_zeros((len(result), count, count))
    if pairs is not None and pairs.numel():
        incoming[:-1, :pairs.shape[1], :pairs.shape[2]] = pairs
    elif len(states):
        incoming[:-1, :states.shape[1], :states.shape[1]] = hard_pairs(states)
    force_last(result, incoming, state)
    return result, incoming
