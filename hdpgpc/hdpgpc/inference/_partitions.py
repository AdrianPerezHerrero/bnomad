"""Small assignment-only helpers shared by structural inference engines."""
from array import array
from hashlib import blake2b

import torch


def _partition_key(resp):
    """Compact MAP membership key, invariant to cluster-slot permutations."""
    names = {}
    labels = resp.argmax(1).detach().cpu().tolist()
    canonical = array('q', (names.setdefault(label, len(names)) for label in labels))
    return blake2b(canonical.tobytes(), digest_size=16).digest()


def route_assignments(resp, pair, routing):
    """Push singleton/pair marginals through a time-dependent stochastic map.

    routing[t,old,new] maps each old state to new states. Empty boundary pair
    rows stay empty. This includes splitting, merging, deleting and reordering.
    """
    result = torch.einsum('ti,tij->tj', resp, routing)
    if pair is None:
        return result, None
    previous = torch.cat((routing[:1], routing[:-1]), 0)
    pairs = torch.einsum('tij,tia,tjb->tab', pair, previous, routing)
    return result, pairs
