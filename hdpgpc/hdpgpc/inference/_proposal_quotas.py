"""Deterministic quotas for independently ranked structural proposal families."""
import math

FAMILIES = ('score_guided', 'noise_aware', 'dynamics')


def canonical_name(value):
    return 'score_guided' if value == 'legacy' else value


def quotas(budget, weights, offset=0):
    """Split a budget; rotate enabled families when fewer slots than families."""
    active = [i for i, weight in enumerate(weights) if weight > 0]
    result = [0] * len(weights)
    if not active or budget == 0:
        return result
    rotated = active[offset % len(active):] + active[:offset % len(active)]
    if budget < len(active):
        for index in rotated[:budget]:
            result[index] = 1
        return result
    scale = max(weights)
    relative = [weight / scale for weight in weights]
    total = sum(relative[index] for index in active)
    exact = [budget * weight / total for weight in relative]
    result = [math.floor(value) for value in exact]
    order = sorted(rotated, key=lambda i: exact[i] - result[i], reverse=True)
    for index in order[:budget - sum(result)]:
        result[index] += 1
    return result


def select(pools, budget, weights, offset, key):
    """Honor per-family quotas, deduplicate, and redistribute unused slots.

    Proxies are ordered only within their own family; their numerical scales
    are never compared. A duplicate is charged only to the family that supplies
    the selected candidate. Entries are returned as (family, candidate) pairs.
    """
    requested = quotas(budget, weights, offset)
    positions = {family: 0 for family in FAMILIES}
    selected, seen = [], set()
    counts = {family: 0 for family in FAMILIES}
    duplicates = 0

    def take(family):
        nonlocal duplicates
        pool = pools.get(family, ())
        while positions[family] < len(pool):
            item = pool[positions[family]]
            positions[family] += 1
            identity = key(item)
            identities = identity if isinstance(identity, frozenset) else frozenset((identity,))
            if seen.intersection(identities):
                duplicates += 1
                continue
            seen.update(identities)
            selected.append((family, item))
            counts[family] += 1
            return True
        return False

    order = list(range(3))
    order = order[offset % 3:] + order[:offset % 3]
    for level in range(max(requested, default=0)):
        for index in order:
            if requested[index] > level:
                take(FAMILIES[index])
    # Exhausted/duplicate families cannot waste a fit slot. Continue in quota
    # rounds over the remaining enabled families until no unique item remains.
    while len(selected) < budget:
        progressed = False
        for index in order:
            if weights[index] > 0 and len(selected) < budget:
                progressed |= take(FAMILIES[index])
        if not progressed:
            break
    return selected, dict(budget=budget,
        requested=dict(zip(FAMILIES, requested)), selected=counts,
        duplicate_candidates=duplicates, unused=budget - len(selected))
