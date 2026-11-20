"""Common candidate pools for quota-based offline structural exploration."""
import torch

from . import _proposal_search as screen, _dynamics_proposals as dynamics
from ._proposal_quotas import FAMILIES, canonical_name, select


def weights(state):
    mode = canonical_name(getattr(state, 'offline_candidate_search', 'combined'))
    return (getattr(state, 'proposal_family_weights', (1., 1., 1.)) if mode == 'combined'
            else tuple(float(family == mode) for family in FAMILIES))


def iteration(state):
    return getattr(state, '_proposal_iteration', max(0, getattr(state, '_birth_iteration', 1) - 1))


def partition_key(resp, item):
    parent = item['parent']
    if item.get('allocation') is None:
        return 'seed', parent, item['seed']
    members = torch.where(resp.argmax(1) == parent)[0]
    child = item['allocation'][members].bool()
    return 'partition', parent, min(tuple(members[child].tolist()), tuple(members[~child].tolist()))


def pools(state, resp, latent, snr, *, ordered=False, original_seeds=None, statistics_cache=None):
    """Build cheap pools in their own coordinate systems, without fitting LDSs."""
    requested = weights(state)
    result = {family: [] for family in FAMILIES}
    latent_scores = state.weight_mean(latent, snr)
    reference = getattr(state, '_offline_proposal_geometry', None)
    original = state._offline_score_features
    if original is None and requested[0] > 0:
        original = screen.score_guided_features(state, state._offline_observations)
    coordinates = {'score_guided': original,
                   'noise_aware': None if reference is None else reference.features}
    if not ordered:
        raw = getattr(state, 'q_last', None)
        if raw is None or raw.shape[:2] != resp.shape:
            raw_scores = torch.zeros_like(resp)
        else:
            raw_scores = state.weight_mean(raw, snr)
        projected = screen.observation_scores(state, resp.shape[1]) if reference is not None else None
        scores = {'score_guided': raw_scores,
                  'noise_aware': None if projected is None else state.weight_mean(projected, snr)}
    for index, family in enumerate(FAMILIES[:2]):
        if requested[index] <= 0:
            continue
        if family == 'score_guided' and original_seeds is not None:
            labels = resp.argmax(1)
            result[family] = [dict(parent=int(labels[seed]), seed=int(seed), allocation=None,
                                   details=dict(proposal='fit_score_seed')) for seed in original_seeds]
            continue
        features = coordinates[family]
        if features is None:
            continue
        if ordered:
            result[family] = temporal_candidates(resp, features, latent_scores)
        else:
            losses = -(state.static_factor * scores[family] + state.dynamic_factor * latent_scores)
            ranked = screen.rank_splits(state, resp, features, losses, latent_scores,
                max(1, 3 * resp.shape[1]), include_windows=False)
            result[family] = [dict(parent=candidate[0], seed=candidate[1],
                allocation=screen.split_allocation(resp, features, candidate),
                details=dict(candidate[-1], selection=selection)) for candidate, selection in ranked]
    values = getattr(state, '_offline_dynamics_features', None)
    if requested[2] > 0 and values is not None:
        result['dynamics'] = dynamics.candidates(state, resp, values, ordered=ordered,
                                                statistics_cache=statistics_cache)
        if not ordered:
            labels = resp.argmax(1)
            for parent in range(resp.shape[1]):
                included = torch.where(labels == parent)[0]
                window = screen.latent_window(state, resp, parent, included, latent_scores)
                if window is None:
                    continue
                seed, assignment, details = window
                rows = torch.arange(len(resp), device=resp.device)
                allocation = ((rows >= details['window_start']) & (rows <= details['window_end'])).to(resp.dtype)
                allocation[included] = assignment.to(resp.dtype)
                result['dynamics'].append(dict(parent=parent, seed=seed,
                                               allocation=allocation, details=details))
    return result


def temporal_candidates(resp, features, latent_scores):
    """Original ordered appearance boundaries plus worst-latent boundary seeds."""
    labels = resp.argmax(1)
    result = []
    for parent in range(resp.shape[1]):
        included = torch.where(labels == parent)[0]
        if len(included) < 2:
            continue
        local = features[included]
        sums = local.cumsum(0)
        sizes = torch.arange(1, len(local), dtype=local.dtype, device=local.device)
        gains = (sums[:-1].square().sum(1) / sizes
            + (sums[-1] - sums[:-1]).square().sum(1) / (len(local) - sizes)
            - sums[-1].square().sum() / len(local))
        chosen = torch.argsort(gains, descending=True)[:2].tolist()
        worst = int(latent_scores[included, parent].argmin())
        chosen += [worst - 1, worst]
        for offset in dict.fromkeys(chosen):
            if 0 <= offset < len(local) - 1:
                cut = int(included[offset + 1])
                allocation = (torch.arange(len(resp), device=resp.device) >= cut).to(resp.dtype)
                result.append(dict(parent=parent, seed=cut, allocation=allocation,
                    details=dict(proposal='temporal_boundary', cut=cut, proxy_gain=float(gains[offset]))))
    return sorted(result, key=lambda item: item['details']['proxy_gain'], reverse=True)


def merge_pools(state, resp, *, ordered=False, statistics_cache=None):
    count = resp.shape[1]
    pairs = ([(first, first + 1) for first in range(count - 1)] if ordered else
             [(first, second) for first in range(count) for second in range(first + 1, count)])
    result = {family: [] for family in FAMILIES}
    mass = resp.sum(0)
    reference = getattr(state, '_offline_proposal_geometry', None)
    raw = None
    if weights(state)[0] > 0:
        raw = getattr(state, '_offline_merge_features', None)
        if raw is None:
            raw = state._offline_observations.reshape(len(resp), -1)
            raw = (raw - raw.mean(0)) / raw.std(0, unbiased=False).clamp_min(1e-8)
            state._offline_merge_features = raw
    for index, (family, features) in enumerate((('score_guided', raw),
            ('noise_aware', None if reference is None else reference.features))):
        if weights(state)[index] > 0 and features is not None:
            centers = resp.T @ features / mass[:, None]
            result[family] = sorted([dict(clusters=(a,b), details=dict(
                proposal='feature_distance_merge', proxy_distance=float((centers[a]-centers[b]).square().sum())))
                for a,b in pairs], key=lambda item:item['details']['proxy_distance'])
    values = getattr(state, '_offline_dynamics_features', None)
    if weights(state)[2] > 0 and values is not None and state.dynamic_factor != 0:
        result['dynamics'] = dynamics.merge_scores(state, resp, values, pairs,
                                                  statistics_cache=statistics_cache)
    return result


def record(state, split, merge=None):
    current = getattr(state, 'offline_search_diagnostics', {})
    entry = dict(iteration=iteration(state), splits=split)
    if merge is not None:
        entry['merges'] = merge
    state.offline_search_diagnostics = {**current, 'last_allocation':entry,
        'allocations':current.get('allocations', []) + [entry]}


def structural(state, resp, latent, snr, *, ordered=False):
    budget = state.n_explore_steps
    statistics_cache = {}
    merges = merge_pools(state, resp, ordered=ordered, statistics_cache=statistics_cache) if budget else {f:[] for f in FAMILIES}
    unique = {item['clusters'] for pool in merges.values() for item in pool}
    merge_budget = min(len(unique), budget // 2)
    split_budget = budget - merge_budget
    can_split = resp.shape[1] < min(len(resp), state.max_models or len(resp))
    candidates = pools(state, resp, latent, snr, ordered=ordered,
                       statistics_cache=statistics_cache) if can_split and budget else {}
    chosen, split_log = select(candidates, split_budget, weights(state), iteration(state),
                               lambda item:partition_key(resp,item))
    # Unused split slots can examine additional unique merges.
    chosen_merges, merge_log = select(merges, budget-len(chosen), weights(state),
        iteration(state)+split_budget, lambda item:item['clusters'])
    record(state, split_log, merge_log)
    return ([('split', family, item) for family,item in chosen]
            + [('merge', family, item) for family,item in chosen_merges])


def birth_seeds(state, resp, latent, snr, representers, original_seeds):
    budget = min(state.n_explore_steps, len(resp))
    candidates = pools(state, resp, latent, snr, original_seeds=original_seeds) if budget else {}
    labels = resp.argmax(1)
    prepared = {family:[] for family in FAMILIES}
    for family, pool in candidates.items():
        for item in pool:
            parent = item['parent']
            if item['allocation'] is None:
                if item['seed'] != int(representers[parent]):
                    prepared[family].append(item)
                continue
            members = torch.where(labels == parent)[0]
            child = members[item['allocation'][members].bool()]
            available = child[child != representers[parent]]
            if not len(available):
                item = {**item, 'allocation':1 - item['allocation']}
                child = members[item['allocation'][members].bool()]
                available = child[child != representers[parent]]
            if not len(available):
                continue
            # Several distinct representative choices allow the same partition
            # to survive a seed collision with a different family's proposal.
            seeds = [item['seed']] if item['seed'] in available else []
            seeds += [int(row) for row in available[:min(4,budget+1)] if int(row) not in seeds]
            prepared[family] += [{**item,'seed':seed} for seed in seeds]
    def key(item):
        return frozenset((partition_key(resp,item), ('birth_seed',item['seed'])))
    chosen, log = select(prepared, budget, weights(state), iteration(state), key)
    record(state, log)
    seeds, details = [], []
    for family,item in chosen:
        seed = item['seed']
        seeds.append(seed)
        if item['allocation'] is not None:
            state._offline_birth_allocations[seed] = item['parent'],item['allocation']
        details.append(dict(family=family,seed=seed,**item['details']))
    state._offline_last_birth_seeds = seeds
    state._offline_last_birth_proposals = details
    return seeds
