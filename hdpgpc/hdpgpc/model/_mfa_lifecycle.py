"""Transactional MFA birth/pruning proposals using the complete objective."""
from copy import copy
import math

import torch

from ..hdp import HDPVariational
from ..inference._partitions import _partition_key, route_assignments


def state_key(state):
    return f'{state.M}:{_partition_key(state.resp).hex()}'


def _isolated(state):
    candidate = copy(state)
    candidate.hdp = HDPVariational(candidate)
    candidate.components = [component.copy() for component in state.components]
    candidate.noise = state.noise.clone()
    candidate.resp = state.resp.clone()
    if state.hmm_switch:
        candidate.respPair = state.respPair.clone()
    # All arrays changed by global updates belong to the transaction.
    for name in ('rho', 'omega', 'transTheta', 'startTheta'):
        if hasattr(state, name):
            setattr(candidate, name, getattr(state, name).clone())
    return candidate


def _birth(state, values, parent):
    from sklearn.cluster import KMeans

    indexes = torch.where(state.resp.argmax(1) == parent)[0]
    if len(indexes) < 2:
        return None
    seed = 0 if state.config.numerics.seed is None else int(state.config.numerics.seed)
    labels = KMeans(n_clusters=2, n_init=1, random_state=seed, max_iter=50).fit_predict(
        values[indexes].cpu().numpy())
    if len(set(labels)) != 2:
        return None
    group = torch.as_tensor(labels, device=values.device)
    centers = torch.stack([values[indexes[group == i]].mean(0) for i in range(2)])
    # Route ALL fractional parent mass, not just its MAP-owned rows.
    allocation = torch.softmax(-torch.cdist(values, centers).square()
                              / state.noise.mean().clamp_min(state.config.noise_floor), 1)
    k = state.M
    routing = torch.eye(k, k + 1, dtype=values.dtype, device=values.device).repeat(len(values), 1, 1)
    routing[:, parent, parent] = allocation[:, 0]
    routing[:, parent, k] = allocation[:, 1]
    work = _isolated(state)
    work.resp, pairs = route_assignments(state.resp, getattr(state, 'respPair', None), routing)
    if pairs is not None:
        work.respPair = pairs
    work.components.append(state.components[parent].copy())
    # copy() retains the ORIGINAL prior. Only variational posterior means are
    # seeded; subsequent updates always rebuild from that original prior.
    for child, center in ((parent, centers[0]), (k, centers[1])):
        component = work.components[child]
        mean = component.parameter_mean.clone()
        mean[:, 0] = center
        component.set_posterior(mean, component.parameter_covariance)
        stats = component.sufficient_statistics(values, work.resp[:, child], noise=work.noise,
                                                chunk_size=state.config.chunk_size)
        component.update_from_statistics(stats, noise=work.noise)
    work.M += 1
    work.update_weights(initialize=True)
    return work


def _prune(state, values, source):
    k = state.M
    others = [i for i in range(k) if i != source]
    means = torch.stack([state.components[i].mean for i in others])
    target = others[int(torch.cdist(state.components[source].mean[None], means).argmin())]
    routing = values.new_zeros((len(values), k, k - 1))
    mapping = {old: new for new, old in enumerate(others)}
    for old in range(k):
        routing[:, old, mapping[target if old == source else old]] = 1.
    work = _isolated(state)
    work.resp, pairs = route_assignments(state.resp, getattr(state, 'respPair', None), routing)
    if pairs is not None:
        work.respPair = pairs
    work.components = [work.components[i] for i in others]
    work.M -= 1
    work.update_weights(initialize=True)
    return work


def explore_structure(state, values, evidence, parts, iteration):
    """Refine bounded proposals in isolation; accept only a full-objective gain."""
    from ..inference._mfa import _evidence, coordinate_sweep, objective_components

    config = state.config
    limit = min(len(values), config.hdp.max_models or len(values))
    masses = state.resp.sum(0)
    prunes = sorted((k for k in range(state.M) if state.M > 1
                    and float(masses[k]) < config.prune_threshold), key=lambda k: float(masses[k]))
    labels = state.resp.argmax(1)
    births = []
    if state.M < limit:
        for k in range(state.M):
            group = values[labels == k]
            if len(group) > 1:
                # Size-weighted unexplained spread only ranks proposals. The
                # complete variational objective alone decides acceptance.
                spread = float((group - group.mean(0)).square().sum())
                births.append((spread, k))
    births = [k for _, k in sorted(births, reverse=True)]
    proposals = []
    for index in range(max(len(prunes), len(births))):
        if index < len(prunes):
            proposals.append(('prune', prunes[index]))
        if index < len(births):
            proposals.append(('birth', births[index]))
    best, best_evidence, best_parts = state, evidence, parts
    records = []
    state.seen_partitions.add(state_key(state))
    for move, source in proposals[:config.n_explore_steps]:
        record = dict(iteration=iteration, move=move, source=source, accepted=False,
                      baseline=parts['total'])
        try:
            work = (_birth if move == 'birth' else _prune)(state, values, source)
            if work is None:
                record['reason'] = 'empty_partition'
                records.append(record)
                continue
            local = _evidence(work, values)
            score = {key: float(value) for key, value in objective_components(work, local).items()}
            for _ in range(config.proposal_refine_steps):
                local, score, _ = coordinate_sweep(work, values, local, score)
            gain = score['total'] - parts['total']
            record.update(proposal=score['total'], delta_elbo=gain, components=work.M)
            key = state_key(work)
            tolerance = 1e-6 + 1e-8 * max(1., abs(parts['total']), abs(score['total']))
            if key in state.seen_partitions:
                record['reason'] = 'visited_partition'
            elif move == 'birth' and not bool((work.resp.argmax(1) == work.M - 1).any()):
                record['reason'] = 'unoccupied_newborn'
            elif math.isfinite(gain) and gain > tolerance and score['total'] > best_parts['total']:
                best, best_evidence, best_parts = work, local, score
                record['reason'] = 'candidate'
            else:
                record['reason'] = 'insufficient_objective_gain'
        except (FloatingPointError, torch.linalg.LinAlgError) as error:
            record['reason'] = str(error)
        records.append(record)
    accepted = best is not state
    if accepted:
        # Copy only fitted scientific state; histories remain owned by fit().
        for name in ('M', 'components', 'noise', 'resp', 'rho', 'omega', 'respPair',
                     'transTheta', 'startTheta'):
            if hasattr(best, name):
                setattr(state, name, getattr(best, name))
        state.seen_partitions.add(state_key(state))
        for record in records:
            if record.get('proposal') == best_parts['total'] and record.get('reason') == 'candidate':
                record['accepted'], record['reason'] = True, 'accepted'
                break
    for record in records:
        if record.get('reason') == 'candidate':
            record['reason'] = 'better_proposal_selected'
    state.structure_history.extend(records)
    return best_evidence, best_parts, accepted
