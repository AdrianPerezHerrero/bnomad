"""Shared, label-free offline proposal screening; never an inference objective.

A common affine decoder supplies noise-whitened likelihood information. Only
proposal geometry and seed ranking use the screened coordinates: model inputs,
posterior updates, messages and complete acceptance scores remain untouched.
"""
from dataclasses import dataclass
import math

import torch

from ..dynamics import _clock
from ..emissions._fixed_observation import AffineGaussianObservation


@dataclass(frozen=True)
class OutputGeometry:
    observation: torch.Tensor
    transform: torch.Tensor
    precision: torch.Tensor


@dataclass(frozen=True)
class ProposalGeometry:
    features: torch.Tensor
    outputs: tuple[OutputGeometry, ...]
    metadata: dict
    codes: torch.Tensor


def _initialize_affine(state, observations):
    """Build once per fit, resetting old caches even when the data shape repeats.

    The calibrated decoder/noise is the common reference, including for MFA whose
    cluster-specific parameters may subsequently change. The Gaussian noise bound
    is a screening heuristic; it does not assert independent data over time or
    discard weak directions from the fitted model. Non-affine emissions retain
    their original search, since they have no common exact information map.
    """
    state._offline_proposal_geometry = None
    state._offline_birth_allocations = {}
    state._offline_last_birth_seeds = []
    from ._proposal_quotas import canonical_name
    mode = canonical_name(getattr(state, 'offline_candidate_search', 'combined'))
    metadata = dict(method=mode, active=False)
    state.offline_search_diagnostics = metadata
    if mode == 'score_guided':
        metadata['reason'] = 'original candidate search requested'
        return None
    values = observations.detach()
    decoders = []
    for models in state.gpmodels:
        model = (getattr(state, 'fixed_observation_model', None)
                 if getattr(models[0], 'factor_analysis', False) else
                 getattr(models[0], 'fixed_observation_model', None))
        if not isinstance(model, AffineGaussianObservation):
            metadata['reason'] = 'no common affine observation decoder'
            return None
        decoders.append(model)
    outputs, features, details, codes = [], [], [], []
    for output, model in enumerate(decoders):
        data = values[:, :, output]
        C = torch.tensor(model.C.copy(), dtype=data.dtype, device=data.device)
        R = torch.tensor(model.Sigma.copy(), dtype=data.dtype, device=data.device)
        offset = torch.tensor(model.offset.copy(), dtype=data.dtype, device=data.device)
        inverse_C = (C / R.diag()[:, None] if torch.equal(R, torch.diag(R.diag()))
                     else torch.cholesky_solve(C, torch.linalg.cholesky(R)))
        H = C.T @ inverse_C
        H = .5 * (H + H.T)
        eigenvalues, eigenvectors = torch.linalg.eigh(H)
        tolerance = torch.finfo(H.dtype).eps * max(H.shape) * eigenvalues.max().clamp_min(1.)
        active = eigenvalues > tolerance
        rank = int(active.sum())
        if rank == 0:
            metadata['reason'] = 'decoder has no observable latent direction'
            return None
        safe = torch.where(active, eigenvalues, torch.ones_like(eigenvalues))
        root = (eigenvectors * torch.where(active, safe.sqrt(), 0.)) @ eigenvectors.T
        inverse_root = (eigenvectors * torch.where(active, safe.rsqrt(), 0.)) @ eigenvectors.T
        # Under the common reference, u's noise covariance is the projector on
        # the observable latent subspace. Orthogonal observation density drops out.
        u = ((data - offset) @ inverse_C) @ inverse_root
        codes.append(u)
        centered = u - u.mean(0)
        count, dimension = u.shape
        covariance = centered.T @ centered / max(1, count - 1)
        spectrum, directions = torch.linalg.eigh(.5 * (covariance + covariance.T))
        probability = .01
        bound = (1 + math.sqrt(rank / max(1, count - 1))
                 + math.sqrt(2 * math.log(2 / probability) / max(1, count - 1))) ** 2
        selected = torch.where(spectrum > bound)[0][-20:]
        fallback = selected.numel() == 0
        if fallback:
            # Keep exploration possible; failing this conservative noise screen
            # is not evidence that temporal or weak regimes do not exist.
            selected = torch.arange(max(0, dimension - min(20, rank)), dimension, device=u.device)
        basis = directions[:, selected]
        transform = root @ basis
        outputs.append(OutputGeometry(u @ basis, transform, transform @ transform.T))
        features.append(centered @ basis)
        details.append(dict(output=output, latent_dimension=dimension, observable_rank=rank,
            retained_rank=len(selected), noise_bound=bound, fallback=fallback,
            noise_reference_failure_probability=probability))
    features = torch.cat(features, dim=1)
    # One scalar preserves relative information; coordinate-wise standardization
    # would amplify the weak/noise directions this search is intended to discount.
    features = features / features.square().mean().sqrt().clamp_min(1e-12)
    metadata.update(active=True, ranking_dimension=features.shape[1], outputs=details)
    geometry = ProposalGeometry(features, tuple(outputs), metadata, torch.cat(codes, dim=1))
    state._offline_proposal_geometry = geometry
    return geometry


def score_guided_features(state, observations):
    """Original standardized decoder-analysis coordinates, with local projection."""
    outputs = []
    for output, models in enumerate(state.gpmodels):
        model = (getattr(state, 'fixed_observation_model', None)
                 if getattr(models[0], 'factor_analysis', False) else
                 getattr(models[0], 'fixed_observation_model', None))
        data = observations[:, :, output]
        if isinstance(model, AffineGaussianObservation):
            decoder = torch.tensor(model.C.copy(), dtype=data.dtype, device=data.device)
            offset = torch.tensor(model.offset.copy(), dtype=data.dtype, device=data.device)
            data = (data - offset) @ torch.linalg.pinv(decoder).T
        outputs.append(data)
    features = torch.cat(outputs, dim=1)
    features = features - features.mean(0)
    scale = features.std(0, unbiased=False)
    active = scale > torch.finfo(features.dtype).eps * scale.max().clamp_min(1.)
    if not bool(active.any()):
        return features[:, :1] * 0
    features = features[:, active] / scale[active]
    if features.shape[1] > 20:
        generator = torch.Generator(device=features.device).manual_seed(0)
        basis = torch.randn((features.shape[1], 20), generator=generator,
                            dtype=features.dtype, device=features.device)
        basis, _ = torch.linalg.qr(basis, mode='reduced')
        features = features @ basis
        features = (features - features.mean(0)) / features.std(0, unbiased=False).clamp_min(1e-8)
    return features


def initialize(state, observations):
    """Initialize family-independent caches once per fit, with explicit fallbacks."""
    from . import _dynamics_proposals
    geometry = _initialize_affine(state, observations)
    mode = state.offline_search_diagnostics['method']
    state._offline_observations = observations.detach()
    state._proposal_iteration = 0
    state._offline_score_features = None
    state._offline_dynamics_features = None
    available = ['score_guided']
    if geometry is not None:
        available.append('noise_aware')
    if mode in ('combined', 'dynamics'):
        if mode == 'combined':
            state._offline_score_features = score_guided_features(state, observations)
        # Dynamics can use weak temporal directions excluded by the appearance
        # noise screen. GP paths use a fixed waveform frame instead of comparing
        # cluster-specific learned C matrices or latent rotations.
        codes = geometry.codes if geometry is not None else observations.reshape(len(observations), -1)
        state._offline_dynamics_features = _dynamics_proposals.features(codes)
        if state.dynamic_factor != 0:
            available.append('dynamics')
        state.offline_search_diagnostics.update(active=True,
            dynamics_dimension=state._offline_dynamics_features.shape[1])
    state.offline_search_diagnostics.update(families_available=available,
        family_weights=list(getattr(state, 'proposal_family_weights', (1., 1., 1.))))
    return geometry


def _covariance_traces(covariances, precision):
    """Contract history in bounded chunks, avoiding a T*d*d temporary tensor."""
    if torch.is_tensor(covariances):
        return torch.einsum('ij,nji->n', precision, covariances)
    chunk = max(1, (1 << 20) // precision.numel())
    return torch.cat([torch.einsum('ij,nji->n', precision,
                                  torch.stack(covariances[start:start + chunk]))
                      for start in range(0, len(covariances), chunk)])


def observation_scores(state, count):
    """Expected projected scores for ranking only, using current latent moments.

    The reference stays fixed when a cluster learns C/Sigma; in that case this is
    an auxiliary proxy, not a factorization of its learned emission likelihood.
    No first-observation noise inflation is applied to this proposal-only proxy.
    """
    geometry = getattr(state, '_offline_proposal_geometry', None)
    if geometry is None:
        return None
    rows = torch.arange(len(geometry.features), device=geometry.features.device)
    outputs = []
    for output, reference in enumerate(geometry.outputs):
        scores = []
        for gp in state.gpmodels[output][:count]:
            local = getattr(gp, '_factor_local_moments', None)
            if (getattr(gp, 'factor_analysis', False) and gp._independent_factors()
                    and local is not None and len(local[0]) == len(rows)):
                means, covariances = local
                projected = means.reshape(len(rows), -1) @ reference.transform
                trace = _covariance_traces(covariances, reference.precision)
            elif _clock.enabled(gp) and getattr(gp, '_sampling_trajectory', None) is not None:
                trajectory = _clock.trajectory(gp)
                projected = torch.stack(trajectory.means[:len(rows)]).reshape(len(rows), -1) @ reference.transform
                trace = _covariance_traces(trajectory.covariances[:len(rows)], reference.precision)
            else:
                indexes = torch.as_tensor(gp.indexes, device=rows.device, dtype=torch.long)
                positions = torch.searchsorted(indexes, rows, right=True)
                # Contract each retained covariance once before expanding its
                # view over skipped observations on the cluster clock.
                projected = (torch.stack(gp.f_star_sm).reshape(len(gp.f_star_sm), -1)
                             @ reference.transform)[positions]
                trace = _covariance_traces(gp.cov_f_sm, reference.precision)[positions]
            residual = reference.observation - projected
            scores.append(-.5 * (residual.square().sum(1) + trace
                                 + reference.transform.shape[1] * math.log(2 * math.pi)))
        outputs.append(torch.stack(scores, dim=1))
    return torch.stack(outputs, dim=-1)


def split_gain(features, weights, assignment):
    """Diagonal Gaussian mixture gain with a BIC-style proposal-only penalty."""
    groups = torch.stack((assignment == 0, assignment == 1), 1).to(weights.dtype)
    grouped_weights = weights[:, None] * groups
    masses = grouped_weights.sum(0)
    # Singleton children are valid, including fractional parent responsibility
    # in soft mode. Only empty children cannot define the weighted proxy fit.
    if bool((masses <= 0.).any()):
        return None
    mass = masses.sum()
    mean = (weights[:, None] * features).sum(0) / mass
    variance = (weights[:, None] * (features - mean).square()).sum(0) / mass
    means = grouped_weights.T @ features / masses[:, None]
    variances = torch.stack([
        (grouped_weights[:, child, None] * (features - means[child]).square()).sum(0) / masses[child]
        for child in range(2)])
    # Features have a common global scale: use the same floor in both models.
    variance, variances = variance.clamp_min(.01), variances.clamp_min(.01)
    dimension = features.shape[1]
    normalizer = dimension * math.log(2. * math.pi)
    single = -.5 * (normalizer + variance.log().sum()
                    + ((features - mean).square() / variance).sum(1))
    components = -.5 * (normalizer + variances.log().sum(1)[None, :]
        + ((features[:, None, :] - means[None, :, :]).square() / variances[None, :, :]).sum(2))
    mixture = torch.logsumexp(components + (masses / mass).log()[None, :], dim=1)
    likelihood_gain = (weights * (mixture - single)).sum()
    # One Gaussian: 2p parameters. Two Gaussians plus a weight: 4p + 1.
    penalty = .5 * (2 * dimension + 1) * mass.log()
    gain = likelihood_gain - penalty
    if not bool(torch.isfinite(gain)):
        return None
    return {'proxy_gain': float(gain), 'proxy_likelihood_gain': float(likelihood_gain),
            'proxy_complexity_penalty': float(penalty), 'proxy_child_mass': masses.tolist(),
            'proxy_dimension': dimension}


def latent_window(state, resp, parent, included, latent_scores):
    """Propose a low-q_lat interval on the parent's retained-state clock.

    q_lat is zero for unretained observations, and the first retained state has
    an initial-prior score rather than a transition score. Neither is eligible
    for window selection. Prefix sums keep this screening linear in parent size.
    """
    if len(included) < 3 or state.dynamic_factor == 0:
        return None
    eligible = torch.ones_like(included, dtype=torch.bool)
    for models in state.gpmodels:
        indexes = models[parent].indexes
        retained = torch.zeros(len(resp), dtype=torch.bool, device=included.device)
        if len(indexes) > 1:
            retained[torch.as_tensor(indexes[1:], device=included.device, dtype=torch.long)] = True
        eligible &= retained[included]
    rows = included[eligible]
    if len(rows) < 2:
        return None
    weights, scores = resp[rows, parent], latent_scores[rows, parent]
    # Do not bridge missing/nonfinite scores to invent a valid transition run.
    if not bool(torch.isfinite(scores).all() and torch.isfinite(weights).all()
                and (weights > 0).all()):
        return None
    length = min(max(2, math.isqrt(len(included))), len(rows), len(included) - 1)
    cumulative_weight = torch.cat((weights.new_zeros(1), weights.cumsum(0)))
    cumulative_score = torch.cat((scores.new_zeros(1), (weights * scores).cumsum(0)))
    means = ((cumulative_score[length:] - cumulative_score[:-length])
             / (cumulative_weight[length:] - cumulative_weight[:-length]))
    start = int(means.argmin())
    mean = (weights * scores).sum() / weights.sum()
    gap = mean - means[start]
    tolerance = 32 * torch.finfo(scores.dtype).eps * scores.abs().max().clamp_min(1.)
    if not bool(torch.isfinite(gap) and gap > tolerance):
        return None
    window = rows[start:start + length]
    # Use a time interval also for fractional responsibility outside MAP members.
    assignment = ((included >= window[0]) & (included <= window[-1])).long()
    seed = int(window[scores[start:start + length].argmin()])
    details = {'proposal': 'worst_q_lat_window', 'window_start': int(window[0]),
               'window_end': int(window[-1]), 'window_length': length,
               'window_q_lat_mean': float(means[start]),
               'parent_transition_q_lat_mean': float(mean),
               'window_q_lat_gap': float(gap)}
    return seed, assignment, details


def rank_splits(state, resp, split_features, losses, latent_scores, split_budget, *, include_windows=True):
    """Shared bounded appearance/window candidates, before any model refits."""
    count = resp.shape[1]
    labels = resp.argmax(1)
    if split_budget == 0:
        return []
    splits = []
    if count < len(resp) and (state.max_models is None or count < state.max_models):
        members = {k: torch.where(labels == k)[0] for k in range(count)}
        parents = [k for k in range(count) if len(members[k]) >= 2]
        parent_loss = {k: float((resp[:, k] * losses[:, k]).sum()) for k in parents}
        candidates, windows = [], []
        used = set()
        for parent in parents:
            included = members[parent]
            window = latent_window(state, resp, parent, included, latent_scores) if include_windows else None
            if window is None:
                continue
            seed, assignment, details = window
            child_rows = included[assignment == 1]
            key = (parent, min(tuple(child_rows.tolist()), tuple(included[assignment == 0].tolist())))
            local, weights = split_features[included], resp[included, parent]
            proxy = split_gain(local, weights, assignment)
            if proxy is None:
                continue
            used.add(key)
            centers_local = [(local[assignment == child] * weights[assignment == child, None]).sum(0)
                             / weights[assignment == child].sum() for child in range(2)]
            windows.append((parent, seed, included, assignment, *centers_local, {**proxy, **details}))
        # Screen every eligible parent before allocating expensive fit slots.
        # Two appearance restarts and one latent window keep screening linear.
        for restart in range(2):
            for parent in parents:
                included = members[parent]
                local = split_features[included]
                local_weights = resp[included, parent]
                center = (local_weights[:, None] * local).sum(0) / local_weights.sum()
                seed = (torch.argmax(losses[included, parent]) if restart == 0 else
                        torch.argmax((local - center).square().sum(1)))
                first = local[seed]
                second = local[torch.argmax((local - first).square().sum(1))]
                assignment = None
                for _ in range(6):
                    distances = torch.stack(((local - first).square().sum(1),
                                              (local - second).square().sum(1)), 1)
                    current = distances.argmin(1)
                    if not bool((current == 0).any() and (current == 1).any()):
                        break
                    if assignment is not None and torch.equal(current, assignment):
                        break
                    assignment = current
                    first = (local[current == 0] * local_weights[current == 0, None]).sum(0) / local_weights[current == 0].sum()
                    second = (local[current == 1] * local_weights[current == 1, None]).sum(0) / local_weights[current == 1].sum()
                if assignment is None or not bool((assignment == 0).any() and (assignment == 1).any()):
                    continue
                child_rows = included[assignment == 1]
                # A split and its complemented partition are the same proposal.
                key = (parent, min(tuple(child_rows.tolist()), tuple(included[assignment == 0].tolist())))
                if key in used:
                    continue
                used.add(key)
                details = split_gain(local, local_weights, assignment)
                if details is None:
                    continue
                child_seed = child_rows[torch.argmin((split_features[child_rows] - second).square().sum(1))]
                candidates.append((parent, int(child_seed), included, assignment, first, second, details))
        windows.sort(key=lambda item: item[-1]['window_q_lat_gap'], reverse=True)
        window_slots = min(len(windows), max(1, split_budget // 3))
        splits = [(item, 'q_lat_window') for item in windows[:window_slots]]
        appearance_slots = split_budget - len(splits)
        candidates.sort(key=lambda item: item[-1]['proxy_gain'], reverse=True)
        # Preserve some score-driven exploration without letting it dominate.
        reserve = max(1, appearance_slots // 5) if appearance_slots >= 2 else 0
        selected = candidates[:appearance_slots - reserve]
        splits += [(item, 'proxy_gain') for item in selected]
        remaining = candidates[appearance_slots - reserve:]
        remaining.sort(key=lambda item: (parent_loss[item[0]], item[-1]['proxy_gain']), reverse=True)
        splits += [(item, 'score_exploration') for item in remaining[:split_budget - len(splits)]]
        # Unused appearance slots can explore other parents' latent windows.
        splits += [(item, 'q_lat_window') for item in windows[window_slots:
                    window_slots + split_budget - len(splits)]]
    return splits


def split_partition(resp, pair, parent, allocation):
    """Redistribute one parent's mass, preserving both pair marginals exactly."""
    count = resp.shape[1]
    child = resp[:, parent] * allocation
    proposed = resp.clone()
    proposed[:, parent] -= child
    proposed = torch.cat((proposed, child[:, None]), 1)
    before = torch.cat((allocation.new_zeros(1), allocation[:-1]))
    left = resp.new_ones((len(resp), count)); left[:, parent] = 1 - before
    right = resp.new_ones((len(resp), count)); right[:, parent] = 1 - allocation
    proposed_pair = pair.new_zeros((len(resp), count + 1, count + 1))
    proposed_pair[:, :count, :count] = pair * left[:, :, None] * right[:, None, :]
    proposed_pair[:, count, :count] = pair[:, parent, :] * before[:, None] * right
    proposed_pair[:, :count, count] = pair[:, :, parent] * allocation[:, None] * left
    proposed_pair[:, count, count] = pair[:, parent, parent] * before * allocation
    return proposed, proposed_pair


def split_allocation(resp, features, candidate):
    parent, _seed, included, assignment, first, second, details = candidate
    if details.get('proposal') == 'worst_q_lat_window':
        rows = torch.arange(len(resp), device=resp.device)
        allocation = ((rows >= details['window_start']) & (rows <= details['window_end'])).to(resp.dtype)
    else:
        allocation = ((features - second).square().sum(1)
                      < (features - first).square().sum(1)).to(resp.dtype)
    allocation[included] = assignment.to(resp.dtype)
    return allocation


def birth_seeds(state, resp, latent, snr, representers, *, original_seeds=()):
    """Adapt shared group screening to the legacy optimizer's seed proposals.

    Shared group allocations initialize births; their pair tensors are built
    lazily when each candidate is tried, rather than caching O(budget*T*K**2).
    Subsequent messages, refitting, native likelihoods and birth acceptance use
    their original implementations. Remaining slots explore farthest members in
    the same fixed screened geometry, excluding duplicates and the current
    representative of each member's parent.
    """
    state._offline_birth_allocations = {}
    from ._proposal_quotas import canonical_name
    mode = canonical_name(getattr(state, 'offline_candidate_search', 'combined'))
    if mode in ('combined', 'dynamics'):
        from ._proposal_families import birth_seeds as portfolio_seeds
        return portfolio_seeds(state, resp, latent, snr, representers, original_seeds)
    geometry = getattr(state, '_offline_proposal_geometry', None)
    if geometry is None:
        return None
    budget = min(state.n_explore_steps, len(resp))
    if budget == 0:
        return []
    features = geometry.features
    labels = resp.argmax(1)
    rows = torch.arange(len(resp), device=resp.device)
    eligible = rows != representers[labels]
    eligible &= torch.bincount(labels, minlength=resp.shape[1])[labels] >= 2
    scores = state.weight_mean(observation_scores(state, resp.shape[1]), snr)
    latent_scores = state.weight_mean(latent, snr)
    losses = -(state.static_factor * scores + state.dynamic_factor * latent_scores)
    candidates = rank_splits(state, resp, features, losses, latent_scores, budget)
    selected, used = [], set()
    for candidate, _selection in candidates:
        parent, seed, included, assignment, _first, second, _details = candidate
        child_rows = included[assignment == 1]
        ranking = child_rows[torch.argsort((features[child_rows] - second).square().sum(1))]
        # Keep the shared proposal's seed when it is eligible (latent windows
        # use their worst member rather than their nearest-center member).
        for row in [seed] + ranking.tolist():
            row = int(row)
            if bool(eligible[row]) and row not in used:
                selected.append(row)
                used.add(row)
                allocation = split_allocation(resp, features, candidate)
                state._offline_birth_allocations[row] = parent, allocation
                break
    # Geometry-based diversity fills unused slots without reintroducing raw
    # observation residuals or coordinate-wise noise amplification.
    safe_representers = representers[labels].clamp(0, len(resp) - 1)
    distance = (features - features[safe_representers]).square().sum(1)
    for row in selected:
        distance = torch.minimum(distance, (features - features[row]).square().sum(1))
        eligible[row] = False
    while len(selected) < budget and bool(eligible.any()):
        row = int(distance.masked_fill(~eligible, -torch.inf).argmax())
        selected.append(row)
        eligible[row] = False
        parent = int(labels[row])
        included = rows[labels == parent]
        center = (resp[included, parent, None] * features[included]).sum(0) / resp[included, parent].sum()
        allocation = ((features - features[row]).square().sum(1)
                      < (features - center).square().sum(1)).to(resp.dtype)
        allocation[row] = 1.
        if bool((allocation[included] == 1).all()):
            other = included[included != row]
            allocation[other[torch.argmin((features[other]-center).square().sum(1))]] = 0.
        state._offline_birth_allocations[row] = parent, allocation
        distance = torch.minimum(distance, (features - features[row]).square().sum(1))
    state._offline_last_birth_seeds = selected
    return selected


