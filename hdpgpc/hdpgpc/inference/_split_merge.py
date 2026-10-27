"""Transactional offline split/merge search using the native GP/LDS and HDP.

Only proposal construction and orchestration live here. Filtering, smoothing,
parameter learning, warp scores, messages and objective terms retain their
existing implementations in the corresponding components. The legacy optimizer
in _offline is deliberately independent of this opt-in strategy.
"""
from copy import copy, deepcopy
from array import array
from hashlib import blake2b
import math

import torch

from . import _offline as legacy
from ._birth_diagnostics import capture_seed, capture_refit
from ._messages import assignment_messages
from ..hdp import _topology
from . import _ordered_messages


def _partition_key(resp, pair=None):
    """Compact MAP membership key, invariant to any permutation of cluster slots.

    The first observed member names each group. Occupancy totals alone cannot
    distinguish partitions, and slot numbers change when components are merged.
    Deliberately omit parameters/representers so refitting cannot disguise a
    structural cycle. Soft-weight refinement is still unrestricted.
    """
    names = {}
    labels = (resp.argmax(1) if pair is None else
              _ordered_messages.posterior_path(resp, pair)).detach().cpu().tolist()
    canonical = array('q', (names.setdefault(label, len(names)) for label in labels))
    return blake2b(canonical.tobytes(), digest_size=16).digest()


def _convergence(previous, current, accepted, streak):
    """Plateau only on stable full assignments, small gains and no structure move."""
    before, after = float(previous._split_merge_score), float(current._split_merge_score)
    tolerance = 1e-6 + getattr(current, 'split_merge_score_rtol', 1e-3) * max(1., abs(before), abs(after))
    finite = math.isfinite(before) and math.isfinite(after)
    assignments_stable = (not accepted
        and previous.resp_last.shape == current.resp_last.shape
        and previous.respPair_last.shape == current.respPair_last.shape
        and bool(torch.allclose(previous.resp_last, current.resp_last, atol=1e-6, rtol=0))
        and bool(torch.allclose(previous.respPair_last, current.respPair_last, atol=1e-6, rtol=0)))
    small_change = finite and abs(after - before) <= tolerance
    streak = streak + 1 if assignments_stable and small_change else 0
    return streak, {'assignments_stable': assignments_stable,
                    'score_change': after - before, 'score_tolerance': tolerance,
                    'stable_iterations': streak,
                    'patience': getattr(current, 'split_merge_stability_patience', 3)}


def _proposal_features(state):
    """Fixed, cluster-independent coordinates for inexpensive split ranking.

    Affine decoders supply their calibrated analysis map; never use a cluster's
    learned C or noise to standardize away its unexplained variability. A local
    RNG provides a common compact projection without touching inference's RNG.
    """
    from ..emissions._fixed_observation import AffineGaussianObservation

    values = state._split_merge_y.detach()
    outputs = []
    for output, models in enumerate(state.gpmodels):
        model = getattr(models[0], 'fixed_observation_model', None)
        if getattr(models[0], 'factor_analysis', False):
            # Ranking coordinates must not drift with one cluster's learned
            # loading/noise; retain the common affine initialization here.
            model = state.fixed_observation_model
        data = values[:, :, output]
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


def _split_gain(features, weights, assignment):
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
    # Features have global unit variance: use the same floor in both models.
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


def _latent_window(state, parent, included, latent_scores):
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
        retained = torch.zeros(state.T, dtype=torch.bool, device=included.device)
        if len(indexes) > 1:
            retained[torch.as_tensor(indexes[1:], device=included.device, dtype=torch.long)] = True
        eligible &= retained[included]
    rows = included[eligible]
    if len(rows) < 2:
        return None
    weights, scores = state.resp_last[rows, parent], latent_scores[rows, parent]
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


def _warps(work, previous, sources):
    """Reuse DTW posteriors with unchanged observation references/hyperparameters."""
    x, y, resp = work.x_train, work._split_merge_y, work.resp_last
    if not (work.warp and getattr(work.alignment, 'supports_batch_lds_input_warp', False)):
        return work.warp_batch_by_resp_amtgp_cached(x, y, resp, work.f_ind_old)
    cached = getattr(previous, '_flow_offline_posteriors', {})
    references = getattr(previous, '_dtw_offline_references', {})
    results, refs, missing = {}, {}, []
    for cluster, source in enumerate(sources):
        ref = references.get(source)
        if (ref is not None and ref['index'] == int(work.f_ind_old[cluster])
                and all((row, source) in cached for row in range(work.T))):
            refs[cluster] = ref
            results.update({(row, cluster): cached[row, source] for row in range(work.T)})
        else:
            missing.append(cluster)
    if missing:
        probe = copy(work)
        probe.gpmodels = [[output[k] for k in missing] for output in work.gpmodels]
        probe.wp_sys = [[output[k] for k in missing] for output in work.wp_sys]
        probe.warp_batch_by_resp_amtgp_cached(x, y, resp[:, missing], work.f_ind_old[missing])
        for local, cluster in enumerate(missing):
            refs[cluster] = probe._dtw_offline_references[local]
            results.update({(row, cluster): probe._flow_offline_posteriors[row, local]
                            for row in range(work.T)})
    work._flow_offline_posteriors, work._dtw_offline_references = results, refs
    offsets = torch.stack([torch.stack([results[row, k].mean_displacement
        .expand(-1, work.n_outputs) for k in range(work.M)], dim=-1)
        for row in range(work.T)])
    return y.unsqueeze(-1).expand(-1, -1, -1, work.M), offsets, y.new_zeros((work.T, work.M, work.n_outputs))


def _score(work):
    """Optimize native HDP factors and score the entire batch, entropy once."""
    resp, pair, count = work.resp_last, work.respPair_last, work.M
    work.startStateCount_last = resp[0]
    work.transStateCount_last = work.hdp.transition_counts(pair)
    if work.hmm_switch:
        work.reinit_global_params(count, work.transStateCount_last, work.startStateCount_last)
        for _ in range(2):
            work.transTheta, work.startTheta = work._calcThetaFull(
                work.transStateCount_last, work.startStateCount_last, count + 1)
            work.rho, work.omega = work.find_optimum_rhoOmega()
        work.transTheta, work.startTheta = work._calcThetaFull(
            work.transStateCount_last, work.startStateCount_last, count + 1)
        work.trans_A = work.hdp.stored_log_transitions(count, work.transTheta)
    else:
        work.startStateCount_last = resp.sum(0)
        work.hdp.update_iid_weights(resp)
        work.trans_A = work.hdp.iid_log_transitions(count)
    emission, other, parts = work.compute_q_elbo(
        resp, pair, work.weight_mean(work.q_last, work.snr_last),
        work.weight_mean(work.q_lat_last, work.snr_last), work.gpmodels, count,
        snr=work.snr_last, post=False, verb=False, return_components=True)
    entropy = torch.as_tensor(work.calcELBO_NonlinearTerms(
        work.cond_to_numpy(work.cond_cpu(resp)), work.cond_to_numpy(work.cond_cpu(pair))),
        device=resp.device, dtype=resp.dtype)
    work._split_merge_score = emission + other + entropy
    work.elbo_last = work._split_merge_score - parts['lds'] + parts['lds_raw']
    parts['chain_entropy'] = entropy
    work._split_merge_parts = {key: float(value) for key, value in parts.items()}


def _stage(previous, resp, pair, sources, *, force=False, initial=False, birth_seed=None, birth_slot=None):
    """Fit an isolated proposal, copying posterior state only where necessary."""
    ordered = _topology.enabled(previous)
    if ordered:
        path = _ordered_messages.posterior_path(resp, pair)
        ownership = torch.nn.functional.one_hot(path, resp.shape[1]).to(resp)
        if not bool((ownership.sum(0) > 0).all()) or not bool((resp.sum(0) >= 1e-8).all()):
            return None
    else:
        ownership = resp
        if not bool(legacy._cluster_occupied(previous, resp).all()):
            return None
    work = copy(previous)
    # Global/list/cache writes in native helpers must stay in this transaction.
    work.__dict__ = {name: value.copy() if isinstance(value, (list, dict)) else value
                     for name, value in previous.__dict__.items()}
    work.M = resp.shape[1]
    work.observation_policies = deepcopy(previous.observation_policies)
    work.transition_policies = deepcopy(previous.transition_policies)
    work.resp_last, work.respPair_last = resp, pair
    sources = [int(source) for source in sources]
    # Advanced indexing already allocates independent score arrays.
    work.q_last = previous.q_last[:, sources]
    work.q_lat_last = previous.q_lat_last[:, sources]
    work.snr_last = previous.snr_last[:, sources]
    preferred = previous.f_ind_old[sources].clone()
    if work.offline_representer_selection == 'sorted':
        preferred.fill_(-1)
    if birth_seed is not None:
        preferred[-1 if birth_slot is None else birth_slot] = birth_seed
    scores = work.weight_mean(work.q_last, work.snr_last)
    slots = torch.arange(work.M, device=resp.device)
    birth_source = (work.M-1 if birth_slot is None else birth_slot) if birth_seed is not None else None
    try:
        work.f_ind_old = legacy._fixed_representers(
            ownership, scores, preferred, slots, birth_source=birth_source)
    except RuntimeError as error:
        if str(error) != 'No distinct representative remains for this cluster':
            raise
        # Fixed references can outlive membership changes. An earlier slot can
        # then reserve a later singleton's only member. Repair only this failed
        # selection: retain owned seeds, and reselect stale ones from disjoint
        # MAP groups. Every occupied group consequently has an available beat.
        owners = ownership.argmax(1)
        valid = (preferred >= 0) & (preferred < len(resp))
        valid &= owners[preferred.clamp(0, len(resp) - 1)] == slots
        repaired = torch.where(valid, preferred, preferred.new_full((), -1))
        work.f_ind_old = legacy._fixed_representers(
            ownership, scores, repaired, slots, birth_source=birth_source)
        if work.verbose:
            changed_references = torch.where(work.f_ind_old != preferred)[0].tolist()
            print(f'Representative collision repaired: clusters={changed_references}; '
                  f'representatives={work.f_ind_old.tolist()}', flush=True)
    pooled = work.observation_scope == 'shared' or bool(work.transition_policies)
    sampled_warp = work.warp and getattr(work.alignment, 'uses_posterior_flow', False)
    changed = [initial or force or pooled or sampled_warp
               or not previous.gpmodels[0][source].indexes
               or not torch.equal(resp[:, k], previous.resp_last[:, source])
               or (work.warp and int(work.f_ind_old[k]) != int(previous.f_ind_old[source]))
               for k, source in enumerate(sources)]
    owned = bool(work.observation_policies or work.transition_policies)
    work.gpmodels = [[work.gpmodel_deepcopy(output[source]) if changed[k] or owned else output[source]
                      for k, source in enumerate(sources)] for output in previous.gpmodels]
    legacy._synchronize_cluster_slots(work, sources, work.gpmodels)
    if work.warp:
        # Separate copies also for two children inheriting the same source slot.
        work.wp_sys = [[deepcopy(warper) for warper in output] for output in work.wp_sys]
    for output, models in enumerate(work.gpmodels):
        for cluster, gp in enumerate(models):
            if changed[cluster] or owned:
                source = previous.gpmodels[output][sources[cluster]]
                if hasattr(source, 'fixed_observation_model'):
                    gp.fixed_observation_model = source.fixed_observation_model
                    gp.gp.fixed_observation_model = source.fixed_observation_model
                if work.observation_policies:
                    work.observation_policies[output].attach(gp)
                if work.transition_policies:
                    work.transition_policies[output].attach(gp)
    work.y_w, work.x_w, work.liks = _warps(work, previous, sources)
    for output, models in enumerate(work.gpmodels):
        for cluster, gp in enumerate(models):
            if not changed[cluster]:
                continue
            if initial and work.share_gp and output > 0:
                gp = work.gpmodel_deepcopy(work.gpmodels[0][cluster])
                models[cluster] = gp
                if work.observation_policies:
                    work.observation_policies[output].attach(gp)
                if work.transition_policies:
                    work.transition_policies[output].attach(gp)
            gp.reinit_LDS(save_last=False, warm_start=not initial)
            gp.reinit_GP(save_last=False)
            observations = work.y_w[:, :, [output], cluster]
            work.q_last[:, cluster, output], work.q_lat_last[:, cluster, output] = legacy._fit_emission(
                work, gp, work.x_train, observations, resp[:, cluster], work.x_w, output, cluster,
                q=work.q_last[:, cluster, output], q_lat=work.q_lat_last[:, cluster, output],
                snr=work.snr_norm[:, output])
            work.q_last[:, cluster, output] = legacy._flow_score(
                work, gp, work.q_last[:, cluster, output], work._split_merge_y, output, cluster
            ) + work.liks[:, cluster, output]
            work.snr_last[:, cluster, output] = work.compute_snr(observations[:, :, 0], gp)
    if pooled:
        work.update_shared_observation(resp)
        for output, models in enumerate(work.gpmodels):
            for cluster, gp in enumerate(models):
                observations = work.y_w[:, :, [output], cluster]
                grids = legacy._flow_inputs(work, work.x_train, work.x_w, output, cluster)
                work.q_last[:, cluster, output] = legacy._flow_score(work, gp,
                    gp.compute_sq_err_all(grids, observations), work._split_merge_y, output, cluster
                ) + work.liks[:, cluster, output]
                work.q_lat_last[:, cluster, output] = gp.compute_q_lat_all(grids)
    work.snr_norm = work.normalize_snr(work.snr_last)
    work.y_train = work.select_assigned_warp(work.y_w, resp)
    _score(work)
    if not math.isfinite(float(work._split_merge_score)):
        raise FloatingPointError('Nonfinite split/merge objective')
    return work


def _refine(state, steps):
    """Reuse native messages; keep the best fully scored iterate of this branch."""
    best = state
    for _ in range(steps):
        potential = (state.static_factor * state.weight_mean(state.q_last, state.snr_last)
                     + state.dynamic_factor * state.weight_mean(state.q_lat_last, state.snr_last))
        evidence, _ = state.LogLik(potential)
        start = (state.hdp.expected_log_start(state.M, state.startTheta) if state.hmm_switch
                 else state.hdp.expected_log_iid_weights(state.M))
        state.fmsg = None  # full batch, never the online tail-message cache
        if _topology.enabled(state):
            resp, pair, _ = _ordered_messages.marginals(start, state.trans_A, evidence,
                                               hard=state.hard_assignments)
        else:
            alpha, beta, scales = assignment_messages(state, start, state.trans_A, evidence)
            resp = state._safe_exp(torch.log(alpha) + torch.log(beta))
            if state.hard_assignments:
                pair = resp.new_zeros((state.T, state.M, state.M))
                pair[1:] = resp[:-1, :, None] * resp[1:, None, :]
            else:
                pair = state._safe_exp(state.coupled_state_coef(alpha, beta, state.trans_A, evidence, scales))
        candidate = _stage(state, resp, pair, range(state.M), force=state.lds_update == 'conjugate_batch')
        if candidate is None:
            break
        if legacy._reallocation_improves(best._split_merge_score, candidate._split_merge_score):
            best = candidate
        converged = legacy._offline_converged(
            state.resp_last, resp, state.respPair_last, pair,
            state._split_merge_score, candidate._split_merge_score,
            reallocation_accepted=False, birth_accepted=False)
        state = candidate
        if converged:
            break
    return best


def _moves(state, features, split_features=None):
    """Rank cheap group splits and close-cluster merges; never fit here."""
    if _topology.enabled(state):
        from ._ordered_moves import moves
        yield from moves(state, features if split_features is None else split_features)
        return
    resp, pair, count = state.resp_last, state.respPair_last, state.M
    budget = state.n_explore_steps
    if budget == 0:
        return
    if split_features is None:
        split_features = features
    labels, mass = resp.argmax(1), resp.sum(0)
    centers = resp.T @ features / mass[:, None]
    merges = []
    if count > 1:
        distances = torch.cdist(centers, centers).square()
        merges = sorted(((float(distances[a, b]), a, b)
                         for a in range(count) for b in range(a + 1, count)))[:budget]
    split_budget = budget - min(len(merges), budget // 2)
    splits = []
    if count < state.T and (state.max_models is None or count < state.max_models):
        latent_scores = state.weight_mean(state.q_lat_last, state.snr_last)
        losses = -(state.static_factor * state.weight_mean(state.q_last, state.snr_last)
                   + state.dynamic_factor * latent_scores)
        members = {k: torch.where(labels == k)[0] for k in range(count)}
        parents = [k for k in range(count) if len(members[k]) >= 2]
        parent_loss = {k: float((resp[:, k] * losses[:, k]).sum()) for k in parents}
        candidates, windows = [], []
        used = set()
        for parent in parents:
            included = members[parent]
            window = _latent_window(state, parent, included, latent_scores)
            if window is None:
                continue
            seed, assignment, details = window
            child_rows = included[assignment == 1]
            key = (parent, min(tuple(child_rows.tolist()), tuple(included[assignment == 0].tolist())))
            local, weights = split_features[included], resp[included, parent]
            proxy = _split_gain(local, weights, assignment)
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
                details = _split_gain(local, local_weights, assignment)
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
    # Reserve merge slots when the exploration budget permits both move types.
    queue = [('split', item) for item in splits]
    queue += [('merge', item[1:]) for item in merges[:budget - len(queue)]]
    for kind, item in queue:
        sources = list(range(count))
        proposed = resp.clone()
        if kind == 'merge':
            first, second = item
            proposed[:, first] += proposed[:, second]
            proposed_pair = pair.clone()
            proposed_pair[:, first, :] += proposed_pair[:, second, :]
            proposed_pair[:, :, first] += proposed_pair[:, :, second]
            sources.remove(second)
            proposed, proposed_pair = proposed[:, sources], proposed_pair[:, sources, :][:, :, sources]
            description = {'kind': kind, 'clusters': [first, second]}
        else:
            candidate, selection = item
            parent, seed, included, assignment, first, second, details = candidate
            if details.get('proposal') == 'worst_q_lat_window':
                rows = torch.arange(state.T, device=resp.device)
                allocation = ((rows >= details['window_start'])
                              & (rows <= details['window_end'])).to(resp.dtype)
            else:
                allocation = ((split_features - second).square().sum(1)
                              < (split_features - first).square().sum(1)).to(resp.dtype)
            allocation[included] = assignment.to(resp.dtype)
            child = resp[:, parent] * allocation
            proposed[:, parent] -= child
            proposed = torch.cat((proposed, child[:, None]), 1)
            # Push the existing pair posterior through the time-dependent split.
            # Preserve all unaffected pairs and both adjacent-time marginals.
            before = torch.cat((allocation.new_zeros(1), allocation[:-1]))
            left = resp.new_ones((state.T, count)); left[:, parent] = 1 - before
            right = resp.new_ones((state.T, count)); right[:, parent] = 1 - allocation
            proposed_pair = pair.new_zeros((state.T, count + 1, count + 1))
            proposed_pair[:, :count, :count] = pair * left[:, :, None] * right[:, None, :]
            proposed_pair[:, count, :count] = pair[:, parent, :] * before[:, None] * right
            proposed_pair[:, :count, count] = pair[:, :, parent] * allocation[:, None] * left
            proposed_pair[:, count, count] = pair[:, parent, parent] * before * allocation
            sources.append(parent)
            description = {'kind': kind, 'clusters': [parent], 'seed': seed,
                           'selection': selection, **details}
        yield description, proposed, proposed_pair, sources


def fit(self, x_trains, y_trains, it_limit=None, warp=False):
    """Separate offline optimizer; preserve the public state/score/plot layouts."""
    state = copy(self)
    ordered = _topology.enabled(state)
    if ordered and getattr(self, '_birth_diagnostic_limit', 0):
        raise ValueError('Birth-warp diagnostics do not yet support inserted ordered stages')
    state.__dict__ = self.__dict__.copy()
    state.warp = state.alignment.enabled(warp)
    if state.reduce_outputs:
        y_trains = state.reduce_num_outputs(y_trains)
    state.compute_snr_ini(y_trains)
    state.x_train = state.cond_cuda(state.cond_to_torch(x_trains)).to(dtype=torch.float64)
    state._split_merge_y = state.cond_cuda(state.cond_to_torch(y_trains)).to(dtype=torch.float64)
    state.T = len(y_trains)
    if state.T == 0:
        raise ValueError('split_merge requires at least one observation')
    records = list(getattr(self, '_birth_diagnostics', ()))
    state.y = state._split_merge_y
    state._warp_cache_full = {}
    state._flow_offline_posteriors = {}
    state._dtw_offline_references = {}
    state.M = min(state.M, state.T, state.max_models or state.T)
    labels = torch.arange(state.T, device=state.device) * state.M // state.T
    resp = state._split_merge_y.new_zeros((state.T, state.M))
    resp.scatter_(1, labels[:, None], 1.)
    pair = resp.new_zeros((state.T, state.M, state.M))
    pair[1:] = resp[:-1, :, None] * resp[1:, None, :]
    state.resp_last, state.respPair_last = resp, pair
    state.q_last = resp.new_zeros((state.T, state.M, state.n_outputs))
    state.q_lat_last, state.snr_last = state.q_last.clone(), torch.ones_like(state.q_last)
    if state.reestimate_initial_params:
        state = copy(state)
        state.gpmodels = [[state.gpmodel_deepcopy(gp) for gp in output] for output in state.gpmodels]
        state.redefine_default(state.x_train, state._split_merge_y, resp)
    state = _stage(state, resp, pair, range(state.M), initial=True)
    features = state._split_merge_y.reshape(state.T, -1)
    features = (features - features.mean(0)) / features.std(0, unbiased=False).clamp_min(1e-8)
    split_features = _proposal_features(state)
    history, iteration, stable_iterations = [], 0, 0
    visited = {_partition_key(state.resp_last, state.respPair_last if ordered else None): 0}
    while True:
        baseline = _refine(state, state.split_merge_refine_steps)
        visited.setdefault(_partition_key(baseline.resp_last, baseline.respPair_last if ordered else None), iteration)
        best, chosen, chosen_record = baseline, None, None
        for description, resp, pair, sources in _moves(baseline, features, split_features):
            previous_iteration = visited.get(_partition_key(resp, pair if ordered else None))
            if previous_iteration is not None:
                history.append(dict(description, iteration=iteration + 1, accepted=False,
                                    reason='partition cycle before refit',
                                    previous_iteration=previous_iteration))
                if self.verbose:
                    print(f'Split/merge rejected {description}: partition already retained '
                          f'at iteration {previous_iteration} (before refit)', flush=True)
                continue
            record = None
            try:
                proposal = _stage(baseline, resp, pair, sources, birth_seed=description.get('seed'),
                                  birth_slot=description.get('birth_slot'))
                if proposal is None:
                    history.append(dict(description, iteration=iteration + 1, accepted=False,
                                        reason='proposal has an empty MAP component'))
                    continue
                proposal = _refine(proposal, state.split_merge_refine_steps)
            except (torch.linalg.LinAlgError, FloatingPointError) as error:
                history.append(dict(description, iteration=iteration + 1, accepted=False, error=str(error)))
                if self.verbose:
                    print(f'Split/merge rejected {description}: {error}', flush=True)
                continue
            trace = dict(description, iteration=iteration + 1, accepted=False,
                         baseline_score=float(baseline._split_merge_score),
                         proposal_score=float(proposal._split_merge_score),
                         baseline_elbo=float(baseline.elbo_last), proposal_elbo=float(proposal.elbo_last),
                         baseline_parts=baseline._split_merge_parts, proposal_parts=proposal._split_merge_parts)
            history.append(trace)
            previous_iteration = visited.get(_partition_key(proposal.resp_last, proposal.respPair_last if ordered else None))
            if previous_iteration is not None:
                trace.update(reason='partition cycle after refinement',
                             previous_iteration=previous_iteration)
                if self.verbose:
                    print(f'Split/merge rejected {description}: partition already retained '
                          f'at iteration {previous_iteration} (after refinement)', flush=True)
                continue
            if description['kind'] == 'split' and not ordered:
                diagnostic = copy(baseline)
                diagnostic._birth_diagnostics = records
                diagnostic._birth_iteration = iteration + 1
                diagnostic._flow_offline_posteriors = getattr(proposal, '_flow_offline_posteriors', {})
                diagnostic._dtw_offline_references = getattr(proposal, '_dtw_offline_references', {})
                old_results = getattr(baseline, '_flow_offline_posteriors', {})
                parent, seed = description['clusters'][0], description['seed']
                record = capture_seed(diagnostic, state.x_train, state._split_merge_y,
                    seed=seed, parent_source=parent, provisional_gp=proposal.gpmodels[0][-1],
                    scores=proposal.q_last, responsibilities=proposal.resp_last,
                    birth_source=baseline.M, baseline_results=old_results)
                capture_refit(diagnostic, record, state.x_train, state._split_merge_y,
                    seed, parent, baseline.M, torch.arange(proposal.M, device=proposal.resp_last.device),
                    baseline.resp_last, proposal.resp_last, baseline.q_last, proposal.q_last,
                    proposal.gpmodels, old_results)
                if record is not None:
                    record.update(decision_baseline_score=trace['baseline_score'],
                                  decision_proposal_score=trace['proposal_score'])
            if self.verbose:
                print(f'Split/merge proposal {description}: delta_score='
                      f'{float(proposal._split_merge_score - baseline._split_merge_score):+.6g}; '
                      f'baseline={baseline._split_merge_parts}; proposal={proposal._split_merge_parts}', flush=True)
            if legacy._reallocation_improves(best._split_merge_score, proposal._split_merge_score):
                best, chosen, chosen_record = proposal, trace, record
        if chosen is not None:
            chosen['accepted'] = True
            if chosen_record is not None:
                chosen_record['accepted'] = True
        iteration += 1
        visited.setdefault(_partition_key(best.resp_last, best.respPair_last if ordered else None), iteration)
        best.train_elbo = state.train_elbo + [best.elbo_last.detach().clone()]
        labels = (_ordered_messages.posterior_path(best.resp_last, best.respPair_last)
                  if ordered else best.resp_last.argmax(1))
        best.resp_assigned = state.resp_assigned + [labels]
        best.q = state.q + [best.q_last]
        stable_iterations, convergence = _convergence(state, best, chosen is not None, stable_iterations)
        if self.progress:
            message = ''
            if chosen:
                message = (f'; split cluster {chosen["clusters"][0]} into '
                           f'{chosen["clusters"][0]} and {chosen.get("birth_slot", best.M - 1)}' if chosen['kind'] == 'split'
                           else f'; merged clusters {chosen["clusters"]}')
            print(f'Iteration {iteration}: responsibilities='
                  f'{[round(value, 2) for value in best.resp_last.sum(0).tolist()]}{message}', flush=True)
        if self.verbose:
            print(f'Iteration {iteration}: score={float(best._split_merge_score):.6g}; '
                  f'raw ELBO={float(best.elbo_last):.6g}; parts={best._split_merge_parts}', flush=True)
            print(f'Stopping check: assignments_stable={convergence["assignments_stable"]}; '
                  f'delta_score={convergence["score_change"]:+.6g}; '
                  f'tolerance={convergence["score_tolerance"]:.6g}; '
                  f'stable iterations={stable_iterations}/{convergence["patience"]}', flush=True)
        stable = stable_iterations >= convergence['patience']
        state = best
        if stable or (it_limit is not None and iteration >= it_limit):
            reason = 'stable_assignments_and_small_score_change' if stable else 'iteration_limit'
            state._split_merge_convergence = dict(convergence, reason=reason, iteration=iteration)
            if self.progress:
                print(f'Stopping split/merge: {reason} at iteration {iteration}; '
                      f'delta_score={convergence["score_change"]:+.6g}; '
                      f'tolerance={convergence["score_tolerance"]:.6g}', flush=True)
            break
    state._split_merge_history = history
    state._birth_diagnostics = records
    self.__dict__.update(state.__dict__)
