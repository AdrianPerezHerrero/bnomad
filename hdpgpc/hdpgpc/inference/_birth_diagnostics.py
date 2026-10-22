"""Optional, detached snapshots of offline input-warp birth proposals.

These are diagnostics only. No captured value is fed back into inference.
"""

from __future__ import annotations

import numpy as np
import torch

from ..alignment._gp_flow import GPRepresenter


def _array(value):
    return torch.as_tensor(value).detach().cpu().numpy().copy()


def _selected_scores(scores, responsibilities):
    rows = torch.arange(len(responsibilities), device=responsibilities.device)
    owners = torch.argmax(responsibilities, dim=1)
    return scores[rows, owners, 0]


def _sample(owner, x, y, index, parent, newborn, parent_source, birth_source,
            baseline_results, baseline_source):
    results = owner._flow_offline_posteriors
    grid = x[index, :, 0]
    values = y[index, :, 0]
    trace = {'index': int(index), 'grid': _array(grid), 'raw': _array(values)}
    baseline_result = baseline_results.get((index, baseline_source))
    if baseline_result is not None:
        baseline_model = owner.gpmodels[0][baseline_source]
        baseline_gp = GPRepresenter(baseline_model, grid.device)
        baseline_grid = baseline_result.grid_samples.mean(dim=0)[:, 0]
        baseline_data = baseline_gp.log_prob(
            baseline_result.grid_samples, values).mean()
        trace['baseline'] = {
            'mapped_grid': _array(baseline_grid),
            'aligned': np.interp(_array(grid), _array(baseline_grid), _array(values)),
            'scoring_mean': _array(baseline_gp.mean_at(grid[:, None])).reshape(-1),
            'data_score': float(baseline_data.detach().cpu()),
            'warp_penalty': float(baseline_result.kl.detach().cpu()),
            'total_score': float((baseline_data - baseline_result.kl).detach().cpu()),
        }
    # Preserve the alignment reference separately from the scoring GP.
    reference_models = owner.gpmodels[0]
    for label, model, source in (('parent', parent, parent_source),
                                 ('newborn', newborn, birth_source)):
        result = results.get((index, source))
        if result is None:
            continue
        representer = GPRepresenter(model, grid.device)
        reference_index = min(source, len(reference_models) - 1)
        selected_reference = getattr(owner, '_dtw_offline_references', {}).get(source)
        if selected_reference is not None:
            reference_mean = np.interp(
                _array(grid), _array(selected_reference['grid']).reshape(-1),
                _array(selected_reference['values'])[:, 0])
            reference_label = f'series {selected_reference["index"] + 1}'
        else:
            reference = GPRepresenter(reference_models[reference_index], grid.device)
            reference_mean = _array(reference.mean_at(grid[:, None])).reshape(-1)
            reference_label = f'cluster {reference_index}'
        warped_grid = result.grid_samples.mean(dim=0)[:, 0]
        data_score = representer.log_prob(result.grid_samples, values).mean()
        penalty = result.kl
        trace[label] = {
            'mapped_grid': _array(warped_grid),
            'aligned': np.interp(_array(grid), _array(warped_grid), _array(values)),
            'scoring_mean': _array(representer.mean_at(grid[:, None])).reshape(-1),
            'warp_reference_mean': reference_mean,
            'warp_reference_cluster': int(reference_index),
            'warp_reference_label': reference_label,
            'data_score': float(data_score.detach().cpu()),
            'warp_penalty': float(penalty.detach().cpu()),
            'total_score': float((data_score - penalty).detach().cpu()),
        }
    return trace


def capture_seed(owner, x, y, seed, parent_source, provisional_gp,
                 scores, responsibilities, birth_source, baseline_results):
    """Capture a proposed birth even when its initial responsibility is zero."""
    records = getattr(owner, '_birth_diagnostics', None)
    limit = getattr(owner, '_birth_diagnostic_limit', 0)
    if (records is None or len(records) >= limit or owner.n_outputs != 1
            or not getattr(owner.alignment, 'supports_batch_lds_input_warp', False)):
        return None
    parent_gp = owner.gpmodels[0][parent_source]
    parent_scores = scores[:, parent_source, 0]
    new_scores = scores[:, birth_source, 0]
    worst = int(torch.argmin(new_scores - parent_scores))
    chosen = [seed]
    if worst != seed:
        chosen.append(worst)
    record = {
        'iteration': int(getattr(owner, '_birth_iteration', 1)),
        'seed': int(seed),
        'parent_cluster': int(parent_source),
        'newborn_occupancy': float(responsibilities[:, birth_source].sum().item()),
        'phase': 'initial assignment',
        'accepted': False,
    }
    try:
        record['rows'] = [_sample(owner, x, y, i, parent_gp, provisional_gp,
                                  parent_source, birth_source, baseline_results,
                                  parent_source) for i in chosen]
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        record['rows'] = []
        record['diagnostic_error'] = str(error)
    records.append(record)
    return record


def capture_refit(owner, record, x, y, seed, parent_source, birth_source,
                  reorder, before_resp, after_resp, before_q, after_q,
                  models, baseline_results):
    """Replace seed-only curves with the actual provisional fitted scores."""
    if record is None or not torch.any(after_resp[:, reorder == birth_source]):
        return
    source_order = [int(value) for value in reorder]
    parent_slot = source_order.index(parent_source)
    birth_slot = source_order.index(birth_source)
    baseline = _selected_scores(before_q, before_resp)
    proposal = _selected_scores(after_q, after_resp)
    score_change = proposal - baseline
    chosen = [seed]
    for index in torch.argsort(score_change)[:3].tolist():
        if index not in chosen:
            chosen.append(index)
        if len(chosen) == 3:
            break
    try:
        rows = [_sample(owner, x, y, i, models[0][parent_slot],
                        models[0][birth_slot], parent_source, birth_source,
                        baseline_results,
                        int(torch.argmax(before_resp[i]).item()))
                for i in chosen]
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        record['diagnostic_error'] = str(error)
        return
    record.update({
        'phase': 'provisional refit',
        'newborn_occupancy': float(after_resp[:, birth_slot].sum().item()),
        'baseline_emission': float(baseline.sum().item()),
        'proposal_emission': float(proposal.sum().item()),
        'rows': rows,
        'row_emission_change': [float(score_change[i].item()) for i in chosen],
    })
    current_results = owner._flow_offline_posteriors
    baseline_owners = torch.argmax(before_resp, dim=1).tolist()
    proposal_owners = torch.argmax(after_resp, dim=1).tolist()
    try:
        baseline_penalty = sum(
            float(baseline_results[i, source].kl.detach().cpu())
            for i, source in enumerate(baseline_owners))
        proposal_penalty = sum(
            float(current_results[i, source_order[slot]].kl.detach().cpu())
            for i, slot in enumerate(proposal_owners))
    except KeyError as error:
        record['diagnostic_error'] = f'Missing warp score for {error}'
        return
    record['baseline_warp_penalty'] = baseline_penalty
    record['proposal_warp_penalty'] = proposal_penalty
    record['baseline_gp_data'] = record['baseline_emission'] + baseline_penalty
    record['proposal_gp_data'] = record['proposal_emission'] + proposal_penalty
