"""Temporal proposal adapter for the native transactional split/merge engine."""
import torch

from ._ordered_messages import posterior_path


def _pairs(resp):
    pair = resp.new_zeros((len(resp), resp.shape[1], resp.shape[1]))
    pair[1:] = resp[:-1, :, None]*resp[1:, None, :]
    return pair


def moves(state, features):
    """Split at a time boundary, insert the child there; merge only neighbors.

    Split seeds are valid hard paths even when refinement uses soft marginals.
    Splitting an arbitrary soft chain at a fixed time can create skip edges, so
    simply redistributing its pair tensor is not a valid initialization here.
    """
    resp, pair, count = state.resp_last, state.respPair_last, state.M
    path = posterior_path(resp, pair)
    budget = state.n_explore_steps
    if not budget:
        return
    splits, merges = [], []
    if count < min(state.T, state.max_models or state.T):
        for parent in range(count):
            included = torch.where(path == parent)[0]
            if len(included) < 2:
                continue
            local = features[included]
            sums = local.cumsum(0)
            sizes = torch.arange(1, len(local), dtype=local.dtype, device=local.device)
            gains = (sums[:-1].square().sum(1)/sizes
                +(sums[-1]-sums[:-1]).square().sum(1)/(len(local)-sizes)
                -sums[-1].square().sum()/len(local))
            chosen = torch.argsort(gains, descending=True)[:2].tolist()
            # The worst latent member can suggest a boundary even when its
            # appearance is close to the segment center.
            latent = state.weight_mean(state.q_lat_last, state.snr_last)
            worst = int(latent[included, parent].argmin())
            chosen += [worst-1, worst]
            for offset in dict.fromkeys(chosen):
                if 0 <= offset < len(local)-1:
                    cut = int(included[offset+1])
                    splits.append((float(gains[offset]), parent, cut))
    for first in range(count-1):
        left, right = features[path == first], features[path == first+1]
        distance = float((left.mean(0)-right.mean(0)).square().sum())
        merges.append((distance, first))
    splits.sort(reverse=True)
    merges.sort()
    split_slots = budget-min(len(merges), budget//2)
    queue = [('split', item) for item in splits[:split_slots]]
    queue += [('merge', item) for item in merges[:budget-len(queue)]]
    for kind, item in queue:
        if kind == 'split':
            gain, parent, cut = item
            labels = path+(path > parent).long()
            labels[(path == parent) & (torch.arange(state.T, device=path.device) >= cut)] += 1
            proposed = torch.nn.functional.one_hot(labels, count+1).to(resp)
            sources = list(range(count))
            sources.insert(parent+1, parent)
            yield dict(kind='split', clusters=[parent], seed=cut, birth_slot=parent+1,
                selection='temporal_boundary', cut=cut, proxy_gain=gain), proposed, _pairs(proposed), sources
        else:
            _, first = item
            second = first+1
            proposed, proposed_pair = resp.clone(), pair.clone()
            proposed[:, first] += proposed[:, second]
            proposed_pair[:, first, :] += proposed_pair[:, second, :]
            proposed_pair[:, :, first] += proposed_pair[:, :, second]
            sources = [k for k in range(count) if k != second]
            proposed = proposed[:, sources]
            proposed_pair = proposed_pair[:, sources, :][:, :, sources]
            yield dict(kind='merge', clusters=[first, second]), proposed, proposed_pair, sources
