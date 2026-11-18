"""Cheap conditional-transition proposals in one fixed observation coordinate frame.

These ridge regressions are screening proxies for noisy encoded observations,
not replacements for the Gaussian LDS posterior or its sufficient statistics.
Every fitted proposal is still evaluated by the complete native objective.
"""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class TransitionStatistics:
    rows: torch.Tensor
    sources: torch.Tensor
    prefix_xx: torch.Tensor
    prefix_xy: torch.Tensor
    prefix_yy: torch.Tensor
    prefix_mass: torch.Tensor


def features(codes, components=8):
    """Fixed lag-informed basis, independent of variance-only signal screening.

    Lag one and lag two retain temporal directions whose appearances have the
    same marginal distribution, including opposite-sign AR operators. A common
    scale keeps the regressions bounded without changing the fitted LDS basis.
    """
    values = codes.detach() - codes.detach().mean(0)
    values = values / values.square().mean().sqrt().clamp_min(1e-12)
    dimension = values.shape[1]
    if dimension > components:
        gram = values.new_zeros((dimension, dimension))
        for lag in (1, 2):
            if len(values) > lag:
                cross = values[lag:].T @ values[:-lag] / (len(values) - lag)
                gram += cross @ cross.T + cross.T @ cross
        _, basis = torch.linalg.eigh(.5 * (gram + gram.T))
        values = values @ basis[:, -components:]
    return values


def statistics(state, resp, parent, values):
    """Respect retained-state transitions or adjacent physical ticks.

    Cluster-clock pairs follow consecutive observations retained by every
    output's parent. Physical-clock screening uses only adjacent observed ticks:
    a missing interval is never treated as one physical step. Soft endpoint
    responsibility products are explicit proxy weights, not posterior pair
    expectations. Insufficient transitions simply leave this pool unavailable.
    """
    keep = resp[:, parent] > 0
    for output in state.gpmodels:
        retained = torch.zeros(len(resp), dtype=torch.bool, device=resp.device)
        indexes = torch.as_tensor(output[parent].indexes, device=resp.device, dtype=torch.long)
        retained[indexes[(indexes >= 0) & (indexes < len(resp))]] = True
        keep &= retained
    retained = torch.where(keep)[0]
    source, target = retained[:-1], retained[1:]
    if getattr(state, 'dynamics_clock', 'cluster-clock') == 'physical-clock':
        adjacent = target - source == 1
        source, target = source[adjacent], target[adjacent]
    if len(target) < 4:
        return None
    weights = resp[source, parent] * resp[target, parent]
    x, y = values[source], values[target]
    if getattr(state, 'constant_input', False):
        x = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)

    def prefix(value):
        return torch.cat((value.new_zeros((1,) + value.shape[1:]), value.cumsum(0)), dim=0)

    return TransitionStatistics(target, source,
        prefix(weights[:, None, None] * x[:, :, None] * x[:, None, :]),
        prefix(weights[:, None, None] * x[:, :, None] * y[:, None, :]),
        prefix(weights[:, None, None] * y[:, :, None] * y[:, None, :]), prefix(weights))


def interval(stats, start, stop):
    """Sum transitions whose two endpoints lie within an observation interval."""
    left = int(torch.searchsorted(stats.sources, stats.sources.new_tensor(start)))
    right = int(torch.searchsorted(stats.rows, stats.rows.new_tensor(stop), right=True))
    right = max(left, right)
    return tuple(value[right] - value[left] for value in (
        stats.prefix_xx, stats.prefix_xy, stats.prefix_yy, stats.prefix_mass))


def add(first, second):
    return tuple(a + b for a, b in zip(first, second))


def fit_score(stats):
    """Regularized conditional Gaussian fit, including full proxy innovation noise."""
    xx, xy, yy, mass = stats
    dimension, predictors = yy.shape[-1], xx.shape[-1]
    if float(mass) < max(3, predictors + 1):
        return None
    eye = torch.eye(predictors, dtype=xx.dtype, device=xx.device)
    penalty = 1e-3 * xx.diag().mean().clamp_min(1.)
    operator = torch.linalg.solve(xx + penalty * eye, xy)
    scatter = yy - operator.T @ xy - xy.T @ operator + operator.T @ xx @ operator
    scatter = .5 * (scatter + scatter.T)
    # Common feature scale and common floor for both one- and two-operator fits.
    covariance = scatter / mass + .01 * torch.eye(dimension, dtype=xx.dtype, device=xx.device)
    chol, info = torch.linalg.cholesky_ex(covariance)
    if bool(info):
        return None
    logdet = 2 * chol.diag().log().sum()
    trace = torch.trace(torch.cholesky_solve(scatter, chol))
    score = -.5 * (mass * (dimension * math.log(2 * math.pi) + logdet) + trace)
    parameters = predictors * dimension + dimension * (dimension + 1) / 2
    return score, parameters


def candidates(state, resp, values, *, ordered=False, statistics_cache=None):
    """Rank bounded cuts/windows by two-operator conditional-likelihood gain.

    Cached prefix statistics make interval extraction constant cost. Crossing
    transitions are excluded from BOTH fits, keeping their compared data equal.
    At most 17 cut positions and two window widths are screened per parent.
    """
    if state.dynamic_factor == 0:
        return []
    result = []
    labels = resp.argmax(1)
    for parent in range(resp.shape[1]):
        members = torch.where(labels == parent)[0]
        if len(members) < 8:
            continue
        if statistics_cache is None:
            stats = statistics(state, resp, parent, values)
        else:
            if parent not in statistics_cache:
                statistics_cache[parent] = statistics(state, resp, parent, values)
            stats = statistics_cache[parent]
        if stats is None:
            continue
        offsets = torch.linspace(1, len(members) - 2, min(17, len(members)-2),
                                 device=members.device).round().long().unique()
        intervals = [(int(members[offset]), len(resp) - 1) for offset in offsets]
        if not ordered:
            for width in sorted({max(3, len(members)//4), max(3, len(members)//2)}):
                intervals += [(int(members[offset]), int(members[min(int(offset)+width, len(members)-1)]))
                              for offset in offsets if int(offset)+width < len(members)]
        seen = set()
        for start, stop in intervals:
            inside = interval(stats, start, stop)
            outside = add(interval(stats, -1, start - 1), interval(stats, stop + 1, len(resp)))
            pooled = add(inside, outside)
            first, second, single = fit_score(inside), fit_score(outside), fit_score(pooled)
            if first is None or second is None or single is None:
                continue
            gain = first[0] + second[0] - single[0]
            complexity = .5 * single[1] * pooled[-1].clamp_min(1.).log()
            score = gain - complexity
            if not bool(torch.isfinite(score)):
                continue
            allocation = ((torch.arange(len(resp), device=resp.device) >= start)
                          & (torch.arange(len(resp), device=resp.device) <= stop)).to(resp.dtype)
            child = members[allocation[members].bool()]
            other = members[~allocation[members].bool()]
            if not len(child) or not len(other):
                continue
            key = min(tuple(child.tolist()), tuple(other.tolist()))
            if key in seen:
                continue
            seen.add(key)
            center = values[child].mean(0)
            seed = int(child[(values[child] - center).square().sum(1).argmin()])
            result.append(dict(parent=parent, seed=seed, allocation=allocation,
                details=dict(proposal='conditional_dynamics', proxy_gain=float(score),
                    conditional_likelihood_gain=float(gain), proxy_complexity_penalty=float(complexity),
                    ranking_dimension=values.shape[1], window_start=start, window_end=stop,
                    transition_mass=float(pooled[-1]), clock=getattr(state,'dynamics_clock','cluster-clock'))))
    result.sort(key=lambda item: item['details']['proxy_gain'], reverse=True)
    return result


def merge_scores(state, resp, values, pairs, *, statistics_cache=None):
    """One versus two conditional fits on the same common-frame transitions."""
    fitted = {}
    for parent in {index for pair in pairs for index in pair}:
        if statistics_cache is None:
            stats = statistics(state, resp, parent, values)
        else:
            if parent not in statistics_cache:
                statistics_cache[parent] = statistics(state, resp, parent, values)
            stats = statistics_cache[parent]
        if stats is not None:
            total = interval(stats, -1, len(resp))
            score = fit_score(total)
            if score is not None:
                fitted[parent] = total, score
    ranked = []
    for first, second in pairs:
        if first not in fitted or second not in fitted:
            continue
        left, score_left = fitted[first]
        right, score_right = fitted[second]
        pooled = add(left, right)
        score = fit_score(pooled)
        if score is None:
            continue
        gain = score[0] - score_left[0] - score_right[0] + .5 * score[1] * pooled[-1].log()
        if bool(torch.isfinite(gain)):
            ranked.append(dict(clusters=(first, second), details=dict(
                proposal='conditional_dynamics_merge', proxy_gain=float(gain),
                ranking_dimension=values.shape[1])))
    return sorted(ranked, key=lambda item: item['details']['proxy_gain'], reverse=True)
