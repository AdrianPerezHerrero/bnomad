"""Budgeted, label-free prior screening using short native fits.

No state, parameter update, assignment rule or objective in bnomad is changed.
Clip fits are independent; distant clips never create artificial transitions.
"""
from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
import io
import hashlib
import json
import math
import random
import time

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from .. import HDPGPC
from ..dynamics import estimate_lds_noise_priors
from ._predictive import array, gap_diagnostics, validate_predictive


@dataclass(frozen=True)
class NoisePriorScreenConfig:
    """Small work budgets, not new inference defaults.

    Wall time is checked between fits; an in-progress native fit is allowed to
    finish. Representation preparation is outside this budget. Both priors use
    the bounded range [1e-3, 10.0] by default. The fit budget includes grid pairs,
    local refinement and repeat checks across clips/seeds.
    """
    train_size: int = 128
    validation_size: int = 32
    blocks: int = 5
    iterations: int = 2
    exploration_steps: int = 1
    grid_points: int = 5
    max_fits: int = 60
    seconds: float = 120.
    sigma_range: tuple[float, float] | None = None
    gamma_range: tuple[float, float] | None = None
    seed: int = 42

    def __post_init__(self):
        for name, minimum in (('train_size', 6), ('validation_size', 1), ('blocks', 1),
                              ('iterations', 1), ('exploration_steps', 1),
                              ('grid_points', 2), ('max_fits', 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if not math.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError('seconds must be finite and positive')
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32-1:
            raise ValueError('seed must be an integer in [0, 2**32-2]')
        for name in ('sigma_range', 'gamma_range'):
            bounds = getattr(self, name)
            if bounds is not None and (len(bounds) != 2 or not np.isfinite(bounds).all()
                                      or not 0 < bounds[0] <= bounds[1]):
                raise ValueError(f'{name} must contain two ordered, finite positive values')
            if bounds is not None:
                object.__setattr__(self, name, tuple(float(value) for value in bounds))
        object.__setattr__(self, 'seconds', float(self.seconds))


@contextmanager
def _runtime():
    """Native fits seed process RNGs; restore the caller's numerical settings."""
    py_state, np_state = random.getstate(), np.random.get_state()
    threads, dtype = torch.get_num_threads(), torch.get_default_dtype()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices), threadpool_limits(limits=1):
            torch.set_num_threads(1)
            yield
    finally:
        random.setstate(py_state); np.random.set_state(np_state)
        torch.set_num_threads(threads); torch.set_default_dtype(dtype)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def _clips(n, options):
    if n < 8:
        raise ValueError('Prior screening needs at least eight observations')
    validation = min(options.validation_size, max(1, n//4))
    train = min(options.train_size, n-validation)
    if train < 6:
        raise ValueError('Prior screening needs at least eight observations')
    width = train+validation
    starts = np.unique(np.linspace(0, n-width, min(options.blocks, n-width+1), dtype=int))
    return [(int(start), int(start+train), int(start+width)) for start in starts]


def _scientific_signature(config):
    """Allow variance centres to vary, while enforcing the frozen experiment."""
    # The checkpoint serializer preserves object aliases through graph IDs.
    # Those IDs are not scientific settings and can change when a noise matrix
    # is replaced. Compare plain dataclass values instead.
    def shallow(value):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    data = shallow(config)
    for section in ('emission', 'dynamics', 'observation'):
        data[section] = shallow(data[section])
    for section, policy in (('observation', 'Sigma_policy'), ('dynamics', 'Gamma_policy')):
        if data[section][policy] is not None:
            data[section][policy] = shallow(data[section][policy])
    data['emission'].pop('ini_sigma', None)
    data['dynamics'].pop('ini_gamma', None)
    observation = data['observation']
    for name in ('Sigma', 'fixed_latent_covariance', 'pca_variances'):
        observation.pop(name, None)
    if observation.get('Sigma_policy') is not None:
        observation['Sigma_policy'].pop('value', None)
    if data['dynamics'].get('Gamma_policy') is not None:
        data['dynamics']['Gamma_policy'].pop('value', None)
    def encode(value):
        if is_dataclass(value):
            return shallow(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if hasattr(value, 'get_params'):
            return {'type': type(value).__module__+'.'+type(value).__qualname__,
                    'parameters': value.get_params(deep=False)}
        raise TypeError(f'Unsupported scientific configuration value: {type(value).__name__}')
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=encode).encode()).hexdigest()


def _calibrate(values, grid, config, clips, sigma, gamma, seed):
    """Conservative proposal centres; moment diagnostics never certify clusters."""
    prototype = config.build_model().gpmodels
    sigma_ratios, gamma_ratios, diagnostics = [], [], []
    active_gamma = False
    for output, models in enumerate(prototype):
        gp = models[0]
        Q, R, C = array(gp.Gamma[-1]), array(gp.Sigma[-1]), array(gp.C[-1])
        independent = getattr(gp, '_independent_factors', lambda: False)()
        active_gamma |= bool(np.count_nonzero(Q)) and not independent
        inverse = np.linalg.pinv(C)
        measured = np.diag(inverse @ R @ inverse.T)
        fixed = getattr(gp, 'fixed_observation_model', None)
        for start, end, _ in clips:
            segment = values[start:end, :, output]
            if fixed is not None:
                codes = (segment-fixed.offset) @ inverse.T
            else:
                projected = [array(gp.project_observation_to_basis(
                    gp.cond_to_cuda(gp.cond_to_torch(y)).reshape(-1, 1),
                    gp.cond_to_cuda(gp.cond_to_torch(grid)), allow_warped=True)).reshape(-1)
                    for y in segment]
                codes = np.asarray(projected) @ inverse.T
            if len(codes) < 18:
                continue
            estimate = estimate_lds_noise_priors(codes, max_beats=len(codes),
                segment_length=len(codes), random_state=seed, require_positive_components=False)
            r = estimate.observation_diagonal.reshape(-1)
            q = estimate.process_diagonal.reshape(-1)
            valid_r = (r > 0) & (measured > 0)
            valid_q = (q > 0) & (np.diag(Q) > 0)
            if valid_r.any():
                sigma_ratios.extend((r[valid_r]/measured[valid_r]).tolist())
            if valid_q.any():
                gamma_ratios.extend((q[valid_q]/np.diag(Q)[valid_q]).tolist())
            diagnostics.append({'output': output, 'start': start, 'end': end,
                'observation_moment': estimate.observation, 'process_moment': estimate.process,
                'positive_observation_fraction': float(np.mean(r > 0)),
                'positive_process_fraction': float(np.mean(q > 0))})
    def centre(current, ratios):
        # Restrict an unreliable one-regime moment proposal to sqrt(10) from the
        # supplied scale. Always test the supplied pair as a separate anchor.
        ratio = float(np.median(ratios)) if ratios else 1.
        return current*math.sqrt(float(np.clip(ratio, .1, 10.)))
    return centre(sigma, sigma_ratios), centre(gamma, gamma_ratios), {
        'method': 'Identity-random-walk moments on independent contiguous clips; conservative scale proposal only',
        'clips': diagnostics, 'active_gamma': active_gamma,
        'observation_ratio_median': float(np.median(sigma_ratios)) if sigma_ratios else None,
        'process_ratio_median': float(np.median(gamma_ratios)) if gamma_ratios else None,
        'caveat': 'Regime changes, colored noise and nonidentity dynamics can bias these moments'}


def screen_noise_priors(observations, config_factory, *, grid, sigma_prior, gamma_prior,
                        options=None, progress=None):
    """Return empirical prior candidates and a recommendation, without a final fit.

    ``config_factory(sigma, gamma)`` must return a fresh HDPGPCConfig using the
    same frozen representation and scientific settings for every pair. There is
    deliberately no reference-label input. Validation uses frozen-parameter MAP
    filtering among represented components (no validation births). Nonlinear
    input alignment is rejected because its normalized held-out density is not
    supplied by this screener. No parameters are changed on any caller model.
    """
    options = NoisePriorScreenConfig() if options is None else options
    if not isinstance(options, NoisePriorScreenConfig):
        raise TypeError('options must be NoisePriorScreenConfig')
    values = np.asarray(observations, dtype=np.float64)
    if values.ndim == 2:
        values = values[:, :, None]
    coordinates = np.asarray(grid, dtype=np.float64).reshape(-1, 1)
    if (values.ndim != 3 or values.shape[1] != len(coordinates) or values.shape[2] < 1
            or not np.isfinite(values).all() or not np.isfinite(coordinates).all()):
        raise ValueError('Finite observations must have shape (rows, grid points[, outputs])')
    if not all(math.isfinite(x) and x > 0 for x in (sigma_prior, gamma_prior)):
        raise ValueError('Prior centres must be finite and positive')
    clips = _clips(len(values), options)
    base = config_factory(float(sigma_prior), float(gamma_prior))
    if base.alignment.policy not in ('none', None) or base.inference.warp:
        raise ValueError('Prior screening requires alignment=none; held-out warp densities are not implemented')
    if base.numerics.cuda:
        raise ValueError('Prior screening currently requires CPU inference')
    with _runtime():
        return _screen(values, coordinates, config_factory, sigma_prior, gamma_prior,
                       base, clips, options, progress)


def _screen(values, grid, factory, sigma, gamma, base, clips, options, progress):
    started = time.perf_counter()
    sigma0, gamma0, calibration = _calibrate(values, grid, base, clips, sigma, gamma, options.seed)
    ranges = [options.sigma_range or (1e-3, 10.),
              options.gamma_range or (1e-3, 10.)]
    axes = [np.geomspace(*bounds, options.grid_points) if bounds[0] != bounds[1]
            else np.asarray([bounds[0]]) for bounds in ranges]
    if not calibration['active_gamma']:
        axes[1] = [float(np.clip(gamma, *ranges[1]))]
    candidates, calls = [], 0
    stopped = False
    scientific_signature = _scientific_signature(base)
    # Hold back work for robustness checks instead of spending every fit on
    # unique grid pairs. Tiny explicit budgets still test more than one pair.
    confirmation_reserve = min(3*(2*len(clips)-1), options.max_fits//3)
    refinement_reserve = min(4, options.max_fits//6)
    coarse_limit = options.max_fits-confirmation_reserve-refinement_reserve

    def exhausted():
        return calls >= options.max_fits or time.perf_counter()-started >= options.seconds

    def candidate(s, g):
        for item in candidates:
            if np.isclose(item['sigma_prior'], s, rtol=1e-12, atol=0) and np.isclose(item['gamma_prior'], g, rtol=1e-12, atol=0):
                return item
        item = {'candidate': len(candidates), 'sigma_prior': float(s), 'gamma_prior': float(g), 'fits': []}
        candidates.append(item)
        return item

    def evaluate(item, block, seed, stage):
        nonlocal calls, stopped
        if any(f['block'] == block and f['seed'] == seed for f in item['fits']):
            return
        if exhausted():
            stopped = True
            return
        calls += 1
        start, cut, end = clips[block]
        entry = {'block': block, 'seed': seed, 'stage': stage, 'training_start': start,
                 'training_end': cut, 'validation_end': end}
        tick = time.perf_counter()
        try:
            config = factory(item['sigma_prior'], item['gamma_prior'])
            if _scientific_signature(config) != scientific_signature:
                raise ValueError('config_factory changed the representation or scientific settings')
            config = replace(config, verbose=False,
                inference=replace(config.inference,
                    it_limit=options.iterations if config.inference.schedule in (None, 'offline') else None,
                    n_explore_steps=options.exploration_steps, split_merge_refine_steps=1),
                numerics=replace(config.numerics, seed=seed, num_threads=1))
            with redirect_stdout(io.StringIO()):
                model = HDPGPC(config).fit(values[start:cut], grid=grid)
            with torch.no_grad():
                validation = validate_predictive(model, values[cut:end], grid)
            counts = np.unique(model.labels_, return_counts=True)[1]
            entry.update(status='ok', **validation, train_assignments=model.labels_.astype(int).tolist(),
                clusters=model.n_clusters_, singleton_fraction=float(counts[counts == 1].sum()/counts.sum()),
                cluster_cap_reached=bool(model.n_clusters_ >= (config.hdp.max_models or 10**9)))
            if stage == 'confirm':
                try:
                    entry['gap_diagnostics'] = gap_diagnostics(model)
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                    entry['gap_diagnostics'] = {'error': str(error)}
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            entry.update(status='failed', error=f'{type(error).__name__}: {error}')
        entry['seconds'] = time.perf_counter()-tick
        item['fits'].append(entry)
        if progress:
            progress({'fit': calls, 'sigma_prior': item['sigma_prior'], 'gamma_prior': item['gamma_prior'],
                      'stage': stage, 'status': entry['status'], 'seconds': entry['seconds']})

    # Test the supplied anchor and range corners first, then prioritize the
    # remaining log grid by calibration distance. Repeated pairs are cached.
    initial = [(float(s), float(g)) for s in axes[0] for g in axes[1]]
    corners = [(float(s), float(g)) for s in (axes[0][0], axes[0][-1])
               for g in (axes[1][0], axes[1][-1])]
    anchor_allowed = all(bounds[0] <= value <= bounds[1]
                         for bounds, value in zip(ranges, (sigma, gamma)))
    distance = lambda pair: abs(np.log(pair[0]/sigma0))+abs(np.log(pair[1]/gamma0))
    initial = ([(sigma, gamma)] if anchor_allowed else [])+corners+sorted(initial, key=distance)
    for s, g in initial:
        evaluate(candidate(s, g), 0, options.seed, 'coarse')
        if exhausted() or calls >= coarse_limit:
            break
        if calls >= 3 and time.perf_counter()-started >= .5*options.seconds:
            break  # Reserve time for actual promotion, not just a larger grid.

    def successes(item):
        return [f for f in item['fits'] if f['status'] == 'ok']

    def score(item):
        fits = successes(item)
        return float(np.mean([f['nll_per_feature'] for f in fits])) if fits else math.inf

    viable = sorted([item for item in candidates if successes(item)], key=score)
    refinement_limit = min(options.max_fits-confirmation_reserve, calls+refinement_reserve)
    if viable and not exhausted() and calls < refinement_limit:
        best = viable[0]
        for axis in range(2):
            if len(axes[axis]) < 2:
                continue
            for direction in (-1, 1):
                if calls >= refinement_limit or time.perf_counter()-started >= .7*options.seconds:
                    break
                pair = [best['sigma_prior'], best['gamma_prior']]
                # Half a grid interval in log space; refine more finely than
                # the grid itself, while respecting both range endpoints.
                factor = math.exp((math.log(ranges[axis][1])-math.log(ranges[axis][0]))
                                  /(2*(len(axes[axis])-1)))
                pair[axis] *= factor**direction
                pair[axis] = float(np.clip(pair[axis], *ranges[axis]))
                evaluate(candidate(*pair), 0, options.seed, 'refine')

    survivors = sorted([item for item in candidates if successes(item)], key=score)[:3]
    # Round-robin promotion avoids spending the remaining budget on one winner.
    for block, seed in [(b, options.seed) for b in range(1, len(clips))]+[
            (b, options.seed+1) for b in range(len(clips))]:
        for item in survivors:
            evaluate(item, block, seed, 'confirm')

    for item in candidates:
        fits = successes(item)
        item['successful_fits'] = len(fits)
        item['status'] = 'ok' if fits else 'failed' if item['fits'] else 'untested_budget_limited'
        item['mean_nll_per_feature'] = score(item) if fits else None
        item['nll_standard_error'] = (float(np.std([f['nll_per_feature'] for f in fits], ddof=1)/math.sqrt(len(fits)))
                                      if len(fits) > 1 else None)
        agreements = []
        for i, first in enumerate(fits):
            for second in fits[i+1:]:
                if first['block'] == second['block'] and first['seed'] != second['seed']:
                    agreements.append(adjusted_rand_score(first['train_assignments'], second['train_assignments']))
        item['seed_assignment_agreement_ari'] = float(np.mean(agreements)) if agreements else None
        neighbors = sorted([other for other in candidates if other is not item and successes(other)],
            key=lambda other: abs(np.log(other['sigma_prior']/item['sigma_prior']))
                            +abs(np.log(other['gamma_prior']/item['gamma_prior'])))[:2]
        agreements = []
        for other in neighbors:
            for first in fits:
                for second in successes(other):
                    if first['block'] == second['block'] and first['seed'] == second['seed']:
                        agreements.append(adjusted_rand_score(first['train_assignments'], second['train_assignments']))
        item['nearby_assignment_agreement_ari'] = float(np.mean(agreements)) if agreements else None
    promoted = [item for item in survivors if any(f['stage'] == 'confirm' and f['status'] == 'ok' for f in item['fits'])]
    pool = promoted or [item for item in candidates if successes(item)]
    recommendation, region = None, []
    if pool:
        winner = min(pool, key=score)
        tolerance = max(winner['nll_standard_error'] or 0., 1e-3)
        region = [item for item in pool if score(item) <= score(winner)+tolerance]
        # Choose the best seed agreement within the predictive plateau.
        winner = min(region, key=lambda item: (-(item['seed_assignment_agreement_ari']
                      if item['seed_assignment_agreement_ari'] is not None else -1.), score(item)))
        recommendation = {'candidate': winner['candidate'], 'sigma_prior': winner['sigma_prior'],
            'gamma_prior': winner['gamma_prior'], 'nll_per_feature': score(winner),
            'confirmation': 'promoted' if winner in promoted else 'unconfirmed_budget_limited',
            'evaluated_blocks': sorted({f['block'] for f in successes(winner)}),
            'evaluated_seeds': sorted({f['seed'] for f in successes(winner)}),
            'confirmation_complete': all(any(f['block'] == b and f['seed'] == seed
                for f in successes(winner)) for b in range(len(clips))
                for seed in (options.seed, options.seed+1)),
            'cli': f"--sigma-prior {winner['sigma_prior']!r} --gamma-prior {winner['gamma_prior']!r}"}
    return {'schema_version': 1, 'status': 'ok' if recommendation is not None else 'failed',
        'options': asdict(options), 'seconds': time.perf_counter()-started, 'fit_calls': calls,
        'budget_exhausted': stopped or exhausted(), 'calibration': calibration,
        'clips': [{'start': a, 'train_end': b, 'validation_end': c} for a, b, c in clips],
        'initial_ranges': {'sigma': list(ranges[0]), 'gamma': list(ranges[1])},
        'grid': {'sigma_values': [float(x) for x in axes[0]],
                 'gamma_values': [float(x) for x in axes[1]],
                 'combinations': len(axes[0])*len(axes[1]),
                 'coarse_fit_budget': coarse_limit,
                 'refinement_fit_reserve': refinement_reserve,
                 'confirmation_fit_reserve': confirmation_reserve},
        'recommendation': recommendation,
        'working_pairs': [{'sigma_prior': item['sigma_prior'], 'gamma_prior': item['gamma_prior'],
                           'candidate': item['candidate']} for item in region],
        'candidates': candidates,
        'validation': 'Pre-update known-component Gaussian mixture; frozen parameters; local MAP state filtering; no validation births',
        'scope': 'Empirical screened pairs, not independent rectangular bounds or a guarantee of semantic clusters',
        'representation': 'Factory-frozen; validation is held out from component fits, not necessarily from representation calibration',
        'budget_scope': 'Checked between fits; representation calibration is outside this timer',
        'scientific_config_sha256': scientific_signature, 'clock': base.dynamics.clock}
