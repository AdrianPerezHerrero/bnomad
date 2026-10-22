"""Data-only schema checks and atomic persistence for the typed MFA facade."""
from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
import re
import tempfile

import torch

from ..emissions import FactorAnalysisEmission


def _json_tree(value, active=None, depth=0):
    if depth > 100:
        raise ValueError('State exceeds the supported nesting depth')
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if active is None:
        active = set()
    if not isinstance(value, (Mapping, list)) or id(value) in active:
        raise ValueError('State must contain finite JSON data only, without cycles')
    active.add(id(value))
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError('State keys must be strings')
        values = value.values()
    else:
        values = value
    for item in values:
        _json_tree(item, active, depth + 1)
    active.remove(id(value))


def validate_payload(data):
    _json_tree(data)
    if (not isinstance(data, Mapping) or data.get('type') != 'hdpmfa'
            or type(data.get('version')) is not int or data['version'] not in (1, 2)):
        raise ValueError('Unsupported DP-MFA state type or version')
    common = {'type', 'version', 'config', 'observation_dim', 'noise', 'components',
              'responsibilities', 'rho', 'omega', 'sequence_lengths', 'history',
              'elbo_history', 'elbo_parts', 'n_iter', 'converged'}
    optional = {'structure_history', 'seen_partitions'}
    markov = {'pair_responsibilities', 'transition_parameters', 'start_parameters'}
    if not common <= set(data) or set(data) - common - optional - markov:
        raise ValueError('Missing or unknown model state fields')
    if data['version'] == 2 and not optional <= set(data):
        raise ValueError('Version 2 requires structural history fields')
    if type(data['observation_dim']) is not int or data['observation_dim'] < 1:
        raise ValueError('observation_dim must be a positive integer')
    if not isinstance(data['components'], list) or not data['components']:
        raise ValueError('At least one component is required')
    for component in data['components']:
        if not isinstance(component, Mapping) or 'noise' in component:
            raise ValueError('Component state must use the single model-level noise representation')
        if (component.get('dtype') != 'float64' or component.get('type') != 'factor_analysis'
                or type(component.get('version')) is not int or component.get('version') != 1):
            raise ValueError('Unsupported component schema or facade dtype')
        fields = {'type', 'version', 'observation_dim', 'n_factors', 'dtype', 'noise_floor',
                  'loading_prior_mean', 'loading_prior_precision', 'parameter_mean',
                  'parameter_covariance', 'occupancy'}
        if set(component) != fields:
            raise ValueError('Missing or unknown component state fields')
        d, p = component['observation_dim'], component['n_factors'] + 1
        for name, shape in (('loading_prior_mean', (d,p)), ('loading_prior_precision', (d,p,p)),
                            ('parameter_mean', (d,p)), ('parameter_covariance', (d,p,p))):
            if torch.as_tensor(component[name], dtype=torch.float64).shape != shape:
                raise ValueError(f'Saved {name} must have exact shape {shape}; state arrays do not broadcast')
    for name in ('noise', 'history', 'elbo_history', 'structure_history', 'seen_partitions'):
        if name in data and not isinstance(data[name], list):
            raise ValueError(f'{name} must be a list')
    seen = data.get('seen_partitions', [])
    if (any(type(key) is not str or not re.fullmatch(r'[1-9][0-9]*:[0-9a-f]{32}', key) for key in seen)
            or len(seen) != len(set(seen))):
        raise ValueError('Invalid or repeated visited partition keys')


def _parts(parts):
    required = {'local_evidence', 'allocation', 'assignment_entropy', 'parameter_kl', 'total'}
    if not isinstance(parts, Mapping) or not required <= set(parts):
        raise ValueError('Incomplete objective components')
    if any(type(parts[key]) not in (float, int) or not math.isfinite(parts[key]) for key in required):
        raise ValueError('Nonfinite objective components')
    total = parts['local_evidence'] + parts['allocation'] + parts['assignment_entropy'] - parts['parameter_kl']
    if (parts['parameter_kl'] < -1e-8 or parts['assignment_entropy'] < -1e-8
            or not math.isclose(total, parts['total'], rel_tol=1e-11, abs_tol=1e-8)):
        raise ValueError('Inconsistent objective components')


def validate_restored(state):
    n, d, config = len(state.resp), state.observation_dim, state.config
    if state.M > n:
        raise ValueError('Component support exceeds the number of training observations')
    # A loader must reject corrupted variances rather than silently applying
    # the construction-time floor to a saved numerical state.
    if state._saved_noise.shape != (d,) or not bool(torch.isfinite(state._saved_noise).all()
            and (state._saved_noise >= config.noise_floor).all()):
        raise ValueError('Saved noise must be a finite vector at or above its variance floor')
    if not torch.equal(state.noise, state._saved_noise):
        raise ValueError('Saved noise was altered during restoration')
    expected = FactorAnalysisEmission(d, config.n_factors,
        loading_prior_mean=config.loading_prior_mean,
        loading_prior_precision=config.loading_prior_precision,
        noise=config.noise_initial, noise_floor=config.noise_floor,
        dtype=state.dtype, device=state.device)
    if not config.learn_noise and not torch.equal(state.noise, expected.noise):
        raise ValueError('Fixed saved noise disagrees with the configuration')
    for k, component in enumerate(state.components):
        if (component.noise_floor != config.noise_floor
                or not torch.equal(component.prior_mean, expected.prior_mean)
                or not torch.equal(component.prior_precision, expected.prior_precision)):
            raise ValueError('Saved component prior disagrees with the configuration')
        if not torch.allclose(component.occupancy, state.resp[:, k].sum(), atol=1e-8, rtol=1e-10):
            raise ValueError('Saved component occupancy disagrees with responsibilities')
    kl = sum(float(component.parameter_kl()) for component in state.components)
    if not math.isclose(kl, state.elbo_parts['parameter_kl'], rel_tol=1e-10, abs_tol=1e-7):
        raise ValueError('Saved parameter KL disagrees with component posteriors')
    if state.hmm_switch:
        from ..hdp._markov import markov_counts, markov_linear_elbo
        from ..inference._mfa_assignments import chain_entropy
        allocation = float(markov_linear_elbo(state.hdp, *markov_counts(state)))
        entropy = float(chain_entropy(state.resp, state.respPair, state.sequence_lengths))
    else:
        allocation = float(state.hdp.iid_linear_elbo(state.resp))
        entropy = float(-torch.special.xlogy(state.resp, state.resp).sum())
    for name, value in (('allocation', allocation), ('assignment_entropy', entropy)):
        if not math.isclose(value, state.elbo_parts[name], rel_tol=1e-10, abs_tol=1e-7):
            raise ValueError(f'Saved {name} disagrees with the restored assignment/HDP factors')
    _parts(state.elbo_parts)
    previous = state.elbo_history[0]
    for index, record in enumerate(state.history, 1):
        _parts(record)
        if (type(record.get('iteration')) is not int or record['iteration'] != index
                or not math.isclose(record['total'], state.elbo_history[index], abs_tol=1e-8, rel_tol=1e-11)
                or not math.isclose(record['delta_elbo'], record['total'] - previous, abs_tol=1e-8, rel_tol=1e-10)
                or record['responsibility_change'] < 0 or record.get('pair_change', 0) < 0
                or not isinstance(record['phases'], list)):
            raise ValueError('Saved coordinate history is inconsistent')
        if record['total'] < previous - (1e-8 + 1e-10 * max(1., abs(previous), abs(record['total']))):
            raise ValueError('Saved objective history decreases')
        for phase in record['phases']:
            _parts(phase)
        previous = record['total']
    for move in state.structure_history:
        if (not isinstance(move, Mapping) or move.get('move') not in ('birth', 'prune')
                or type(move.get('accepted')) is not bool or type(move.get('iteration')) is not int
                or not 1 <= move['iteration'] <= state.n_iter or type(move.get('source')) is not int
                or move['source'] < 0 or type(move.get('reason')) is not str):
            raise ValueError('Saved structural history is invalid')
    if state.converged and state.history[-1].get('structure_accepted', False):
        raise ValueError('A model cannot converge immediately after a structural move')
    if not state.hmm_switch and state._saved_markov_fields:
        raise ValueError('Independent model state must not contain Markov factors')
    del state._saved_noise, state._saved_markov_fields


def atomic_save(path, data):
    payload = json.dumps(data, allow_nan=False, indent=2) + '\n'
    destination = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + destination.name + '.', suffix='.tmp',
                                             dir=destination.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
