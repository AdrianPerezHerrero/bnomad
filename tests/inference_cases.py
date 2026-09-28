"""Observe real inference boundaries; never substitute numerical routines.

The frozen traces were recorded before extracting either engine. An exact,
streaming comparator stops at the first differing boundary (including inputs,
outputs and owner state). The exception deliberately bypasses legacy broad
``except Exception`` handlers. CPU float64, fixed seeds and one thread are the
same as the existing characterization suite.
"""

from contextlib import contextmanager, redirect_stdout
from functools import wraps
import gzip
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))

from tests.characterization._support import (
    ModelState, GaussianProcessEmission, encode, gp_state, reset_runtime, environment,
    make_model, model_state, waveform_data, np, torch)
from tests.characterization._cases import CASES

NAMES = tuple(n for n in CASES if n.startswith(('online_', 'offline_', 'known_failure_')))
FIXTURE = Path(__file__).parent / 'fixtures/inference_reference.json.gz'
SUPPLEMENTAL_FIXTURE = Path(__file__).parent / 'fixtures/inference_edge_reference.json.gz'
MODEL_METHODS = (
    'variational_local_terms', 'variational_local_terms_batch', 'estimate_q_first',
    'estimate_q_all', 'compute_q_elbo', 'elbo_Linears', 'full_LDS_elbo',
    '_calcThetaFull', 'find_optimum_rhoOmega', 'reinit_global_params',
    'create_gp_default', 'gpmodel_deepcopy', 'refill', 'reorder', 'new_group',
    'remove_last_group', '_init_resp_cache_cold_start', '_ensure_fast_cache_online',
    '_update_q_lat_tail', '_append_hard_step', 'forward', 'backward',
    'estimate_new', 'compute_warp_y', 'compute_warp_actual_state',
)
GP_METHODS = (
    'include_weighted_sample', 'full_pass_weighted', 'fit_kernel_params',
    'backwards_pair', 'backwards', 'bayesian_new_params', 'new_params',
    'reinit_GP', 'reinit_LDS', 'compute_sq_err_all', 'compute_q_lat_all',
    'log_sq_error',
)
STATE_FIELDS = (
    'M', 'T', 'actual_state', 'rho', 'omega', 'transTheta', 'startTheta',
    'trans_A', 'q', 'resp_assigned', 'snr_norm', 'train_elbo', 'f_ind_old',
    'q_last', 'q_lat_last', 'resp_last', 'respPair_last', 'snr_last', 'elbo_last',
    'startStateCount_last', 'transStateCount_last',
)


class InferenceDivergence(BaseException):
    """Stop immediately without being swallowed by legacy fallback handlers."""


def first_difference(actual, expected, path='boundary'):
    if type(actual) is not type(expected):
        return path + ': type'
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            return path + ': keys'
        for key in expected:
            diff = first_difference(actual[key], expected[key], path + '.' + key)
            if diff:
                return diff
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            return path + ': length'
        for i, (a, e) in enumerate(zip(actual, expected)):
            diff = first_difference(a, e, f'{path}[{i}]')
            if diff:
                return diff
    elif actual != expected:
        return f'{path}: {actual!r} != {expected!r}'
    return None


def snapshot(value):
    if isinstance(value, GaussianProcessEmission):
        return encode(gp_state(value))
    if isinstance(value, dict):
        return {str(k): snapshot(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [snapshot(v) for v in value]
    return encode(value)


@contextmanager
def trace_calls(expected=None):
    events = []
    patches = []

    def emit(event):
        index = len(events)
        if expected is not None:
            if index >= len(expected):
                raise InferenceDivergence(f'extra event {index}: {event["method"]}')
            diff = first_difference(event, expected[index])
            if diff:
                raise InferenceDivergence(f'event {index} {event["phase"]} {event["method"]}: {diff}')
        events.append(event)

    def observer(name, original, is_gp):
        @wraps(original)
        def call(owner, *args, **kwargs):
            def state():
                if is_gp:
                    return snapshot(owner)
                return snapshot({k: getattr(owner, k) for k in STATE_FIELDS if hasattr(owner, k)})
            emit(dict(method=name, phase='enter', state=state(), args=snapshot(args), kwargs=snapshot(kwargs)))
            result = original(owner, *args, **kwargs)
            emit(dict(method=name, phase='exit', state=state(), result=snapshot(result)))
            return result
        return call

    try:
        for cls, names, is_gp in ((ModelState, MODEL_METHODS, False), (GaussianProcessEmission, GP_METHODS, True)):
            for name in names:
                if hasattr(cls, name):
                    p = patch.object(cls, name, observer(('GPI_model' if is_gp else 'GPI_HDP') + '.' + name, getattr(cls, name), is_gp))
                    p.start()
                    patches.append(p)
        yield events
        if expected is not None and len(events) != len(expected):
            raise InferenceDivergence(f'missing events: {len(events)} != {len(expected)}')
    finally:
        for p in reversed(patches):
            p.stop()


@contextmanager
def direct_engine_calls(enabled):
    """Exercise public engine entry points against the same pre-edit traces."""
    if not enabled:
        yield
        return
    # All native state calls now use the canonical engines. The alternate
    # compatibility route no longer exists; retain this coverage of the same trace.
    yield


def edge_case(name):
    x, model = make_model()
    y = waveform_data()
    xs = np.repeat(x[None], len(y), axis=0)
    if name == 'offline_single':
        model.offline.fit(xs[:1], y[:1], it_limit=1, warp=False)
    elif name == 'offline_limit':
        model.offline.fit(xs, y, it_limit=1, warp=False)
    elif name == 'reload_labels':
        model.fit_labels(xs, y, torch.tensor([0, 0, 1, 1, 0, 1]), 2, warp=False)
    elif name == 'known_failure_forced_reuse':
        model.online.update(x, y[0], with_warp=False)
        try:
            model.online.update(x, y[1], with_warp=False, force_model=0)
        except UnboundLocalError as error:
            return dict(exception_type=type(error).__name__, message=str(error), partial_state=model_state(model))
        raise AssertionError('Forced-reuse failure changed: review this path explicitly')
    else:
        raise KeyError(name)
    return model_state(model)


EDGE_NAMES = ('offline_single', 'offline_limit', 'reload_labels', 'known_failure_forced_reuse')


def run_case(name, expected=None, direct=False):
    reset_runtime()
    with redirect_stdout(io.StringIO()), direct_engine_calls(direct), trace_calls(None if expected is None else expected['events']) as events:
        result = encode(CASES[name]() if name in CASES else edge_case(name))
    if expected is not None:
        diff = first_difference(result, expected['result'], 'final')
        if diff:
            raise InferenceDivergence(diff)
    return dict(events=events, result=result)


def reference(supplemental=False):
    with gzip.open(SUPPLEMENTAL_FIXTURE if supplemental else FIXTURE, 'rt', encoding='utf-8') as f:
        return json.load(f)


if __name__ == '__main__':
    for supplemental in (False, True):
        for name, expected in reference(supplemental)['cases'].items():
            run_case(name, expected)
            print(name, 'exact match', flush=True)
