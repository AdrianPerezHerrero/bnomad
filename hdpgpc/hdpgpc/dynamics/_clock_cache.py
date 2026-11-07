"""Ephemeral exact-work caches for physical-clock inference.

Caches do not enter model checkpoints, proposal copies, or configuration. Tensor
identity and mutation versions guard history-dependent work. Autograd tensors,
tensor subclasses and inference tensors without versions take the uncached path.
Small covariance factors additionally compare saved values, so replacement
matrices with equal values can reuse their factorization. Cache limits affect
only recomputation, never the stored trajectory or inference equations.
"""
from collections import OrderedDict
from weakref import WeakKeyDictionary

import torch


_WORK = WeakKeyDictionary()
_RTS_BYTES = 16 * 1024 * 1024


def tensor_key(value):
    if torch.is_inference_mode_enabled() or type(value) is not torch.Tensor or value.requires_grad:
        return None
    try:
        version = value._version
    except RuntimeError:
        return None
    return (id(value), version, tuple(value.shape), value.dtype, value.device,
            tuple(value.stride()), value.storage_offset())


def inputs_key(values):
    keys = tuple(tensor_key(value) for value in values)
    return None if any(key is None for key in keys) else keys


def callable_key(method):
    return id(getattr(method, '__func__', method))


class ClockWork:
    def __init__(self, state):
        self.state = state
        self.views = None
        self.positions = {}
        self.rts = {}
        self.rts_bytes = 0
        self.entropy = {}
        self.smoothing = None
        self.scores = None
        self.statistics = None
        self.factors = OrderedDict()

    def factor(self, role, matrix, operation):
        key = tensor_key(matrix)
        if key is None:
            return operation(matrix)
        entry = self.factors.get(role)
        if (entry is not None and entry[0] == callable_key(operation)
                and entry[1].shape == matrix.shape and entry[1].dtype == matrix.dtype
                and entry[1].device == matrix.device and torch.equal(entry[1], matrix)):
            self.factors.move_to_end(role)
            return entry[2]
        result = operation(matrix)
        self.factors[role] = (callable_key(operation), matrix.detach().clone(), result)
        self.factors.move_to_end(role)
        while len(self.factors) > 8:
            self.factors.popitem(last=False)
        return result

    def retain_rts(self, tick, entry):
        size = sum(value.numel()*value.element_size() for value in entry[2:])
        previous = self.rts.get(tick)
        previous_size = 0 if previous is None else sum(
            value.numel()*value.element_size() for value in previous[2:])
        if self.rts_bytes-previous_size+size <= _RTS_BYTES:
            self.rts[tick] = entry
            self.rts_bytes += size-previous_size


def work(emission, state):
    try:
        cached = _WORK.get(emission)
    except TypeError:
        return ClockWork(state)
    if cached is None or cached.state is not state:
        cached = ClockWork(state)
        _WORK[emission] = cached
    return cached


def discard(emission):
    try:
        _WORK.pop(emission, None)
    except TypeError:
        pass
