"""Small, dataset-independent preparation helpers for waveform sequences."""

import numpy as np


def normalize_samples(signal):
    """Z-score each waveform/channel across its points, independently.

    Return ``(values, centers, scales)``. Parameter arrays have shape
    (observations, 1, outputs). Constant channels become zero and use scale
    one, avoiding division by zero. No information from other samples is used.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 3 or min(signal.shape) == 0 or not np.isfinite(signal).all():
        raise ValueError('signal must contain finite nonempty (samples, points, outputs) data')
    centers = signal.mean(axis=1, keepdims=True)
    standard_deviations = signal.std(axis=1, keepdims=True)
    constant = standard_deviations <= np.finfo(np.float64).eps
    scales = np.where(constant, 1., standard_deviations)
    values = np.where(constant, 0., (signal - centers) / scales)
    return values, centers, scales


def prepare_sequences(signal, *, channels=(0,), scale_from_first=False,
                      unit_grid=False):
    """Select channels and return values plus their common observation grid.

    ``signal`` has shape (sequences, points, channels). When requested, each
    channel is centered and scaled with the first sequence's pointwise mean
    and standard deviation. The same transform is applied to every sequence.
    The returned values always have shape (sequences, points, outputs).
    """
    signal = np.asarray(signal)
    if signal.ndim != 3 or min(signal.shape) == 0:
        raise ValueError('signal must have shape (sequences, points, channels) with nonempty axes')
    channels = tuple(channels)
    if (not channels or any(isinstance(channel, bool) or
                            not isinstance(channel, (int, np.integer)) or
                            channel < 0 or channel >= signal.shape[2]
                            for channel in channels)):
        raise ValueError('channels must select existing nonnegative channel indices')
    raw = np.asarray(signal[:, :, channels], dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError('signal must contain finite values')
    if scale_from_first:
        center = raw[0].mean(axis=0)
        scale = raw[0].std(axis=0)
        if np.any(scale <= np.finfo(np.float64).eps):
            raise ValueError('The first sequence has a zero/invalid scale in a selected channel.')
        values = (raw - center[None, None, :]) / scale[None, None, :]
    else:
        values = raw
    grid_end = 1.0 if unit_grid else raw.shape[1] - 1.0
    grid = np.linspace(0.0, grid_end, raw.shape[1], dtype=np.float64)[:, None]
    return values, grid
