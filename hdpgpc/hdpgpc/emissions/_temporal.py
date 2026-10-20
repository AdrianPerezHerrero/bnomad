"""Shared temporal-pair sampling for independent affine calibration."""

import numpy as np


def calibration_pairs(n_rows, *, max_samples, sampling, random_state, lag,
                      times=None, segment_ids=None):
    """Select real lagged row pairs and a bounded set of calibration rows.

    Callers validate integer controls, sampling policy and observation arrays.
    Returned durations use supplied time units or row-step units. Segment IDs
    identify contiguous runs, so reusing an ID never joins disconnected runs.
    """
    segments = np.zeros(n_rows, dtype=int) if segment_ids is None else np.asarray(segment_ids)
    if segments.shape != (n_rows,):
        raise ValueError('segment_ids must have one entry per observation')
    runs = np.r_[0, np.cumsum(segments[1:] != segments[:-1])]
    time = np.arange(n_rows, dtype=float) if times is None else np.asarray(times, dtype=float)
    if (time.shape != (n_rows,) or not np.isfinite(time).all()
            or np.any((np.diff(time) <= 0) & (runs[1:] == runs[:-1]))):
        raise ValueError('Calibration times must be finite and strictly increasing within each segment')
    starts = np.arange(n_rows - lag)
    starts = starts[runs[starts] == runs[starts + lag]]
    count = min(int(max_samples), n_rows)
    if sampling == 'prefix':
        starts = starts[starts + lag < count]
        indices = np.arange(count)
    elif count == n_rows:
        indices = np.arange(count)
    else:
        pair_count = min(count // 2, len(starts))
        starts = np.sort(np.random.default_rng(random_state).choice(starts, pair_count, replace=False))
        indices = np.unique(np.r_[starts, starts + lag])
    if not len(starts):
        raise ValueError('Temporal calibration has no valid within-segment lagged pairs; increase samples or reduce lag')
    pairs = np.column_stack((starts, starts + lag))
    durations = time[pairs[:, 1]] - time[pairs[:, 0]]
    return indices, pairs, durations
