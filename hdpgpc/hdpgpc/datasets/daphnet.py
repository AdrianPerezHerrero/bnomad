"""Daphnet acceleration loading and chronological experimental windows.

Column order, units and label meanings follow the documentation distributed
with UCI dataset 245. No class labels enter waveform preparation or learning;
label zero identifies periods outside the experimental protocol.
"""

from dataclasses import dataclass
from math import gcd
from pathlib import Path
import re
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
from scipy.signal import resample_poly


DAPHNET_URL = 'https://archive.ics.uci.edu/static/public/245/daphnet+freezing+of+gait.zip'
DAPHNET_CHANNELS = (
    'ankle_forward', 'ankle_vertical', 'ankle_lateral',
    'thigh_forward', 'thigh_vertical', 'thigh_lateral',
    'trunk_forward', 'trunk_vertical', 'trunk_lateral',
)
DAPHNET_FS = 64


@dataclass(frozen=True)
class DaphnetRecord:
    name: str
    time_seconds: np.ndarray
    acceleration_g: np.ndarray
    annotations: np.ndarray


@dataclass(frozen=True)
class DaphnetWindows:
    segment: int
    values: np.ndarray
    grid: np.ndarray
    annotations: np.ndarray
    freeze_fraction: np.ndarray
    start_rows: np.ndarray
    start_seconds: np.ndarray
    end_seconds: np.ndarray


def download_daphnet(path):
    """Download the official ZIP if absent, without extracting its members."""
    path = Path(path)
    if path.is_file():
        with ZipFile(path):
            pass
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    with urlopen(DAPHNET_URL, timeout=60) as source, temporary.open('wb') as target:
        total = 0
        while block := source.read(1024 * 1024):
            total += len(block)
            if total > 64 * 1024 * 1024:
                raise ValueError('Daphnet download exceeded 64 MB')
            target.write(block)
    with ZipFile(temporary):
        pass
    temporary.replace(path)
    return path


def _record_members(names):
    members = {}
    for name in names:
        stem = Path(name).stem.upper()
        if re.fullmatch(r'S\d{2}R\d{2}', stem) and Path(name).suffix.lower() == '.txt':
            if stem in members:
                raise ValueError(f'Duplicate Daphnet recording: {stem}')
            members[stem] = name
    return members


def daphnet_records(source):
    """List recording IDs from the official ZIP or an extracted directory."""
    source = Path(source)
    if source.is_dir():
        return sorted(_record_members(str(path) for path in source.rglob('*.txt')))
    with ZipFile(source) as archive:
        return sorted(_record_members(archive.namelist()))


def load_daphnet_record(source, record):
    """Load timestamps in seconds, nine acceleration channels in g and labels.

    Labels retain the source codes: 0 outside the experiment, 1 experimental
    non-freezing (standing, walking or turning), 2 freezing. Rows are neither
    shuffled nor filtered. Timestamp discontinuities are handled at windowing.
    """
    source = Path(source)
    record = str(record).upper().removesuffix('.TXT')
    if not re.fullmatch(r'S\d{2}R\d{2}', record):
        raise ValueError('Record must be a Daphnet ID such as S02R01')
    if source.is_dir():
        members = _record_members(str(path) for path in source.rglob('*.txt'))
        if record not in members:
            raise ValueError(f'Daphnet recording not found: {record}')
        table = np.loadtxt(members[record], ndmin=2)
    else:
        with ZipFile(source) as archive:
            members = _record_members(archive.namelist())
            if record not in members:
                raise ValueError(f'Daphnet recording not found: {record}')
            with archive.open(members[record]) as stream:
                table = np.loadtxt(stream, ndmin=2)
    if table.shape[1] != 11 or not len(table) or not np.isfinite(table).all():
        raise ValueError('Daphnet records require eleven finite columns')
    if not np.isin(table[:, -1], (0, 1, 2)).all():
        raise ValueError('Daphnet annotations must be 0, 1 or 2')
    return DaphnetRecord(record, table[:, 0] / 1000., table[:, 1:10] / 1000.,
                         table[:, -1].astype(np.int64))


def window_daphnet_record(record, *, channel='ankle_vertical', window_seconds=2.,
                          stride_seconds=None, points=64):
    """Return independent contiguous experimental segments of waveform windows.

    Never join across label-zero periods, nonpositive timestamp increments or
    gaps > 1.5/64 seconds. Windows keep their source order and cannot straddle
    those boundaries. Window labels use a majority of raw annotations (ties
    count as freezing); freeze_fraction preserves mixed-window information.
    A polyphase anti-alias filter resamples each window when points is smaller
    than its native length. Window labels are computed before resampling.
    """
    if channel not in DAPHNET_CHANNELS:
        raise ValueError(f'Unknown Daphnet channel: {channel}')
    stride_seconds = window_seconds if stride_seconds is None else stride_seconds
    for value, name in ((window_seconds, 'window_seconds'), (stride_seconds, 'stride_seconds')):
        if (not np.isfinite(value) or value <= 0 or
                not np.isclose(value * DAPHNET_FS, round(value * DAPHNET_FS), atol=1e-8, rtol=0)):
            raise ValueError(f'{name} must be a positive multiple of 1/64 second')
    length, stride = round(window_seconds * DAPHNET_FS), round(stride_seconds * DAPHNET_FS)
    if (isinstance(points, (bool, np.bool_)) or not isinstance(points, (int, np.integer))
            or not 2 <= points <= length):
        raise ValueError('points must be an integer between 2 and the native window length')
    eligible = record.annotations != 0
    dt = np.diff(record.time_seconds)
    cuts = np.flatnonzero((eligible[1:] != eligible[:-1]) | (dt <= 0)
                         | (dt > 1.5 / DAPHNET_FS)) + 1
    boundaries = np.r_[0, cuts, len(eligible)]
    segments = []
    segment = 0
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        if not eligible[begin]:
            continue
        segment += 1
        starts = np.arange(begin, end - length + 1, stride, dtype=np.int64)
        if not len(starts):
            continue
        rows = starts[:, None] + np.arange(length)
        waveforms = record.acceleration_g[rows, DAPHNET_CHANNELS.index(channel)]
        fractions = (record.annotations[rows] == 2).mean(axis=1)
        if points != length:
            divisor = gcd(points, length)
            waveforms = resample_poly(waveforms, points // divisor, length // divisor,
                                      axis=1, padtype='line')
        segments.append(DaphnetWindows(
            segment, waveforms[:, :, None],
            (np.arange(points) * window_seconds / points)[:, None],
            np.where(fractions >= 0.5, 2, 1), fractions, starts,
            record.time_seconds[starts],
            record.time_seconds[starts + length - 1] + 1 / DAPHNET_FS))
    return segments
