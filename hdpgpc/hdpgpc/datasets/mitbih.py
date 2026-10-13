"""MIT-BIH beat extraction and alignment for the legacy bundled arrays.

The original extraction filtered WFDB annotation symbols and sample positions
together. It removed invalid leading windows from both, but could leave a
trailing label when the final waveform window exceeded the signal length.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


BEAT_SYMBOLS = frozenset(('N', 'L', 'R', 'a', 'A', 'J', 'S', 'e', 'j',
                          'V', 'E', 'F', '/', 'f', 'Q', '!', 'n'))


@dataclass(frozen=True)
class MITBIHRecord:
    waveforms: np.ndarray
    annotations: np.ndarray
    annotation_samples: np.ndarray | None = None
    omitted_trailing_annotations: int = 0


def align_legacy_annotations(waveforms, annotations):
    """Pair archived waveforms with labels after legacy end-window omission.

    Legacy extraction already removed invalid *leading* labels alongside their
    sample positions. In the bundled archive, the only observed length mismatch
    is one trailing label for a window extending past the signal. Equal-length
    arrays retain their order. Other mismatches need source sample positions;
    guessing a shift from waveform appearance is unsafe.
    """
    data = np.asarray(waveforms)
    labels = np.asarray(annotations).reshape(-1)
    if data.ndim != 3 or not all(data.shape):
        raise ValueError('MIT-BIH waveforms must have shape (beats, points, channels)')
    extra = len(labels) - len(data)
    if extra not in (0, 1):
        raise ValueError(
            f'Cannot align {len(labels)} annotations to {len(data)} beats without '
            'the original WFDB annotation sample positions')
    return labels[:len(data)], extra


def load_mitbih_record(directory, record) -> MITBIHRecord:
    """Load one bundled record and apply the documented legacy tail rule."""
    record = str(record)
    if not record.isdigit():
        raise ValueError('MIT-BIH record must be a numeric record identifier')
    root = Path(directory)
    waveforms = np.load(root / f'{record}.npy', allow_pickle=False)
    labels = np.load(root / f'{record}_labels.npy', allow_pickle=False)
    aligned, omitted = align_legacy_annotations(waveforms, labels)
    return MITBIHRecord(waveforms, aligned, omitted_trailing_annotations=omitted)


def extract_annotated_beats(signal, annotation_samples, annotation_symbols, *,
                            window_start=-27, window_stop=63,
                            center_per_beat=True) -> MITBIHRecord:
    """Extract the old 90-point window while filtering samples and labels jointly.

    Defaults reproduce ``get_data(database='filtered', samples=[60, 150],
    scale_type='mean', ann='atr')``: WFDB sample plus [-27, 63), strict
    upper-bound exclusion, and per-beat channel means removed. All three
    returned arrays use the same validity mask.
    """
    values = np.asarray(signal)
    positions = np.asarray(annotation_samples)
    symbols = np.asarray(annotation_symbols).reshape(-1)
    if values.ndim != 2 or not all(values.shape):
        raise ValueError('signal must have shape (signal samples, channels)')
    if (positions.ndim != 1 or positions.dtype.kind not in 'iu'
            or len(positions) != len(symbols)):
        raise ValueError('annotation samples and symbols must be paired and ordered')
    positions = positions.astype(np.int64, copy=False)
    if np.any(np.diff(positions) < 0):
        raise ValueError('annotation samples must be ordered')
    if not isinstance(window_start, int) or not isinstance(window_stop, int) or window_start >= window_stop:
        raise ValueError('window_start must be smaller than window_stop')
    keep_symbols = np.isin(symbols, tuple(BEAT_SYMBOLS))
    positions, symbols = positions[keep_symbols], symbols[keep_symbols]
    starts, stops = positions + window_start, positions + window_stop
    valid = (starts >= 0) & (stops < len(values))
    omitted_trailing = int(np.count_nonzero(stops >= len(values)))
    positions, symbols = positions[valid], symbols[valid]
    starts = starts[valid]
    width = window_stop - window_start
    if len(starts):
        windows = np.stack([values[start:start + width] for start in starts]).astype(np.float64)
        if center_per_beat:
            windows -= windows.mean(axis=1, keepdims=True)
        windows = np.nan_to_num(windows, copy=False)
    else:
        windows = np.empty((0, width, values.shape[1]), dtype=np.float64)
    return MITBIHRecord(windows, symbols, positions, omitted_trailing)


def extract_wfdb_record(record_base, *, annotation='atr') -> MITBIHRecord:
    """Read a local filtered WFDB record; requires the optional datasets extra.

    Pass the same filtered ``*_filtered`` record base used to make the archived
    arrays. Annotation samples and symbols are read together from its ``atr``
    file, so omitted windows cannot shift later labels.
    """
    try:
        import wfdb
    except ImportError as error:
        raise ImportError("Install hdpgpc[datasets] to extract WFDB records") from error
    base = str(record_base)
    signal = wfdb.rdrecord(base, return_res=32, physical=False).d_signal
    annotations = wfdb.rdann(base, annotation)
    return extract_annotated_beats(signal, annotations.sample, annotations.symbol)
