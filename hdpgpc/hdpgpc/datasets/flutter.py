"""Published pitch-and-plunge flutter recordings; no artificial joined trajectory.

Santos et al. (2024), doi:10.1007/s44245-024-00043-w, supplementary archive.
The two source motion columns remain in their original units. Record-level
response descriptions are references, not sample-wise segmentation targets.
"""
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
from urllib.request import Request, urlopen
import zipfile

import numpy as np


FLUTTER_ARCHIVE = 'flutter_2024_supplement.zip'
FLUTTER_URL = ('https://media.springernature.com/original/springer-static/esm/'
    'art%3A10.1007%2Fs44245-024-00043-w/MediaObjects/44245_2024_43_MOESM1_ESM.zip')
FLUTTER_SHA256 = 'c3e659d8e009817159c71c0969ecd76de9a37d109314c667d0dc9f0849f61927'
FLUTTER_PREFIX = 'MMS_et_al_2024_DiscMechEng_SupplementaryMaterial/'
FLUTTER_RECORDS = {
    '0700_largeIC': (7., 'large', 'damped', '609ee6294fdf949d6b4012416fccbc3a4c9edd86d4183978a27c675171e3dfdd'),
    '0700_smallIC': (7., 'small', 'damped', 'f902b5398d0bd8c98f3d26e35824ce528b39c4dcf2e983a9300a7c0af063d226'),
    '0815': (8.15, 'unspecified', 'near-onset / marginal bounded response', '02b62b75f60647c74b28f95e34842f7d033ba127ad67634ce75f6b24f259dec7'),
    '0840': (8.4, 'unspecified', 'bounded limit-cycle response', '46da84f54b7e71b9127377d0ea3aeb865de2bbc21ec9d2a3a6b38392c0ab9cd3'),
    '0900': (9., 'unspecified', 'bounded limit-cycle response', 'e42ac42381bce72ac8ad254def08e46bc929ed843de0f75a70252ddb1a5ed620'),
    '0950': (9.5, 'unspecified', 'unstable response / protective test stop', 'd532c833da5854e5eb48fb4ac9fc6de416106017d0981e47eefb678f8197105d'),
}


@dataclass(frozen=True)
class FlutterRecord:
    name: str
    times: np.ndarray
    motion: np.ndarray
    airflow_speed: float
    initial_condition: str
    reference: str
    file_sha256: str
    archive_sha256: str


@dataclass(frozen=True)
class FlutterWindows:
    record: FlutterRecord
    values: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    times: np.ndarray
    window_samples: int
    step: int


def download_flutter(directory):
    """Cache the checksum-pinned small publisher supplement; never execute it."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path = directory/FLUTTER_ARCHIVE
    if path.is_file():
        if hashlib.sha256(path.read_bytes()).hexdigest() != FLUTTER_SHA256:
            raise ValueError(f'Flutter archive checksum mismatch: {path}')
        return path
    temporary = path.with_suffix('.zip.part')
    try:
        request = Request(FLUTTER_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urlopen(request, timeout=60) as source, temporary.open('wb') as target:
            size = 0
            while block := source.read(65536):
                size += len(block)
                if size > 4*1024*1024:
                    raise ValueError('Flutter archive exceeds 4 MB')
                target.write(block)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != FLUTTER_SHA256:
            raise ValueError('Flutter archive checksum mismatch')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_flutter_record(directory, record='0950'):
    """Load one pinned member directly from the ZIP, validating its real clock."""
    if record not in FLUTTER_RECORDS:
        raise ValueError(f'Unknown flutter record: {record}')
    path = Path(directory)/FLUTTER_ARCHIVE
    if not path.is_file():
        raise FileNotFoundError(f'Flutter archive absent in {directory}; use --download')
    archive_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if archive_sha != FLUTTER_SHA256:
        raise ValueError('Flutter archive checksum mismatch')
    speed, initial, reference, expected = FLUTTER_RECORDS[record]
    member = FLUTTER_PREFIX+f'timeRespData_{record}.txt'
    with zipfile.ZipFile(path) as bundle:
        if bundle.namelist().count(member) != 1 or bundle.getinfo(member).file_size > 4*1024*1024:
            raise ValueError('Missing, duplicate or oversized flutter motion member')
        raw = bundle.read(member)
    sha = hashlib.sha256(raw).hexdigest()
    if sha != expected:
        raise ValueError(f'Flutter record checksum mismatch: {record}')
    values = np.loadtxt(io.BytesIO(raw), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 3 or not np.isfinite(values).all():
        raise ValueError('Flutter data require finite time and two motion columns')
    times, motion = values[:, 0], values[:, 1:]
    spacing = np.diff(times)
    if np.any(spacing <= 0) or not np.allclose(spacing, np.median(spacing), rtol=.01, atol=2e-6):
        raise ValueError('Flutter timestamps must increase regularly; gaps must not be bridged')
    return FlutterRecord(record, times.copy(), motion.copy(), speed, initial, reference, sha, archive_sha)


def window_flutter_record(record, *, window_samples=60, step=12):
    """Causal, complete delay windows; preserve amplitude and original timestamps.

    Flatten in time-major/channel-interleaved order. No filtering, derivatives,
    phase alignment, per-window normalization or padding is performed.
    """
    if (not isinstance(window_samples, int) or not isinstance(step, int)
            or window_samples < 2 or step < 1 or window_samples > len(record.times)):
        raise ValueError('Require window_samples >= 2, positive step and enough source samples')
    starts = np.arange(0, len(record.times)-window_samples+1, step, dtype=np.int64)
    ends = starts+window_samples-1
    values = np.stack([record.motion[start:start+window_samples].reshape(-1) for start in starts])
    return FlutterWindows(record, values, starts, ends, record.times[ends], window_samples, step)
