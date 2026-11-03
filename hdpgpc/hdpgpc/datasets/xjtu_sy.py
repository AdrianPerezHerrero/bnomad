"""XJTU-SY raw vibration windows on the acquisition clock, without gap filling."""
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen
import zipfile

import numpy as np

XJTU_SOURCE = 'https://biaowang.tech/xjtu-sy-bearing-datasets/'
XJTU_REVISION = '274226336036cd8deb2c40b4ba323af2d11a7290'
XJTU_ARCHIVE_URL = ('https://huggingface.co/datasets/DavidNguyen/XJTU-SY_Bearing_Datasets/'
                    f'resolve/{XJTU_REVISION}/XJTU-SY_Bearing_Datasets.zip')
XJTU_ARCHIVE_BYTES = 5447931446
XJTU_SAMPLE_RATE = 25600
XJTU_SAMPLES = 32768
XJTU_PERIOD_SECONDS = 60.
XJTU_CHANNELS = ('horizontal', 'vertical')
# Published terminal inspection and file counts; no intermediate health labels.
XJTU_RECORDS = {
    'Bearing1_1': (123, 'outer race'), 'Bearing1_2': (161, 'outer race'),
    'Bearing1_3': (158, 'outer race'), 'Bearing1_4': (122, 'cage'),
    'Bearing1_5': (52, 'inner and outer race'), 'Bearing2_1': (491, 'inner race'),
    'Bearing2_2': (161, 'outer race'), 'Bearing2_3': (533, 'cage'),
    'Bearing2_4': (42, 'outer race'), 'Bearing2_5': (339, 'outer race'),
    'Bearing3_1': (2538, 'outer race'), 'Bearing3_2': (2496, 'inner race, ball, cage and outer race'),
    'Bearing3_3': (371, 'inner race'), 'Bearing3_4': (1515, 'inner race'),
    'Bearing3_5': (114, 'outer race'),
}
XJTU_CONDITIONS = {1: ('35Hz12kN', 2100, 12.), 2: ('37.5Hz11kN', 2250, 11.),
                   3: ('40Hz10kN', 2400, 10.)}


@dataclass(frozen=True)
class XJTUBearingWindows:
    name: str
    values: np.ndarray
    acquisition_numbers: np.ndarray
    times: np.ndarray
    window_starts: np.ndarray
    window_ends: np.ndarray
    acquisition_rms: np.ndarray
    acquisition_peak: np.ndarray
    file_sha256: tuple
    source_fingerprint: str
    window_samples: int
    published_acquisitions: int
    fault_element: str
    rpm: int
    radial_load_kn: float


def acquisition_numbers(record, *, step=1, max_series=None):
    if record not in XJTU_RECORDS:
        raise ValueError(f'Unknown XJTU-SY bearing: {record}')
    if not isinstance(step, int) or step < 1 or (max_series is not None and max_series < 1):
        raise ValueError('Acquisition step and prefix length must be positive integers')
    numbers = np.arange(1, XJTU_RECORDS[record][0]+1, step, dtype=np.int64)
    return numbers if max_series is None else numbers[:max_series]


def _parse_csv(raw):
    if not raw.strip():
        raise ValueError('Empty XJTU-SY acquisition')
    first = raw.splitlines()[0].decode('utf-8-sig').strip().lower()
    header = first.replace(' ', '').replace('_', '')
    skip = int(header == 'horizontalvibrationsignals,verticalvibrationsignals')
    try:
        values = np.loadtxt(io.BytesIO(raw), delimiter=',', skiprows=skip, dtype=np.float64)
    except ValueError as error:
        raise ValueError('Expected horizontal,vertical XJTU-SY acceleration columns') from error
    if values.shape != (XJTU_SAMPLES, 2) or not np.isfinite(values).all():
        raise ValueError(f'XJTU-SY acquisitions require {XJTU_SAMPLES} finite rows and two columns')
    return values


class _RemoteZip(io.RawIOBase):
    """Small bounded HTTP range reads; never download the full 5.4 GB mirror."""
    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0
        self.cache_start, self.cache = 0, b''

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset+(self.position if whence == 1 else self.size if whence == 2 else 0)
        if whence not in (0, 1, 2) or position < 0:
            raise ValueError('Invalid archive seek')
        self.position = position
        return position

    def read(self, size=-1):
        size = self.size-self.position if size < 0 else min(size, self.size-self.position)
        if size <= 0:
            return b''
        if size > 8*1024*1024:
            raise ValueError('Remote ZIP read exceeds 8 MB')
        start = self.position
        if not (self.cache_start <= start and start+size <= self.cache_start+len(self.cache)):
            end = min(self.size, start+max(size, 65536))-1
            request = Request(self.url+f'?download=true&bnomad_range={start}-{end}',
                headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity',
                         'User-Agent': 'bnomad-xjtu-sy'})
            with urlopen(request, timeout=45) as source:
                expected = f'bytes {start}-{end}/{self.size}'
                if source.status != 206 or source.headers.get('Content-Range') != expected:
                    raise ValueError('Mirror must honor byte ranges; refusing full archive download')
                data = source.read(end-start+2)
            if len(data) != end-start+1:
                raise ValueError('Truncated remote archive range')
            self.cache_start, self.cache = start, data
        offset = start-self.cache_start
        self.position += size
        return self.cache[offset:offset+size]


def _write_manifest(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def download_xjtu_sy(directory, record='Bearing1_1', *, step=1, max_series=None):
    """Fetch selected CSVs from a pinned third-party copy of the published ZIP.

    Each member is ZIP-CRC checked, structurally validated and locally SHA-256
    recorded. The source provides no official per-file SHA manifest; a local
    digest is provenance/integrity evidence, not an independent authenticity pin.
    """
    numbers = acquisition_numbers(record, step=step, max_series=max_series)
    directory = Path(directory)/record
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory/'download_manifest.json'
    manifest = (json.loads(manifest_path.read_text()) if manifest_path.is_file() else
                dict(source=XJTU_SOURCE, mirror=XJTU_ARCHIVE_URL, revision=XJTU_REVISION, files={}))
    needed = []
    for number in numbers:
        name = f'{number}.csv'; path = directory/name
        known = manifest['files'].get(name)
        if path.is_file() and known:
            if hashlib.sha256(path.read_bytes()).hexdigest() != known['sha256']:
                raise ValueError(f'Cached acquisition checksum mismatch: {path}')
        else:
            needed.append(int(number))
    if not needed:
        return directory
    with _RemoteZip(XJTU_ARCHIVE_URL, XJTU_ARCHIVE_BYTES) as remote, zipfile.ZipFile(remote) as bundle:
        members = {}
        for info in bundle.infolist():
            parts = Path(info.filename).parts
            if len(parts) >= 2 and parts[-2] == record and parts[-1].endswith('.csv'):
                if '..' in parts or info.file_size > 4*1024*1024 or info.compress_size > 4*1024*1024:
                    raise ValueError('Invalid or oversized bearing CSV member')
                if parts[-1] in members:
                    raise ValueError('Duplicate bearing CSV member')
                members[parts[-1]] = info
        for number in needed:
            name = f'{number}.csv'
            if name not in members:
                raise ValueError(f'{record}/{name} missing from pinned mirror')
            info = members[name]
            raw = bundle.read(info)
            _parse_csv(raw)
            path = directory/name
            if path.is_file() and path.read_bytes() != raw:
                raise ValueError(f'Refusing to overwrite different local acquisition: {path}')
            temporary = path.with_suffix('.csv.part')
            temporary.write_bytes(raw); temporary.replace(path)
            manifest['files'][name] = dict(sha256=hashlib.sha256(raw).hexdigest(),
                zip_crc32=info.CRC, member=info.filename, bytes=len(raw))
            _write_manifest(manifest_path, manifest)
            if number == needed[0] or number == needed[-1] or number % 20 == 0:
                print(f'XJTU-SY cached {record}/{name}', flush=True)
    return directory


def load_xjtu_sy_windows(directory, record='Bearing1_1', *, window_samples=512,
                         window_offset=0, step=1, max_series=None):
    """One raw two-channel window per numbered burst; time is acquisition time.

    No filtering, spectral features, phase alignment, per-window normalization,
    interpolation or concatenation across the unobserved intervals is applied.
    RMS and peak of the full burst are evaluation/display metadata only.
    """
    if (not isinstance(window_samples, int) or not 2 <= window_samples <= XJTU_SAMPLES
            or not isinstance(window_offset, int) or window_offset < 0
            or window_offset+window_samples > XJTU_SAMPLES):
        raise ValueError('Raw window must fit entirely inside one acquisition')
    numbers = acquisition_numbers(record, step=step, max_series=max_series)
    root = Path(directory)
    condition, rpm, load = XJTU_CONDITIONS[int(record[7])]
    candidates = [root] if root.name == record else []
    candidates.extend([root/record, root/condition/record,
                       root/'XJTU-SY_Bearing_Datasets'/condition/record])
    found = list(dict.fromkeys(path for path in candidates if path.is_dir()))
    if len(found) != 1:
        raise FileNotFoundError(f'Expected one {record} directory under {root}; use --download or an extracted dataset')
    folder = found[0]
    manifest_path = folder/'download_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {'files': {}}
    values, rms, peak, hashes = [], [], [], []
    for number in numbers:
        path = folder/f'{number}.csv'
        if not path.is_file():
            raise FileNotFoundError(f'Missing acquisition {path}; gaps cannot be silently renumbered')
        raw = path.read_bytes(); sha = hashlib.sha256(raw).hexdigest()
        known = manifest['files'].get(path.name)
        if known and sha != known['sha256']:
            raise ValueError(f'Acquisition checksum mismatch: {path}')
        data = _parse_csv(raw)
        values.append(data[window_offset:window_offset+window_samples].reshape(-1))
        rms.append(np.sqrt(np.mean(data*data, axis=0))); peak.append(np.max(np.abs(data), axis=0))
        hashes.append(sha)
    # Time origin: first scheduled acquisition, not an assumed exact installation time.
    times = (numbers-1)*XJTU_PERIOD_SECONDS+(window_offset+window_samples-1)/XJTU_SAMPLE_RATE
    fingerprint = hashlib.sha256(json.dumps(list(zip(numbers.tolist(), hashes))).encode()).hexdigest()
    count, fault = XJTU_RECORDS[record]
    return XJTUBearingWindows(record, np.stack(values), numbers, times,
        np.full(len(numbers), window_offset), np.full(len(numbers), window_offset+window_samples-1),
        np.stack(rms), np.stack(peak), tuple(hashes), fingerprint, window_samples,
        count, fault, rpm, load)
