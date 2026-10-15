"""UCR archive access and TRAIN-only waveform normalization."""
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import numpy as np

ARCHIVE_URL = 'https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip'
ARCHIVE_PASSWORD = b'someone'


@dataclass(frozen=True)
class PreparedDataset:
    train: np.ndarray
    test: np.ndarray
    train_labels: np.ndarray
    test_labels: np.ndarray
    raw_points: int
    missing_rows: int
    center: float = 0.
    scale: float = 1.


def download_archive(path: Path) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.zip.part')
    with urlopen(ARCHIVE_URL, timeout=90) as response, temporary.open('wb') as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
    temporary.replace(path)


def archive_datasets(archive: ZipFile) -> dict[str, tuple[str, str]]:
    names = set(archive.namelist())
    pairs = {}
    for train in sorted(names):
        parts = train.split('/')
        if (len(parts) != 3 or parts[0] != 'UCRArchive_2018'
                or parts[2] != f'{parts[1]}_TRAIN.tsv'):
            continue
        test = f'UCRArchive_2018/{parts[1]}/{parts[1]}_TEST.tsv'
        if test not in names:
            raise ValueError(f'Missing TEST split for {parts[1]}')
        pairs[parts[1]] = (train, test)
    if not pairs:
        raise ValueError('Archive has no canonical UCR 2018 TRAIN/TEST pairs')
    return pairs


def _read_tsv(archive: ZipFile, member: str) -> tuple[np.ndarray, np.ndarray]:
    with archive.open(member, pwd=ARCHIVE_PASSWORD) as stream:
        table = np.genfromtxt(stream, delimiter='\t', dtype=np.float64,
                              missing_values=('?', 'NaN', 'nan'),
                              filling_values=np.nan)
    table = np.atleast_2d(table)
    if table.shape[1] < 3 or not len(table) or not np.isfinite(table[:, 0]).all():
        raise ValueError(f'{member} needs finite labels and at least two data points')
    return table[:, 1:], table[:, 0]


def _resample(rows: np.ndarray, points: int) -> tuple[np.ndarray, int]:
    result = np.empty((len(rows), points), dtype=np.float64)
    target = np.linspace(0.0, 1.0, points)
    source = np.linspace(0.0, 1.0, rows.shape[1])
    missing = 0
    for index, row in enumerate(rows):
        finite = np.isfinite(row)
        if finite.sum() < 2:
            raise ValueError(f'Series {index} has fewer than two finite points')
        missing += int(not finite.all())
        result[index] = np.interp(target, source[finite], row[finite])
    return result, missing


def prepare_dataset(archive: ZipFile, pair: tuple[str, str],
                    max_points: int) -> PreparedDataset:
    train_raw, train_labels = _read_tsv(archive, pair[0])
    test_raw, test_labels = _read_tsv(archive, pair[1])
    points = min(max_points, max(train_raw.shape[1], test_raw.shape[1]))
    if points < 3:
        raise ValueError('Need at least three waveform points')
    train, missing_train = _resample(train_raw, points)
    test, missing_test = _resample(test_raw, points)
    center = float(train.mean())
    scale = float(train.std())
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError('TRAIN series have zero or non-finite global variation')
    return PreparedDataset((train - center) / scale, (test - center) / scale,
                           train_labels, test_labels,
                           max(train_raw.shape[1], test_raw.shape[1]),
                           missing_train + missing_test, center, scale)
