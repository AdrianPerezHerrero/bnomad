"""CAS-Colon extracted frames, ordered by acquisition second rather than label.

Expects the extracted Frame archive and Label/frame/Frame_Label.csv from
https://doi.org/10.6084/m9.figshare.28287929. Labels are evaluation metadata;
neither filenames nor annotations become model features.
"""
from dataclasses import dataclass
import csv
from pathlib import Path
import re

import numpy as np


CAS_COLON_REGIONS = ('terminalIleum', 'cecum', 'ascendingColon', 'hepaticFlexure',
                    'transverseColon', 'splenicFlexure', 'descendingColon',
                    'sigmoidColon', 'rectum', 'analCanal')
_FRAME_NAME = re.compile(r'(?P<video>\d+)_(?P<region>[A-Za-z]+)_(?P<second>\d+)s\.(jpg|jpeg|png)', re.I)


@dataclass(frozen=True)
class CASColonFrame:
    filename: str
    video: str
    second: int
    annotation: str
    variants: tuple[str, ...] = ()
    annotation_options: tuple[str, ...] = ()


@dataclass(frozen=True)
class CASColonFrames:
    images: np.ndarray
    seconds: np.ndarray
    annotations: np.ndarray
    filenames: tuple[str, ...]
    video: str
    annotation_options: tuple[tuple[str, ...], ...] = ()
    source_variants: tuple[tuple[str, ...], ...] = ()

    @property
    def values(self):
        return self.images.reshape(len(self.images), -1, 1)

    @property
    def evaluation_mask(self):
        return self.annotations != 'ambiguous'


def cas_colon_annotations(directory, annotations=None):
    """Read the official frame CSV, accepting its public repository aliases.

    Return the resolved CSV and video -> chronologically ordered frame records.
    Repeated timestamps are grouped for image identity checks during loading.
    Conflicting anatomical labels are retained as metadata, never guessed.
    """
    root = Path(directory)
    if annotations is None:
        candidates = [root / 'Label/frame/Frame_Label.csv', root / 'Frame_Label.csv']
        candidates = [path for path in candidates if path.is_file()]
        if len(candidates) != 1:
            raise FileNotFoundError('Provide --annotations pointing to Label/frame/Frame_Label.csv')
        path = candidates[0]
    else:
        path = Path(annotations)
    timestamps = {}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        columns = {re.sub(r'[^a-z0-9]', '', name.lower()): name for name in reader.fieldnames or ()}
        video_column = columns.get('videoid', columns.get('video'))
        if not {'filename', 'labelraw'} <= columns.keys() or video_column is None:
            raise ValueError('Frame CSV requires Filename, Label_raw and VideoID (or video_id) columns')
        for number, row in enumerate(reader, 2):
            filename = row[columns['filename']].strip()
            match = _FRAME_NAME.fullmatch(filename)
            if match is None:
                raise ValueError(f'{path}:{number}: unsupported frame filename {filename!r}')
            video = str(int(row[video_column]))
            region = row[columns['labelraw']].strip()
            if (str(int(match['video'])) != video or match['region'] != region
                    or region not in CAS_COLON_REGIONS):
                raise ValueError(f'{path}:{number}: inconsistent video or anatomical label')
            second = int(match['second'])
            timestamps.setdefault((video, second), []).append((filename, region))
    if not timestamps:
        raise ValueError('CAS-Colon frame CSV is empty')
    grouped = {}
    for (video, second), rows in timestamps.items():
        variants = tuple(sorted({filename for filename, _ in rows}))
        options = tuple(sorted({region for _, region in rows}))
        grouped.setdefault(video, []).append(CASColonFrame(
            variants[0], video, second, options[0] if len(options) == 1 else 'ambiguous',
            variants, options))
    return path, {video: tuple(sorted(frames, key=lambda frame: frame.second))
                  for video, frames in grouped.items()}


def load_cas_colon_frames(directory, video, *, annotations=None, image_size=(32, 32),
                         crop=None, frame_step=1, max_frames=None, gap_policy='split',
                         frame_records=None):
    """Load one video's grayscale images in [0,1], returning contiguous segments.

    image_size is (height, width); crop is an optional pixel (left, top, right,
    bottom) box applied before resizing. Seconds are the original filename's
    one-second bins, not an inferred frame counter. Selection is by second modulo
    frame_step; gaps start fresh segments, or raise with gap_policy='error'.
    frame_records may reuse the index returned by cas_colon_annotations.
    """
    from PIL import Image

    height, width = image_size
    if (height < 2 or width < 2 or frame_step < 1 or gap_policy not in ('split', 'error')
            or (max_frames is not None and max_frames < 2)):
        raise ValueError('Invalid image dimensions, frame step, subset or gap policy')
    if crop is not None and (len(crop) != 4 or min(crop) < 0
                             or crop[0] >= crop[2] or crop[1] >= crop[3]):
        raise ValueError('crop must be a nonempty (left, top, right, bottom) pixel box')
    video = str(int(video))
    if frame_records is None:
        _, frame_records = cas_colon_annotations(directory, annotations)
    if video not in frame_records:
        raise ValueError(f'Video {video} is absent from the annotation table')
    selected = [frame for frame in frame_records[video] if frame.second % frame_step == 0]
    if max_frames is not None:
        selected = selected[:max_frames]
    if len(selected) < 2:
        raise ValueError(f'Video {video} has fewer than two selected frames')
    root = Path(directory)
    image_root = root / 'Frame' if (root / 'Frame').is_dir() else root
    needed = {filename for frame in selected for filename in (frame.variants or (frame.filename,))}
    paths = {}
    for path in image_root.rglob('*'):
        if path.is_file() and path.name in needed:
            if path.name in paths:
                raise ValueError(f'Duplicate image basename {path.name}: use a single extracted Frame tree')
            paths[path.name] = path
    missing = sorted(needed - paths.keys())
    if missing:
        raise FileNotFoundError(f'{len(missing)} selected CAS-Colon images missing; first: {missing[0]}. '
                                'Extract Frame.rar under --data-dir/Frame.')
    images = []
    for frame in selected:
        variants = frame.variants or (frame.filename,)
        if len(variants) > 1:
            reference_bytes = paths[frame.filename].read_bytes()
            for filename in variants[1:]:
                if paths[filename].read_bytes() != reference_bytes:
                    # Metadata/JPEG encodings can differ while decoded pixels
                    # agree. Compare before cropping/resizing hides differences.
                    with Image.open(paths[frame.filename]) as first, Image.open(paths[filename]) as other:
                        same = np.array_equal(np.asarray(first.convert('RGB')),
                                              np.asarray(other.convert('RGB')))
                    if not same:
                        raise ValueError(f'Conflicting images at video {video}, second {frame.second}: '
                                         f'{frame.filename}, {filename}')
        with Image.open(paths[frame.filename]) as source:
            if crop is not None:
                if crop[2] > source.width or crop[3] > source.height:
                    raise ValueError(f'Crop exceeds image dimensions: {frame.filename}')
                source = source.crop(tuple(crop))
            resized = source.convert('L').resize((width, height), Image.Resampling.LANCZOS)
            images.append(np.asarray(resized, dtype=np.float64) / 255.)
    seconds = np.array([frame.second for frame in selected], dtype=np.int64)
    breaks = np.flatnonzero(np.diff(seconds) != frame_step) + 1
    if len(breaks) and gap_policy == 'error':
        raise ValueError('Selected frames contain time gaps; use --gap-policy split or a complete sequence')
    images = np.stack(images)
    labels = np.array([frame.annotation for frame in selected])
    return [CASColonFrames(images[indices], seconds[indices], labels[indices],
                tuple(selected[index].filename for index in indices), video,
                tuple(selected[index].annotation_options or (selected[index].annotation,) for index in indices),
                tuple(selected[index].variants or (selected[index].filename,) for index in indices))
            for indices in np.split(np.arange(len(selected)), breaks)]
