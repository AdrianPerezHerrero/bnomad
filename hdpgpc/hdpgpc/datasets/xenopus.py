"""Unsmoothed polyA+ expression time courses from Owens et al. (2016).

Retain every endogenous Gene row. Isoform rows would duplicate gene abundance;
ERCC/ArrayControl rows are exogenous calibration standards, saved separately.
Clutches are independent sequences. No gene selection uses expression/variance.
"""
import csv
import gzip
import hashlib
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from ._expression import ExpressionTimecourse


XENOPUS_URL = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE65nnn/GSE65785/suppl/'
XENOPUS_FILES = {
    'A': 'GSE65785_clutchApolyA_relative_TPM_gene_isoform.txt.gz',
    'B': 'GSE65785_clutchB_polyA_relative_TPM_gene_isoform.txt.gz',
}
XENOPUS_SHA256 = {
    'A': '28a779e4e94ea7d3d18422313e7ec194e377d3f20f0fdd28beeadd58f01551e9',
    'B': 'c6ef8098da46f27ae98a38785de8ac0de5787805b56a7cc5017e569ccc9f2c3a',
}
# Observed NF stages/time anchors, Supplemental Table S1, not interpolated.
# https://pmc.ncbi.nlm.nih.gov/articles/instance/4731879/bin/NIHMS746658-supplement-6.pdf
# Source PDF SHA256: 647b69784fc36a563f088d533343f655159346f29452b9c75641e924ff8bdb09
XENOPUS_NF_STAGES = (
    (0., '1'), (1., '2'), (1.33, '3'), (1.67, '4'), (2., '5'), (2.33, '6'),
    (3., '7'), (4., '8'), (4.5, '9'), (6.5, '10'), (8., '11'), (8.5, '12'),
    (10.5, '12.5'), (12., '13'), (13.5, '14'), (14.5, '15'), (15., '16/17'),
    (15.5, '18/19'), (16., '20'), (16.5, '21'), (17.5, '22'), (18.5, '23'),
    (19., '24'), (21., '25'), (22.5, '26'), (24., '27'), (27., '28'),
    (29., '29/30'), (30., '31'), (31., '32'), (33., '33/34'), (36., '35'),
    (37., '36'), (38., '37'), (39., '38'), (41., '39'), (46., '40'),
    (51., '41'), (56., '42'),
)
XENOPUS_QC_FAILURES = {'A': (21.5, 33., 52.), 'B': (22.5,)}


def download_xenopus(directory, *, clutches=('A',)):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for clutch in dict.fromkeys(clutches):
        if clutch not in XENOPUS_FILES:
            raise ValueError('Xenopus clutch must be A or B')
        path = directory / XENOPUS_FILES[clutch]
        if path.is_file():
            if hashlib.sha256(path.read_bytes()).hexdigest() != XENOPUS_SHA256[clutch]:
                raise ValueError(f'Xenopus checksum mismatch: {path}')
            continue
        temporary = path.with_suffix(path.suffix + '.part')
        try:
            with urlopen(XENOPUS_URL + path.name, timeout=60) as source, temporary.open('wb') as target:
                size = 0
                while block := source.read(1024*1024):
                    size += len(block)
                    if size > 128*1024*1024:
                        raise ValueError('Xenopus expression download exceeded 128 MB')
                    target.write(block)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != XENOPUS_SHA256[clutch]:
                raise ValueError(f'Xenopus checksum mismatch: {path.name}')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return directory


def _targets(hours):
    # Coarse time-interval references, explicitly derived from Table S1 anchors.
    # They are not independently measured per-sample biological ground truth.
    edges = np.array([3., 6.5, 12., 17.5, 51.])
    classes = np.array(['NF1-6 interval', 'NF7-9 interval', 'NF10-12.5 interval',
                        'NF13-21 interval', 'NF22-40 interval', 'NF41+ interval'])
    broad = classes[np.searchsorted(edges, hours, side='right')]
    stage = np.full(len(hours), 'unannotated', dtype='<U20')
    annotated = np.zeros(len(hours), dtype=bool)
    for hour, label in XENOPUS_NF_STAGES:
        matched = np.isclose(hours, hour, atol=1e-8, rtol=0)
        stage[matched] = 'NF' + label
        annotated |= matched
    return {'broad': broad, 'stage': stage}, {'stage': annotated}


def load_xenopus_expression(directory, clutch='A', *, exclude_qc_failures=True):
    """Load measured gene TPMs; retain source times and exclude published QC failures.

    No interpolation, GP smoothing, sample concatenation or variance filtering.
    Coarse evaluation targets are time-derived intervals. Exact NF labels only
    cover matching Table S1 anchors; other times are excluded from NF scoring.
    """
    if clutch not in XENOPUS_FILES:
        raise ValueError('Xenopus clutch must be A or B')
    path = Path(directory) / XENOPUS_FILES[clutch]
    if not path.is_file():
        raise FileNotFoundError(f'Xenopus expression file absent: {path}; use --download')
    ids, abundance, controls = [], [], {}
    with gzip.open(path, 'rt', newline='') as stream:
        reader = csv.reader(stream, delimiter='\t')
        header, time_row = next(reader, []), next(reader, [])
        samples = np.asarray(header[2:])
        if len(header) < 4 or len(time_row) != len(header) or len(set(samples)) != len(samples):
            raise ValueError('Xenopus expression requires unique samples and aligned time header')
        hours = np.asarray(time_row[2:], dtype=np.float64)
        if not np.isfinite(hours).all() or np.any(hours < 0) or len(np.unique(hours)) != len(hours):
            raise ValueError('Xenopus times must be finite, nonnegative and unique')
        for row in reader:
            if len(row) != len(header) or row[0] not in ('Gene', 'Isoform'):
                raise ValueError('Malformed Xenopus expression row')
            if row[0] == 'Isoform':
                continue
            if not row[1]:
                raise ValueError('Xenopus gene identifiers must be nonempty')
            values = np.asarray(row[2:], dtype=np.float64)
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError('Xenopus TPMs must be finite and nonnegative')
            if row[1].startswith(('ERCC-', 'AC_EC')):
                if row[1] in controls:
                    raise ValueError('Duplicate Xenopus control identifier')
                controls[row[1]] = values
            else:
                ids.append(row[1]); abundance.append(values)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Xenopus expression requires unique gene identifiers')
    order = np.argsort(hours)
    hours, samples = hours[order], samples[order]
    tpm = np.asarray(abundance).T[order]
    source_hours = hours.copy()
    keep = ~np.isin(hours, XENOPUS_QC_FAILURES[clutch]) if exclude_qc_failures else np.ones(len(hours), bool)
    excluded = hours[~keep].copy()
    hours, samples, tpm = hours[keep], samples[keep], tpm[keep]
    if len(hours) < 2:
        raise ValueError('Xenopus time course requires at least two usable observations')
    values = np.log1p(tpm)
    targets, masks = _targets(hours)
    names = np.array([identifier.partition('|')[2] or identifier.partition('|')[0] for identifier in ids])
    stage_names = np.array([f'{hour:g}h' for hour in hours])
    return ExpressionTimecourse(f'xenopus_clutch_{clutch}', f'Xenopus clutch {clutch}',
        np.array(ids), names, values, hours, stage_names, targets,
        {path.name: hashlib.sha256(path.read_bytes()).hexdigest()},
        dict(excluded_qc_hours=excluded, source_hours=source_hours,
            control_ids=np.array(list(controls))),
        dict(dataset_accession='GSE65785', source_doi='10.1016/j.celrep.2015.12.050', clutch=clutch,
            source_samples=len(source_hours), aggregation='none; one measured polyA+ sample per time and clutch',
            expression_units='log1p(TPM)', timing_source='Measured hpf in the unsmoothed expression-table header',
            target_source='Table S1 NF anchors; broad targets are time-derived intervals, not independent annotations',
            target_labels={'broad': 'Time-derived NF intervals', 'stage': 'Published NF anchors'},
            annotation_source_sha256='647b69784fc36a563f088d533343f655159346f29452b9c75641e924ff8bdb09',
            source_gene_rows=len(ids)+len(controls), calibration_control_rows=len(controls),
            excluded_qc_hours=excluded, qc_policy='Supplemental methods: A 21.5/33/52 hpf; B 22.5 hpf',
            author_smoothed_expression=False), masks,
        dict(log1p_tpm=values, tpm=tpm, replicate_counts=np.ones(len(hours), dtype=np.int64),
            hours_post_fertilisation=hours, stages=stage_names, categories=targets['broad'],
            nf_stages=targets['stage'], nf_evaluation_included=masks['stage'], sample_names=samples,
            control_tpm=np.stack(list(controls.values()))[:, order][:, keep].T if controls else np.empty((len(hours), 0))))
