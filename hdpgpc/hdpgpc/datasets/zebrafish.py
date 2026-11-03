"""White et al. (2017) zebrafish RNA-seq snapshots and developmental targets.

All genes in supplementary file 2 are retained, including zero/constant rows.
Stage names organize simultaneous biological replicates; no expression-based
gene selection or synthetic ordering of replicate samples is performed.
"""
from dataclasses import dataclass
import csv
import hashlib
from pathlib import Path
from urllib.request import urlopen

import numpy as np


ZEBRAFISH_STAGES = (
    ('1-cell', 0.), ('2-cell', .75), ('128-cell', 2.25), ('1k-cell', 3.),
    ('Dome', 4.3), ('50pc-epiboly', 5.25), ('Shield', 6.), ('75pc-epiboly', 8.),
    ('1-4-somites', 10.3), ('14-19-somites', 16.), ('20-25-somites', 19.),
    ('Prim-5', 24.), ('Prim-15', 30.), ('Prim-25', 36.), ('Long-pec', 48.),
    ('Protruding-mouth', 72.), ('Day-4', 96.), ('Day-5', 120.),
)
# Approximate hours post-fertilisation, Figure 1A, doi:10.7554/eLife.30860.
ZEBRAFISH_FILES = ('elife-30860-supp1-v1.tsv', 'elife-30860-supp2-v1.tsv')
ZEBRAFISH_URL = 'https://cdn.elifesciences.org/articles/30860/'
ZEBRAFISH_SHA256 = {
    ZEBRAFISH_FILES[0]: '01b48e4b61368370c7a6da4176d64d0c6dea622f201b0b6f38012cb001e44563',
    ZEBRAFISH_FILES[1]: '029a678ce1ccaa5a8ab552206c8d6ab4c566002cd67d73149b897504cde16783',
}


@dataclass(frozen=True)
class ZebrafishExpression:
    gene_ids: np.ndarray
    gene_names: np.ndarray
    sample_names: np.ndarray
    sample_stages: np.ndarray
    sample_categories: np.ndarray
    counts: np.ndarray              # samples x genes, unfiltered source values
    stages: np.ndarray
    hours: np.ndarray
    categories: np.ndarray
    source_sha256: dict
    counting_summary: dict           # HTSeq bookkeeping, never expression features

    def stage_values(self):
        """Mean log1p(CPM) per sampling stage, with between-replicate SD.

        Log transformation precedes aggregation. Retain every gene and use
        the known sampling chronology, never the source table's row order.
        """
        library_sizes = self.counts.sum(axis=1, keepdims=True)
        if np.any(library_sizes <= 0):
            raise ValueError('Expression samples require positive library sizes')
        transformed = np.log1p(self.counts / library_sizes * 1e6)
        groups = [transformed[self.sample_stages == stage] for stage in self.stages]
        return (np.stack([group.mean(axis=0) for group in groups]),
                np.stack([group.std(axis=0, ddof=1) if len(group) > 1
                          else np.zeros(group.shape[1]) for group in groups]),
                np.array([len(group) for group in groups], dtype=np.int64))

    def prepare_timecourse(self):
        from ._expression import ExpressionTimecourse
        values, deviations, replicates = self.stage_values()
        return ExpressionTimecourse('zebrafish_development', 'Zebrafish',
            self.gene_ids, self.gene_names, values, self.hours, self.stages,
            {'broad': self.categories, 'stage': self.stages}, self.source_sha256,
            dict(sample_names=self.sample_names,
                sample_stages=self.sample_stages, library_sizes=self.counts.sum(axis=1),
                **{f'htseq{key}': value for key, value in self.counting_summary.items()}),
            dict(dataset_accession='E-ERAD-475 / ERP014517', source_doi='10.7554/eLife.30860',
                source_samples=len(self.sample_names), non_gene_counting_rows=list(self.counting_summary),
                aggregation='mean of log1p(CPM) across simultaneous biological replicates',
                expression_units='log1p(CPM)',
                timing_source='Approximate hours post-fertilisation, White et al. Figure 1A',
                target_source='Supplied developmental stage names and categories'), {},
            dict(mean_log1p_cpm=values, replicate_sd_log1p_cpm=deviations,
                replicate_counts=replicates, stages=self.stages, categories=self.categories,
                hours_post_fertilisation=self.hours))


def download_zebrafish(directory):
    """Cache versioned metadata/unfiltered counts; never download FASTQs."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in ZEBRAFISH_FILES:
        path = directory / name
        if path.is_file():
            if hashlib.sha256(path.read_bytes()).hexdigest() != ZEBRAFISH_SHA256[name]:
                raise ValueError(f'Zebrafish checksum mismatch: {path}')
            continue
        temporary = path.with_suffix(path.suffix + '.part')
        try:
            with urlopen(ZEBRAFISH_URL + name, timeout=60) as source, temporary.open('wb') as target:
                size = 0
                while block := source.read(1024 * 1024):
                    size += len(block)
                    if size > 128 * 1024 * 1024:
                        raise ValueError('Zebrafish supplement exceeded 128 MB')
                    target.write(block)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != ZEBRAFISH_SHA256[name]:
                raise ValueError(f'Zebrafish checksum mismatch: {name}')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return directory


def load_zebrafish_expression(directory):
    """Join RNA-seq metadata to count columns by published sample ID.

    DeTCT samples are excluded by assay, not expression. Missing/nonfinite or
    negative counts, duplicate genes/sample IDs, and unknown stages fail loudly.
    Source SHA256 values participate in experiment provenance.
    """
    directory = Path(directory)
    paths = [directory / name for name in ZEBRAFISH_FILES]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError(f'Zebrafish supplements absent in {directory}; use --download')
    with paths[0].open(newline='', encoding='utf-8-sig') as stream:
        records = list(csv.DictReader(stream, delimiter='\t'))
    required = {'sequencing', 'sample', 'stageName', 'stage'}
    if not records or not required.issubset(records[0]):
        raise ValueError('Zebrafish metadata lacks required sample/stage columns')
    records = [row for row in records if row['sequencing'] == 'RNASeq']
    if not records or len({row['sample'] for row in records}) != len(records):
        raise ValueError('RNASeq metadata requires unique sample IDs')
    times = dict(ZEBRAFISH_STAGES)
    if any(row['stageName'] not in times or ':' not in row['stage'] for row in records):
        raise ValueError('Unknown zebrafish stage or missing developmental category')
    records.sort(key=lambda row: (times[row['stageName']], row['sample']))
    sample_names = np.array([row['sample'] for row in records])
    with paths[1].open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.reader(stream, delimiter='\t')
        header = next(reader, [])
        if (len(set(header)) != len(header) or 'Gene ID' not in header or 'Name' not in header
                or not set(sample_names).issubset(header)):
            raise ValueError('Count columns must uniquely match metadata sample IDs')
        columns = [header.index(name) for name in sample_names]
        gene_column, name_column = header.index('Gene ID'), header.index('Name')
        ids, names, values, counting_summary = [], [], [], {}
        for row in reader:
            if len(row) != len(header):
                raise ValueError('Malformed zebrafish count row')
            if row[gene_column].startswith('__'):
                key = row[gene_column]
                if key in counting_summary:
                    raise ValueError('Duplicate HTSeq counter')
                counting_summary[key] = np.array([float(row[column]) for column in columns])
                continue
            ids.append(row[gene_column]); names.append(row[name_column])
            values.append([float(row[column]) for column in columns])
    counts = np.asarray(values, dtype=np.float64).T
    if (not ids or len(set(ids)) != len(ids) or any(not gene for gene in ids)
            or not np.isfinite(counts).all() or np.any(counts < 0)):
        raise ValueError('Counts require unique gene IDs and finite nonnegative values')
    sample_stages = np.array([row['stageName'] for row in records])
    sample_categories = np.array([row['stage'].split(':')[0] for row in records])
    stages = np.array([stage for stage, _ in ZEBRAFISH_STAGES if stage in sample_stages])
    categories = []
    for stage in stages:
        labels = np.unique(sample_categories[sample_stages == stage])
        if len(labels) != 1:
            raise ValueError(f'Conflicting developmental categories for {stage}')
        categories.append(labels[0])
    return ZebrafishExpression(np.array(ids), np.array(names), sample_names,
        sample_stages, sample_categories, counts, stages,
        np.array([times[stage] for stage in stages]), np.array(categories),
        {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}, counting_summary)
