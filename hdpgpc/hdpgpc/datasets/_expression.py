"""Prepared expression time courses shared by dataset-specific loaders."""
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ExpressionTimecourse:
    name: str
    title: str
    gene_ids: np.ndarray
    gene_names: np.ndarray
    values: np.ndarray
    hours: np.ndarray
    stage_names: np.ndarray
    targets: dict
    source_sha256: dict
    prepared_arrays: dict
    metadata: dict
    evaluation_masks: dict
    time_arrays: dict = field(default_factory=dict)
