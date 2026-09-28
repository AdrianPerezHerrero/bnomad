"""Small public scientific snapshots; arrays never alias the live model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy import ndarray


def _snapshot(value):
    import numpy as np
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, eq=False)
class Prediction:
    """Current component observation distribution from legacy observe_last.

    mean: (points, outputs); covariance: (outputs, points, points). There is no
    cross-output covariance or added LDS forecast step. In particular, at the
    native grid the existing code returns observation Sigma, not an independently
    reconstructed total predictive covariance. See docs/public_api.md.
    """
    cluster: int
    grid: ndarray
    mean: ndarray
    covariance: ndarray


@dataclass(frozen=True, eq=False)
class LatentState:
    """One output's latest stored filtered/smoothed state on its native basis.

    Means have shape (basis_points,), covariances (basis_points, basis_points).
    observation_index is the last global sample index included in this component.
    """
    output: int
    observation_index: int
    grid: ndarray
    filtered_mean: ndarray
    filtered_covariance: ndarray
    smoothed_mean: ndarray
    smoothed_covariance: ndarray


@dataclass(frozen=True, eq=False)
class UpdateResult:
    """One committed online update, with optional copied scientific snapshots.

    responsibilities is the retained hard assignment, indexed by cluster_ids.
    assignment_probabilities is None: the existing schedules do not retain
    calibrated soft probabilities. Do not interpret hard zeros as uncertainty.
    new_cluster means allocated cluster count increased during this update;
    the first constructor-initialized cluster is therefore not a new birth.
    Cluster numbers are current indices and may be reassigned by later inference.
    """
    observation_index: int
    assigned_cluster: int
    cluster_ids: ndarray
    responsibilities: ndarray
    new_cluster: bool
    assignment_probabilities: ndarray | None = None
    prediction: Prediction | None = None
    posterior: tuple[LatentState, ...] | None = None
