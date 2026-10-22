"""Typed, immutable snapshots returned by the MFA facade."""
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, eq=False)
class MFAFactors:
    """Separate local coordinates: mean (N,K,r), covariance (K,r,r).

    Arrays returned by HDPMFA are independent read-only snapshots. Components
    need not share a coordinate basis; do not average their coordinates.
    """
    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]
