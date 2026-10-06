"""Optional, model-independent calibration of scalar LDS noise prior scales.

This is a pilot calculation on observations, before constructing an HDP-GPC.
It assumes an identity random-walk LDS and does not modify fitted models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LDSNoisePriorEstimate:
    """Scalar prior scales and diagnostics from the identity-LDS pilot model.

    ``observation`` and ``process`` can be passed as ``ini_sigma`` and
    ``ini_gamma``. They are variances: robust summaries of the estimated
    covariance diagonals, not standard deviations or full covariance matrices.
    """

    observation: float
    process: float
    observation_diagonal: np.ndarray
    process_diagonal: np.ndarray
    difference_variance: np.ndarray
    difference_lag_covariance: np.ndarray
    segment_starts: tuple[int, ...]
    segment_length: int
    negative_lag_fraction: float


def estimate_lds_noise_priors(
    observations: np.ndarray,
    *,
    max_beats: int = 256,
    segment_length: int = 32,
    random_state: int = 0,
) -> LDSNoisePriorEstimate:
    """Estimate isotropic Gamma/Sigma prior scales from a contiguous minibatch.

    With ``f_t = f_(t-1) + eta_t`` and ``y_t = f_t + epsilon_t``, independent
    white noises with covariances Gamma and Sigma imply

        Var(y_t - y_(t-1)) = Gamma + 2 Sigma
        Cov(y_t - y_(t-1), y_(t-1) - y_(t-2)) = -Sigma.

    The function estimates these moments separately for every waveform point
    and output, then uses the median diagonal estimate as the scalar prior
    scale, provided that median is positive. It samples nonoverlapping blocks
    of consecutive beats so lagged pairs
    never cross a block boundary. It does not know the true latent states or
    cluster boundaries. A nonnegative median lag covariance or nonpositive
    process estimate indicates the pilot model cannot separate the noises;
    this raises ValueError rather than silently substituting an arbitrary
    fraction of waveform variance.

    The selected data should represent typical within-cluster behavior. For
    online inference, supply an independent calibration record or prefix, not
    future observations from the stream being evaluated.
    """
    values = np.asarray(observations, dtype=np.float64)
    if values.ndim == 2:
        values = values[:, :, None]
    if values.ndim != 3 or values.shape[1] < 1 or values.shape[2] < 1:
        raise ValueError("observations must have shape (beats, points[, outputs])")
    if values.shape[0] < 3:
        raise ValueError("at least three consecutive beats are required")
    if not np.isfinite(values).all():
        raise ValueError("observations must be finite")
    if not isinstance(max_beats, int) or max_beats < 3:
        raise ValueError("max_beats must be an integer of at least 3")
    if not isinstance(segment_length, int) or segment_length < 3:
        raise ValueError("segment_length must be an integer of at least 3")

    length = min(segment_length, values.shape[0], max_beats)
    candidate_starts = np.arange(0, values.shape[0] - length + 1, length)
    n_blocks = min(candidate_starts.size, max_beats // length)
    if n_blocks * (length - 2) < 16:
        raise ValueError(
            "calibration needs at least 16 adjacent-difference pairs; "
            "increase max_beats or supply more consecutive observations"
        )
    rng = np.random.default_rng(random_state)
    starts = tuple(sorted(int(i) for i in rng.choice(
        candidate_starts, size=n_blocks, replace=False,
    )))
    differences = [np.diff(values[start:start + length], axis=0)
                   for start in starts]
    all_differences = np.concatenate(differences, axis=0)
    mean_difference = all_differences.mean(axis=0)
    difference_variance = np.mean(
        (all_differences - mean_difference) ** 2, axis=0,
    )
    lag_products = [
        (delta[1:] - mean_difference) * (delta[:-1] - mean_difference)
        for delta in differences
    ]
    difference_lag_covariance = np.mean(
        np.concatenate(lag_products, axis=0), axis=0,
    )

    observation_diagonal = -difference_lag_covariance
    process_diagonal = difference_variance + 2.0 * difference_lag_covariance
    observation = float(np.median(observation_diagonal))
    process = float(np.median(process_diagonal))
    if observation <= 0 or process <= 0:
        raise ValueError(
            "identity random-walk noise moments are not positive: "
            f"median Sigma={observation:.6g}, median Gamma={process:.6g}; "
            "select typical contiguous segments or use explicit prior scales"
        )
    return LDSNoisePriorEstimate(
        observation=observation,
        process=process,
        observation_diagonal=observation_diagonal,
        process_diagonal=process_diagonal,
        difference_variance=difference_variance,
        difference_lag_covariance=difference_lag_covariance,
        segment_starts=starts,
        segment_length=length,
        negative_lag_fraction=float(np.mean(difference_lag_covariance < 0)),
    )
