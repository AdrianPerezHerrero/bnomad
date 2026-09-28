"""Private HDP assignment entropy terms, copied without renormalization.

These accept the NumPy arrays used by the legacy objective callers. Despite the
legacy Hstart name, the state term sums all supplied rows, not just the first.
Pair normalization uses an epsilon in the denominator and another before log.
"""

import numpy as np


def state_entropy_terms(resp, eps=1e-30):
    """Return the existing per-state terms without mutating responsibilities."""
    firstHvec = -1 * np.sum(resp * np.log(resp + eps), axis=0)
    return firstHvec


def pair_entropy_terms(respPair, eps=1e-30):
    """Return the existing pair terms; only the newly allocated sigma is mutated."""
    sigma = respPair / (respPair.sum(axis=2)[:, :, np.newaxis] + eps)
    sigma += eps  # make it safe for taking logs!
    logsigma = sigma  # alias
    np.log(logsigma, out=logsigma)
    H_KxK = -1 * np.sum(respPair * logsigma, axis=0)
    return H_KxK
