"""Maximum-shift operations preserving the legacy LogLik contract.

This is not log-sum-exp normalization. Infinite maxima bypass subtraction for
the entire input, and axes other than zero/one leave the input unshifted.
"""

import numpy as np
import torch


def max_shift_log_scores(logSoftEv, axis=1):
    """Return shifted scores and maxima without modifying the input.

    Preserve the original exact Tensor type check, NumPy scalar special case,
    broadcasting, dtype/device selection, and input alias on bypass branches.
    """
    if type(logSoftEv) is torch.Tensor:
        lognormC = torch.max(logSoftEv, dim=axis)[0]
        if torch.any(torch.isinf(lognormC)):
            return logSoftEv, lognormC
    else:
        lognormC = np.max(logSoftEv, axis)
        if np.any(np.isinf(lognormC)):
            return logSoftEv, lognormC
    if axis == 0:
        if type(lognormC) is np.float64:
            lognormC = np.array([lognormC])
        SoftEv = logSoftEv - lognormC[np.newaxis, :]
    elif axis == 1:
        SoftEv = logSoftEv - lognormC[:, np.newaxis]
    else:
        SoftEv = logSoftEv
    return SoftEv, lognormC
