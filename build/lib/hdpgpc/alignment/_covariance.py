"""Private AMTGP displacement-prior covariance calculation.

This preserves the prior's coordinate normalization, amplitude convention and
fixed noise/jitter addition. Parameter parsing, bounds and cached factorizations
remain owned by WarpPriorAMTGP; this function neither reads nor updates a cache.
"""

import torch


def warp_prior_rbf_covariance(x, rho, omega, noise2, normalize_x, jitter):
    """Build the original RBF covariance on the supplied tensor's dtype/device."""
    if normalize_x:
        x0 = x[0]
        xr = x - x0
        rng = (xr[-1] - xr[0]).abs() + 1e-12
        x_use = xr / rng
    else:
        x_use = x

    dx = x_use[:, None] - x_use[None, :]
    K = (omega * omega) * torch.exp(-0.5 * (dx * dx) / (rho * rho))
    K = K + (noise2 + jitter) * torch.eye(x.numel(), device=x.device, dtype=x.dtype)
    return K
