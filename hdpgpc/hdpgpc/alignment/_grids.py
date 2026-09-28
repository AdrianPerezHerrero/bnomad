"""Private AMTGP monotone-grid calculations, retaining two distinct paths.

The scalar caller expands controls on its stored coordinates before calling
here. The batch path interpolates in index space. Keeping these calculations
separate preserves irregular-grid behavior, arithmetic order, and autograd.
"""

import torch
import torch.nn.functional as F


def monotone_grid_from_expanded_controls(uT, x_basis):
    """Transform already expanded scalar controls to a grid and displacement."""
    inc = torch.nn.functional.softplus(uT) + 1e-6  # positive increments
    g_raw = torch.cumsum(inc, dim=0)
    x_min = x_basis[0]
    x_max = x_basis[-1]
    g = (g_raw - g_raw[0]) / (g_raw[-1] - g_raw[0] + 1e-12)
    g = x_min + (x_max - x_min) * g
    x_warp = g - x_basis
    return g, x_warp


def batch_monotone_grid(u_ctrl_bt, x_model, T):
    """Expand batch controls to explicit length T, then apply the original map."""
    uT = F.interpolate(u_ctrl_bt[:, None, :], size=T, mode="linear", align_corners=True).squeeze(1)

    inc = F.softplus(uT) + 1e-6
    g_raw = torch.cumsum(inc, dim=1)

    x_min = x_model[0]
    x_max = x_model[-1]
    g = (g_raw - g_raw[:, [0]]) / (g_raw[:, [-1]] - g_raw[:, [0]] + 1e-12)
    g = x_min + (x_max - x_min) * g
    x_warp = g - x_model[None, :]
    return g, x_warp
