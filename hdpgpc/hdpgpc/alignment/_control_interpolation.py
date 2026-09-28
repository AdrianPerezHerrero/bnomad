"""Per-call scalar control interpolation constants; no cross-call cache.

The expressions are copied from _expand_ctrl_to_T/_lin_interp_1d. Only the
coordinate-only work moves out of the optimizer loop. Callers must use the
original path when basis coordinates require gradients.
"""
import torch


def prepare_control_interpolation(x_basis, n_ctrl, device, dtype):
    x0 = x_basis[0]
    x1 = x_basis[-1]
    x_ctrl = torch.linspace(float(x0), float(x1), steps=n_ctrl, device=device, dtype=dtype)
    xq = torch.clamp(x_basis, x_ctrl[0], x_ctrl[-1])
    idx_hi = torch.searchsorted(x_ctrl, xq, right=False)
    idx_hi = torch.clamp(idx_hi, 1, x_ctrl.numel() - 1)
    idx_lo = idx_hi - 1
    x_lo = x_ctrl[idx_lo]
    x_hi = x_ctrl[idx_hi]
    w = ((xq - x_lo) / (x_hi - x_lo + 1e-12)).unsqueeze(-1)
    return idx_lo, idx_hi, w


def expand_controls(u_ctrl, interpolation):
    idx_lo, idx_hi, w = interpolation
    y_grid = u_ctrl[:, None]
    y_lo = y_grid[idx_lo]
    y_hi = y_grid[idx_hi]
    return ((1.0 - w) * y_lo + w * y_hi)[:, 0]
