"""Fast deterministic DTW alignment with a low-rank GP displacement prior.

This is a two-stage point estimate, not the variational GP-flow posterior.
The DTW path supplies noisy displacement observations; a whitened inducing
point GP gives their regularized posterior mean. No ODE or optimizer is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ._identity import _ZeroWarpScore


def _dtw_displacement(observed, template, grid, points):
    """Map observation indices to template coordinates using an unrestricted DTW path."""
    length = len(grid)
    take = np.rint(np.linspace(0, length - 1, min(points, length))).astype(int)
    x = grid[take]
    a = observed[take]
    b = template[take]
    # Alignment uses shape only. The untouched values enter the GP likelihood.
    a = (a - a.mean(0)) / np.maximum(a.std(0), 1e-6)
    b = (b - b.mean(0)) / np.maximum(b.std(0), 1e-6)
    cost = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
    n = len(x)
    accumulated = np.full((n + 1, n + 1), np.inf)
    previous = np.zeros((n, n), dtype=np.uint8)
    accumulated[0, 0] = 0.0
    for i in range(n):
        for j in range(n):
            best = accumulated[i, j]
            direction = 0  # prefer diagonal in ties
            if accumulated[i, j + 1] < best:
                best = accumulated[i, j + 1]
                direction = 1
            if accumulated[i + 1, j] < best:
                best = accumulated[i + 1, j]
                direction = 2
            accumulated[i + 1, j + 1] = cost[i, j] + best
            previous[i, j] = direction
    mapped = [[] for _ in range(n)]
    i = j = n - 1
    while True:
        mapped[i].append(j)
        if i == 0 and j == 0:
            break
        direction = previous[i, j]
        if direction == 0:
            i -= 1
            j -= 1
        elif direction == 1:
            i -= 1
        else:
            j -= 1
    # Retain sub-grid matches, including paths with repeated template indices.
    matched = np.interp([np.mean(indices) for indices in mapped],
                        np.arange(n), x)
    return np.interp(grid, x, matched) - grid


def _monotone_fixed_ends(grid, mapped):
    """Make the smoothed map strictly increasing and preserve its end points."""
    step = (grid[-1] - grid[0]) / (len(grid) - 1) * 1e-3
    index = np.arange(len(grid))
    shifted = np.maximum.accumulate(mapped - step * index) + step * index
    spread = shifted[-1] - shifted[0]
    return grid[0] + (grid[-1] - grid[0]) * (shifted - shifted[0]) / spread


@dataclass
class PointWarpResult:
    grid_samples: torch.Tensor       # one deterministic grid, (1,T,1)
    mean_displacement: torch.Tensor  # (T,1)
    std_displacement: torch.Tensor   # zero: not a posterior uncertainty
    kl: torch.Tensor                 # final-map GP quadratic penalty, not a KL divergence


class DTWGPWarp:
    """Coarse DTW followed by inducing-point GP smoothing and prior scoring."""

    def __init__(self, x_basis_warp, noise_warp=0.05,
                 bound_noise_warp=(1e-10, 1e10), recursive=False, cuda=False,
                 bayesian=True, mode='balanced', *, points=24,
                 inducing=10, lengthscale=0.2, amplitude=0.005, anchor_noise=0.0005):
        self.device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')
        self.points = int(points)
        self.inducing = int(inducing)
        self.lengthscale = float(lengthscale)
        self.amplitude = float(amplitude)
        self.anchor_noise = float(anchor_noise)
        self.warp_gp = _ZeroWarpScore()  # native model conversion hook
        self._smoother_grid = None
        self._smoother_basis = None
        self._smoother_projector = None

    def _smoother(self, grid):
        if self._smoother_grid is not None and np.array_equal(grid, self._smoother_grid):
            return self._smoother_basis, self._smoother_projector
        normalized = (grid - grid[0]) / (grid[-1] - grid[0])
        inducing = np.linspace(0.0, 1.0, min(self.inducing, len(grid)))
        kuu = self.amplitude ** 2 * np.exp(
            -0.5 * ((inducing[:, None] - inducing[None, :]) / self.lengthscale) ** 2)
        kuu += 1e-8 * np.eye(len(inducing))
        ktu = self.amplitude ** 2 * np.exp(
            -0.5 * ((normalized[:, None] - inducing[None, :]) / self.lengthscale) ** 2)
        chol = np.linalg.cholesky(kuu)
        basis = np.linalg.solve(chol, ktu.T).T  # K_TU L_UU^-T
        precision = np.eye(len(inducing)) + basis.T @ basis / self.anchor_noise ** 2
        projector = np.linalg.solve(precision, basis.T / self.anchor_noise ** 2)
        self._smoother_grid = grid.copy()
        self._smoother_basis = basis
        self._smoother_projector = projector
        return basis, projector

    def infer(self, x, y, representers):
        x = torch.as_tensor(x, dtype=torch.float64, device=self.device).reshape(-1)
        y = torch.as_tensor(y, dtype=torch.float64, device=self.device)
        if y.ndim == 1:
            y = y[:, None]
        if y.shape != (len(x), len(representers)):
            raise ValueError('One GP representer is required for every output')
        if len(x) < 2 or not bool(torch.all(x[1:] > x[:-1])):
            raise ValueError('DTW-GP alignment needs a strictly increasing grid')
        grid = x.detach().cpu().numpy()
        observed = y.detach().cpu().numpy()
        with torch.no_grad():
            template = torch.stack([gp.mean_at(x[:, None]).reshape(-1)
                                    for gp in representers], dim=-1).cpu().numpy()
        displacement = _dtw_displacement(observed, template, grid, self.points)
        basis, projector = self._smoother(grid)
        span = grid[-1] - grid[0]
        whitened = projector @ (displacement / span)
        smoothed = basis @ whitened * span
        mapped = _monotone_fixed_ends(grid, grid + smoothed)
        offset = torch.as_tensor(mapped - grid, dtype=x.dtype,
                                 device=self.device)[:, None]
        warped = torch.as_tensor(mapped, dtype=x.dtype,
                                  device=self.device)[None, :, None]
        # Score the map actually returned after monotonic projection. For
        # K = B B.T + anchor_noise**2 I, this is 0.5 v.T K^-1 v (the
        # log-determinant is fixed for a common grid and cancels between
        # candidate clusters). The residual term matters when projection
        # moves the map outside the inducing GP subspace.
        final_displacement = (mapped - grid) / span
        final_whitened = projector @ final_displacement
        residual = final_displacement - basis @ final_whitened
        gp_penalty = 0.5 * (final_whitened @ final_whitened
                            + residual @ residual / self.anchor_noise ** 2)
        penalty = torch.as_tensor(gp_penalty, dtype=x.dtype,
                                  device=self.device)
        return PointWarpResult(warped, offset, torch.zeros_like(offset), penalty)

    def update_warp(self, x_train, x_warp):
        pass

    def compute_warp(self, x_model, y_target, y_model, theta=None, noise=None,
                     visualize=False, verbose=False, train_iter=None):
        """Retain the existing low-level warper shape for direct callers."""
        x = torch.as_tensor(x_model, dtype=torch.float64, device=self.device).reshape(-1, 1)
        target = torch.as_tensor(y_target, dtype=torch.float64, device=self.device)
        vector = target.ndim == 1
        if vector:
            target = target[:, None]
        template = torch.as_tensor(y_model, dtype=torch.float64, device=self.device)
        if template.ndim == 1:
            template = template[:, None]
        if target.shape != template.shape or target.shape[0] != len(x):
            raise ValueError('Target and template must have shape (points, outputs)')

        class Template:
            def __init__(self, values):
                self.values = values

            def mean_at(self, grid):
                return self.values

        result = self.infer(x, target, [Template(template[:, [ld]])
                                        for ld in range(target.shape[1])])
        query = result.grid_samples[0, :, 0].detach().cpu().numpy()
        base = x[:, 0].detach().cpu().numpy()
        observed = target.detach().cpu().numpy()
        aligned = torch.as_tensor(np.stack([
            np.interp(query, base, observed[:, ld])
            for ld in range(target.shape[1])], axis=-1),
            dtype=target.dtype, device=target.device)
        trace = {name: [] for name in ('loss', 'data', 'smooth', 'amp')}
        return (result.mean_displacement, aligned[:, 0] if vector else aligned,
                -result.kl, trace)

    def compute_warp_batch(self, x_model, y_target_batch, y_model, theta=None,
                           noise=None, weights=None, visualize=False,
                           verbose=False, train_iter=None):
        batch = torch.as_tensor(y_target_batch, dtype=torch.float64, device=self.device)
        if batch.ndim == 1:
            batch = batch[None, :, None]
        elif batch.ndim == 2:
            batch = batch[:, :, None]
        template = torch.as_tensor(y_model, dtype=torch.float64, device=self.device)
        if template.ndim == 3 and template.shape[0] != len(batch):
            raise ValueError('Batch templates must match the observation count')
        offsets, aligned, scores = [], [], []
        for i, sample in enumerate(batch):
            reference = template[i] if template.ndim == 3 else template
            offset, waveform, score, _ = self.compute_warp(
                x_model, sample, reference, theta, noise, visualize,
                verbose, train_iter)
            offsets.append(offset)
            aligned.append(waveform)
            scores.append(score)
        trace = {name: [] for name in ('loss', 'data', 'smooth', 'amp')}
        return (torch.stack(offsets), torch.stack(aligned), torch.stack(scores), trace)

    def reset(self):
        self._smoother_grid = None
        self._smoother_basis = None
        self._smoother_projector = None
