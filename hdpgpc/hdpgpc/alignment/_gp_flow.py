"""Variational monotonic GP flow for HDP-GPC observations.

This is the uncertain-drift ODE construction of Mikheeva et al. (2022),
adapted to the existing per-cluster GP emission model.  A path-wise GP draw is
held fixed at every ODE evaluation for one Monte Carlo warp sample.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, Product, RBF, Sum, WhiteKernel


def _column(value, *, device=None):
    result = torch.as_tensor(value, dtype=torch.float64, device=device)
    return result.reshape(-1, 1)


def _kernel_torch(kernel, x, y=None):
    """Differentiable counterpart of the sklearn kernels used by GP emissions."""
    if isinstance(kernel, Sum):
        return _kernel_torch(kernel.k1, x, y) + _kernel_torch(kernel.k2, x, y)
    if isinstance(kernel, Product):
        return _kernel_torch(kernel.k1, x, y) * _kernel_torch(kernel.k2, x, y)
    if isinstance(kernel, ConstantKernel):
        n = x.shape[-2]
        m = n if y is None else y.shape[-2]
        return x.new_full((*x.shape[:-2], n, m), float(kernel.constant_value))
    if isinstance(kernel, WhiteKernel):
        n = x.shape[-2]
        if y is None:
            return float(kernel.noise_level) * torch.eye(n, dtype=x.dtype, device=x.device).expand(
                *x.shape[:-2], n, n)
        return x.new_zeros((*torch.broadcast_shapes(x.shape[:-2], y.shape[:-2]), n, y.shape[-2]))
    if isinstance(kernel, (RBF, Matern)):
        rhs = x if y is None else y
        scale = torch.as_tensor(kernel.length_scale, dtype=x.dtype, device=x.device)
        delta = (x.unsqueeze(-2) - rhs.unsqueeze(-3)) / scale
        radius2 = torch.sum(delta.square(), dim=-1)
        if isinstance(kernel, RBF):
            return torch.exp(-0.5 * radius2)
        radius = torch.sqrt(radius2.clamp_min(1e-30))
        if kernel.nu == 0.5:
            return torch.exp(-radius)
        if kernel.nu == 1.5:
            z = math.sqrt(3.0) * radius
            return (1.0 + z) * torch.exp(-z)
        if kernel.nu == 2.5:
            z = math.sqrt(5.0) * radius
            return (1.0 + z + z.square() / 3.0) * torch.exp(-z)
        raise ValueError('GP flow supports Matern nu=0.5, 1.5 or 2.5')
    raise TypeError(f'GP flow has no differentiable adapter for {type(kernel).__name__}')


class GPRepresenter:
    """Differentiable predictive density from a current cluster GP/LDS state."""

    def __init__(self, gp, device):
        self.device = device
        self.kernel = gp.gp.kernel
        self.x_basis = _column(gp.x_basis, device=device).detach()
        self.mean_basis = torch.as_tensor(
            gp.C[-1] @ gp.f_star_sm[-1], dtype=torch.float64, device=device).reshape(-1, 1).detach()
        self.sigma = torch.as_tensor(gp.Sigma[-1], dtype=torch.float64, device=device).detach()
        self.eye = torch.eye(len(self.x_basis), dtype=torch.float64, device=device)
        # The native GP projection evaluates K(X, X) with two explicit
        # arguments.  In sklearn that omits WhiteKernel's diagonal term.
        self.kbb = _kernel_torch(self.kernel, self.x_basis, self.x_basis)
        jitter = 1e-4 * torch.diagonal(self.sigma).abs().mean().clamp_min(
            torch.finfo(torch.float64).eps)
        self.chol = torch.linalg.cholesky(0.5 * (self.kbb + self.kbb.T) + jitter * self.eye)
        self.alpha = torch.cholesky_solve(self.mean_basis, self.chol)
        self.constant_noise = bool(torch.all(torch.isclose(
            torch.diagonal(self.sigma), torch.diagonal(self.sigma).mean())))

    def mean_at(self, grid):
        """Current cluster's GP/LDS predictive mean at an input grid."""
        query = _column(grid, device=self.device)
        return _kernel_torch(self.kernel, self.x_basis, query).T @ self.alpha

    def log_prob(self, grids, values):
        """Return one properly normalized Gaussian log density per sampled grid."""
        grids = torch.as_tensor(grids, dtype=torch.float64, device=self.device)
        if grids.ndim == 2:
            grids = grids.unsqueeze(-1)
        values = _column(values, device=self.device)
        if grids.shape[-2:] != values.shape:
            raise ValueError('Warped grids and observations must share their point count')
        kbg = _kernel_torch(self.kernel, self.x_basis, grids)
        mean = torch.matmul(kbg.transpose(-1, -2), self.alpha)
        n = values.shape[0]
        residual = values.unsqueeze(0) - mean
        if self.constant_noise:
            variance = torch.diagonal(self.sigma).mean() + 1e-6
            return -0.5 * (residual.square().sum(dim=(-2, -1)) / variance
                           + n * torch.log(2.0 * math.pi * variance))
        solve = torch.cholesky_solve(kbg, self.chol)
        eye = torch.eye(n, dtype=grids.dtype, device=grids.device)
        kgg = _kernel_torch(self.kernel, grids)
        cov = kgg - torch.matmul(kbg.transpose(-1, -2), solve)
        cov = cov + torch.matmul(solve.transpose(-1, -2),
                                 torch.matmul(self.sigma, solve))
        cov = 0.5 * (cov + cov.transpose(-1, -2)) + 1e-6 * eye
        chol = torch.linalg.cholesky(cov)
        alpha = torch.cholesky_solve(residual, chol)
        quad = (residual * alpha).sum(dim=(-2, -1))
        logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)
        return -0.5 * (quad + logdet + n * math.log(2.0 * math.pi))


@dataclass
class WarpPosteriorResult:
    """One observation's warp posterior, evaluated on its input grid."""

    grid_samples: torch.Tensor       # (S,T,1)
    mean_displacement: torch.Tensor  # (T,1)
    std_displacement: torch.Tensor   # (T,1)
    kl: torch.Tensor
    expected_log_likelihood: torch.Tensor
    elbo: torch.Tensor
    mean_whitened: torch.Tensor
    chol_whitened: torch.Tensor
    trace: list[float]


class VariationalFlowWarp:
    """Per-observation GP drift posterior with a coherent path-wise ODE sampler."""

    def __init__(self, x_basis_warp, noise_warp=0.05,
                 bound_noise_warp=(1e-10, 1e10), recursive=False, cuda=False,
                 bayesian=True, mode='balanced', n_inducing=10,
                 n_features=64, n_samples=4, ode_steps=10, train_iter=30,
                 lengthscale=0.2, amplitude=0.1, lr=0.03):
        self.device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float64
        self.x_basis = _column(x_basis_warp, device=self.device)
        self.n_inducing = int(n_inducing)
        self.n_features = int(n_features)
        self.n_samples = int(n_samples)
        self.ode_steps = int(ode_steps)
        self.train_iter = int(train_iter)
        self.lengthscale = float(lengthscale)
        self.amplitude = float(amplitude)
        self.lr = float(lr)
        self.recursive = bool(recursive)
        if min(self.n_inducing, self.n_features, self.n_samples, self.ode_steps) < 1:
            raise ValueError('Flow sizes and ODE steps must be positive')
        if self.lengthscale <= 0 or self.amplitude <= 0 or self.lr <= 0:
            raise ValueError('Flow lengthscale, amplitude and learning rate must be positive')

    def _prior(self):
        u = torch.linspace(0.0, 1.0, self.n_inducing, dtype=self.dtype, device=self.device)
        delta = (u[:, None] - u[None, :]) / self.lengthscale
        kuu = self.amplitude ** 2 * torch.exp(-0.5 * delta.square())
        kuu = kuu + 1e-6 * torch.eye(self.n_inducing, dtype=self.dtype, device=self.device)
        return u, torch.linalg.cholesky(kuu)

    def _factor(self, raw):
        lower = torch.tril(raw, diagonal=-1)
        diag = F.softplus(torch.diagonal(raw)) + 1e-6
        return lower + torch.diag(diag)

    def _kl(self, mean, chol):
        m = self.n_inducing
        return 0.5 * (mean.square().sum() + chol.square().sum() - m
                      - 2.0 * torch.log(torch.diagonal(chol)).sum())

    def _draw(self, grid, mean, chol_q, n_samples):
        x = _column(grid, device=self.device)[:, 0]
        span = x[-1] - x[0]
        if not bool(span > 0) or not bool(torch.all(x[1:] > x[:-1])):
            raise ValueError('GP flow needs a strictly increasing input grid')
        xn = (x - x[0]) / span
        u, chol_prior = self._prior()
        s = int(n_samples)
        omega = torch.randn(s, self.n_features, dtype=self.dtype, device=self.device) / self.lengthscale
        phase = 2.0 * math.pi * torch.rand(s, self.n_features, dtype=self.dtype, device=self.device)
        alpha = torch.randn(s, self.n_features, dtype=self.dtype, device=self.device)
        epsilon = torch.randn(s, self.n_inducing, dtype=self.dtype, device=self.device)
        z = mean[None, :] + epsilon @ chol_q.T
        v = z @ chol_prior.T

        def features(locations):
            return math.sqrt(2.0 / self.n_features) * self.amplitude * torch.cos(
                locations[..., None] * omega[:, None, :] + phase[:, None, :])

        prior_u = (features(u[None, :].expand(s, -1)) * alpha[:, None, :]).sum(-1)
        correction = torch.cholesky_solve((v - prior_u).T, chol_prior).T

        def drift(locations):
            prior = (features(locations) * alpha[:, None, :]).sum(-1)
            kernel = self.amplitude ** 2 * torch.exp(
                -0.5 * ((locations[..., None] - u) / self.lengthscale).square())
            return prior + (kernel * correction[:, None, :]).sum(-1)

        steps = self.ode_steps
        for _ in range(6):
            locations = xn[None, :].expand(s, -1)
            for _ in range(steps):
                locations = locations + drift(locations) / steps
            if bool(torch.all(locations[:, 1:] > locations[:, :-1])):
                return (x[0] + span * locations).unsqueeze(-1)
            steps *= 2
        raise RuntimeError('ODE integration lost monotonicity after adaptive subdivision')

    def infer(self, x, y, representers, *, train_iter=None, n_samples=None,
              initial=None, weights=None):
        """Fit q(v) against the current cluster GP representers and this observation."""
        x = _column(x, device=self.device)
        y = torch.as_tensor(y, dtype=self.dtype, device=self.device)
        if y.ndim == 1:
            y = y[:, None]
        if y.shape != (len(x), len(representers)):
            raise ValueError('One GP representer is required for every observation output')
        if initial is None:
            mean0 = torch.zeros(self.n_inducing, dtype=self.dtype, device=self.device)
            raw0 = torch.diag(torch.full((self.n_inducing,),
                math.log(math.expm1(1.0)), dtype=self.dtype, device=self.device))
        else:
            mean0, factor0 = initial
            mean0 = torch.as_tensor(mean0, dtype=self.dtype, device=self.device).detach().clone()
            factor0 = torch.as_tensor(factor0, dtype=self.dtype, device=self.device).detach().clone()
            raw0 = torch.tril(factor0, -1) + torch.diag(torch.log(torch.expm1(
                torch.diagonal(factor0).clamp_min(1e-5))))
        mean = torch.nn.Parameter(mean0)
        raw = torch.nn.Parameter(raw0)
        optimizer = torch.optim.Adam([mean, raw], lr=self.lr)
        count = self.train_iter if train_iter is None else int(train_iter)
        draws = self.n_samples if n_samples is None else int(n_samples)
        if draws < 1 or count < 0:
            raise ValueError('n_samples must be positive and train_iter nonnegative')
        trace = []

        def expected_log_likelihood(grids):
            parts = torch.stack([gp.log_prob(grids, y[:, index])
                                 for index, gp in enumerate(representers)], dim=-1)
            if weights is not None:
                channel_weights = torch.as_tensor(weights, dtype=self.dtype, device=self.device)
                parts = parts * channel_weights
            return parts.sum(dim=-1).mean()

        for _ in range(count):
            optimizer.zero_grad(set_to_none=True)
            factor = self._factor(raw)
            grids = self._draw(x, mean, factor, draws)
            ell = expected_log_likelihood(grids)
            kl = self._kl(mean, factor)
            loss = kl - ell
            loss.backward()
            optimizer.step()
            trace.append(float(loss.detach().cpu()))

        with torch.no_grad():
            factor = self._factor(raw)
            # More draws for a stable posterior summary, without changing q.
            grids = self._draw(x, mean, factor, max(draws, 16))
            displacement = grids - x[None, :, :]
            kl = self._kl(mean, factor)
            ell = expected_log_likelihood(grids)
            result = WarpPosteriorResult(
                grid_samples=grids.detach(),
                mean_displacement=displacement.mean(0).detach(),
                std_displacement=displacement.std(0, unbiased=False).detach(),
                kl=kl.detach(), expected_log_likelihood=ell.detach(),
                elbo=(ell - kl).detach(), mean_whitened=mean.detach().clone(),
                chol_whitened=factor.detach().clone(), trace=trace)
        return result

    def compute_warp(self, x_model, y_target, y_model, theta=None, noise=None,
                     visualize=False, verbose=False, train_iter=None):
        """Legacy-shaped posterior summary; inference should consume ``infer``."""
        x = _column(x_model, device=self.device)
        target = torch.as_tensor(y_target, dtype=self.dtype, device=self.device)
        target_was_vector = target.ndim == 1
        if target_was_vector:
            target = target[:, None]
        template = torch.as_tensor(y_model, dtype=self.dtype, device=self.device)
        if template.ndim == 1:
            template = template[:, None]
        if target.shape != template.shape or target.shape[0] != len(x):
            raise ValueError('Target and template must have shape (points, outputs)')

        class Template:
            def __init__(self, values):
                self.values = values

            def log_prob(self, grids, values):
                # Only a compatibility objective; this is not a cluster GP.
                query = grids[..., 0]
                indices = torch.searchsorted(x[:, 0].contiguous(), query.contiguous()).clamp(1, len(x) - 1)
                lo = indices - 1
                t = ((query - x[lo, 0]) / (x[indices, 0] - x[lo, 0])).unsqueeze(-1)
                predicted = self.values[lo] * (1.0 - t) + self.values[indices] * t
                variance = torch.as_tensor(noise if noise is not None else 0.05,
                                           dtype=x.dtype, device=x.device).mean().clamp_min(1e-8)
                return -0.5 * (((values[None] - predicted).square() / variance)
                               + torch.log(2.0 * math.pi * variance)).sum((-2, -1))

        result = self.infer(x, target, [Template(template[:, [ld]])
                                        for ld in range(target.shape[1])], train_iter=train_iter)
        self._last_result = result
        trace = {'loss': result.trace, 'data': [], 'smooth': [], 'amp': []}
        query = (x + result.mean_displacement)[:, 0]
        indices = torch.searchsorted(x[:, 0].contiguous(), query.contiguous()).clamp(1, len(x) - 1)
        lo = indices - 1
        t = ((query - x[lo, 0]) / (x[indices, 0] - x[lo, 0]))[:, None]
        aligned = target[lo] * (1.0 - t) + target[indices] * t
        return result.mean_displacement, aligned[:, 0] if target_was_vector else aligned, result.elbo, trace

    def compute_warp_batch(self, x_model, y_target_batch, y_model, theta=None,
                           noise=None, weights=None, visualize=False, verbose=False,
                           train_iter=None):
        data = torch.as_tensor(y_target_batch, dtype=self.dtype, device=self.device)
        if data.ndim == 1:
            data = data[None, :, None]
        elif data.ndim == 2:
            data = data[..., None]
        template = torch.as_tensor(y_model, dtype=self.dtype, device=self.device)
        if template.ndim == 3 and template.shape[0] != len(data):
            raise ValueError('Batch templates must match the number of observations')
        displacements, observations, scores, traces = [], [], [], []
        for index, sample in enumerate(data):
            template_sample = template[index] if template.ndim == 3 else template
            w, yw, score, trace = self.compute_warp(x_model, sample, template_sample,
                theta, noise, visualize, verbose, train_iter)
            displacements.append(w)
            observations.append(yw)
            scores.append(score)
            traces.append(trace)
        lengths = {len(item['loss']) for item in traces}
        loss = ([float(torch.as_tensor([item['loss'][i] for item in traces]).mean())
                 for i in range(next(iter(lengths)))] if lengths else [])
        return (torch.stack(displacements), torch.stack(observations),
                torch.stack(scores), {'loss': loss, 'data': [], 'smooth': [], 'amp': []})

    def update_warp(self, x_train, x_warp):
        return None

    def reset(self):
        self._last_result = None
