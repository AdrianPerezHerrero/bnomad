"""Experimental scalar row-ratio integration for the existing ordered HDP ELBO.

If X = logit(V) - log(W), the exact characteristic function of X is a
product of two beta-function ratios. Invert it and its shape derivatives
with a real FFT. All objectives use the same independent Beta stick family.
"""
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
import math

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import betaln, digamma, expit, gammaln, loggamma, polygamma
from . import _topology


def _support(a, b, c, d, tail):
    """Bound the integration interval using the exact moment generating function."""
    mean = digamma(a) - digamma(b) - digamma(c) + digamma(c+d)
    variance = polygamma(1, a) + polygamma(1, b) + (polygamma(1, c) - polygamma(1, c+d))
    if not np.isfinite(variance) or variance <= 0:
        raise FloatingPointError('Invalid row-ratio variance')
    sd = math.sqrt(variance)
    radii = []
    for sign, rate in ((-1, a), (1, min(b, c))):
        magnitude = np.unique(np.minimum(rate * np.array([.125, .25, .5, .7, .85, .95]), 12/sd))
        tilt = sign*magnitude
        moment = (betaln(a+tilt, b-tilt) - betaln(a, b) + betaln(c-tilt, d)
                  - betaln(c, d) - tilt*mean)
        radii.append(np.min((moment + tail)/magnitude))
    return mean-radii[0], mean+radii[1], sd, mean


def _centered_log_gamma(shape, shift):
    """Return log Γ(shape+shift) − log Γ(shape) − shift·ψ(shape)."""
    z = shape+shift
    if shape < 32:
        return loggamma(z)-gammaln(shape)-shift*digamma(shape)
    # Difference the Stirling expansion before evaluation. Subtracting two
    # large log-gamma values loses digits when a stick is concentrated.
    scaled = shift.imag/shape
    relative_log = .5*np.log1p(scaled**2)+1j*np.arctan(scaled)
    residual = (shape-.5)*relative_log-shift

    def correction(value):
        inverse = 1/value
        square = inverse**2
        return inverse*(1/12-square*(1/360-square*(1/1260-square/1680)))

    return (residual + shift*(np.log(shape)-digamma(shape)+relative_log)
            + correction(z)-correction(shape))


@lru_cache(maxsize=2048)
def _row_value_gradient(a, b, c, d, alpha, kappa, log_stay, log_advance, *, resolution=1, tail=42.):
    """Integrate a row's stick-dependent ELBO and log-shape gradient together.

    Tail and grid controls are runtime safeguards for the experimental
    inversion. Agreement across grids is checked outside the optimizer.
    """
    lo, hi, sd, mean = _support(a, b, c, d, tail)
    width = hi-lo
    # Resolve the density and the sigmoid transition in the row integrand.
    target = resolution*width/min(.25, sd/4)
    if not np.isfinite(target) or target <= 0 or target > 2**18:
        raise FloatingPointError('Row-ratio integration grid exceeds the experimental limit')
    size = max(128, 2**math.ceil(math.log2(target)))
    step = width/size
    frequency = 2j*np.pi*np.fft.rfftfreq(size, d=step)
    ap, bm, cm, cmd = a+frequency, b-frequency, c-frequency, c+d-frequency
    phase = (_centered_log_gamma(a, frequency)+_centered_log_gamma(b, -frequency)
             + _centered_log_gamma(c, -frequency)-_centered_log_gamma(c+d, -frequency)
             + frequency*(mean-lo))
    characteristic = np.exp(phase)
    characteristic[0] = 1.
    slope = np.array([
        a*(digamma(ap)-digamma(a)),
        b*(digamma(bm)-digamma(b)),
        c*(digamma(cm)-digamma(cmd)-digamma(c)+digamma(c+d)),
        d*(-digamma(cmd)+digamma(c+d)),
    ])
    spectrum = np.vstack((characteristic, characteristic*slope))
    spectrum[1:, 0] = 0.
    if not np.isfinite(spectrum).all():
        raise FloatingPointError('Nonfinite row-ratio spectrum')
    # The inversion gives bin weights with total mass one, including aliases.
    weights = np.fft.irfft(spectrum.conj(), n=size, axis=-1)
    if np.max(np.abs(spectrum[:, -1])) > 1e-10 or weights[0].min() < -1e-10:
        if resolution >= 16:
            raise FloatingPointError('Row-ratio spectrum did not converge on the refinement grids')
        return _row_value_gradient(a, b, c, d, alpha, kappa, log_stay, log_advance,
                                   resolution=2*resolution, tail=tail)
    x = lo+step*np.arange(size)
    stay, advance = expit(x), expit(-x)
    shape_stay, shape_advance = alpha*stay+kappa, alpha*advance
    log_shape_stay = np.log(shape_stay) if kappa else np.log(alpha)-np.logaddexp(0., -x)
    log_shape_advance = np.log(alpha)-np.logaddexp(0., x)
    integrand = (gammaln(alpha+kappa) - gammaln(1+shape_stay) - gammaln(1+shape_advance)
                 + log_shape_stay + log_shape_advance + (shape_stay-1)*log_stay
                 + (shape_advance-1)*log_advance)
    center = integrand[np.argmax(weights[0])]
    result = weights @ (integrand-center)
    result[0] += center
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite row-ratio integral')
    return result


def _value_gradient(parameters, gamma, alpha, kappa, log_pi, entropy, *, resolution=1, tail=42.):
    count = len(log_pi)
    a, b = np.exp(parameters).reshape(2, count)
    gradient = np.zeros((2, count))
    rows = np.empty(count)
    for i, (log_stay, log_advance) in enumerate(log_pi):
        c, d = (a[i+1], b[i+1]) if i+1 < count else (1., gamma)
        value, da, db, dc, dd = _row_value_gradient(a[i], b[i], c, d, alpha, kappa,
            log_stay, log_advance, resolution=resolution, tail=tail)
        rows[i] = value
        gradient[:, i] += (da, db)
        if i+1 < count:
            gradient[:, i+1] += (dc, dd)
    shared = (a+b-1-gamma)*polygamma(1, a+b)
    gradient[0] -= a*((a-1)*polygamma(1, a)-shared)
    gradient[1] -= b*((b-gamma)*polygamma(1, b)-shared)
    value = (rows+entropy).sum() - _topology._beta_kl(a, b, gamma)
    return float(value), gradient.ravel()


def _context(state, theta):
    a, b = _topology.shapes(_topology._array(state.rho), _topology._array(state.omega))
    index = np.arange(len(a))
    matrix = _topology._array(theta)
    posterior = np.stack((matrix[index, index], matrix[index, index+1]), axis=1)
    log_pi, entropy = _topology._row_posterior_terms(posterior)
    return np.log(np.r_[a, b]), (state.gamma, state.transAlpha, state.kappa, log_pi, entropy)


def optimize_sticks(state, theta):
    """Run the experimental optimizer with its own converged-grid audit."""
    evaluations = 0
    rejected_trials = 0
    scale = 1.

    def evaluate(parameters, **grid):
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            value, gradient = _value_gradient(parameters, *args, **grid)
            if not np.isfinite(value) or not np.isfinite(gradient).all():
                raise FloatingPointError('Nonfinite stick objective or gradient')
            return value, gradient

    def objective(parameters):
        nonlocal evaluations, rejected_trials
        evaluations += 1
        try:
            value, gradient = evaluate(parameters)
        except FloatingPointError:
            rejected_trials += 1
            return 1e100, np.zeros_like(parameters)
        return -value/scale, -gradient/scale

    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            initial, args = _context(state, theta)
        start, initial_gradient = objective(initial)
        if start == 1e100:
            raise FloatingPointError('Initial row-ratio integration failed')
        # A fixed scalar leaves the maximizer intact and reduces initial
        # line-search excursions into poorly conditioned Beta shapes.
        scale = max(1., float(np.max(np.abs(initial_gradient))))
        result = minimize(objective, initial, jac=True, method='L-BFGS-B',
            bounds=[(math.log(1e-3), math.log(1e8))]*len(initial),
            options={'maxiter': 40, 'ftol': 1e-10, 'gtol': 1e-5/scale, 'maxls': 30})
        improved = result.fun < start/scale
        if not improved and np.max(np.abs(initial_gradient)) > 1e-4:
            raise FloatingPointError('Optimizer stopped without improving a nonstationary point')
        chosen = result.x if improved else initial
        fine, fine_gradient = evaluate(chosen, resolution=2, tail=52.)
        finer, finer_gradient = evaluate(chosen, resolution=3, tail=58.)
        start_fine, _ = evaluate(initial, resolution=3, tail=58.)
        error = abs(finer-fine)
        gradient_error = np.max(np.abs(finer_gradient-fine_gradient))
        coarse, coarse_gradient = evaluate(chosen)
        tolerance = 1e-7+1e-10*abs(finer)
        objective_error = max(error, abs(coarse-finer))
        total_gradient_error = max(gradient_error, np.max(np.abs(coarse_gradient-finer_gradient)))
        if (objective_error > tolerance or total_gradient_error > 1e-6
                or finer < start_fine-tolerance):
            raise FloatingPointError(f'Row-ratio audit errors were {objective_error:.3g} in value and '
                                     f'{total_gradient_error:.3g} in gradient')
        a, b = np.exp(chosen).reshape(2, -1)
        rho, omega = (torch.as_tensor(value, dtype=torch.float64, device=state.device)
                      for value in (a/(a+b), a+b))
        return rho, omega, dict(solver='row_ratio', success=bool(result.success), objective=finer,
            iterations=int(result.nit),
            evaluations=evaluations, quadrature_accepted=True, used_fallback=False, improved=improved,
            rejected_trials=rejected_trials,
            integration_error=float(error), gradient_integration_error=float(gradient_error),
            objective_grid_error=float(abs(coarse-finer)))
    except FloatingPointError as error:
        rho, omega, info = _topology.optimize_sticks(state, theta)
        return rho, omega, dict(info, solver='quadrature', evaluations=evaluations, rejected_trials=rejected_trials,
                                used_fallback=True, fallback_reason=str(error))


@dataclass
class FitDiagnostics:
    """Count numerical updates across all staged branches of one fit."""
    updates: int = 0
    optimizer_stops: int = 0
    rejected_audits: int = 0
    fallbacks: Counter = field(default_factory=Counter)

    def record(self, info):
        self.updates += 1
        self.optimizer_stops += not info['success']
        self.rejected_audits += not info['quadrature_accepted']
        if info['used_fallback']:
            self.fallbacks[info['fallback_reason']] += 1

    def summary(self):
        fallback_updates = sum(self.fallbacks.values())
        return dict(requested_solver='row_ratio', updates=self.updates,
                    row_ratio_updates=self.updates-fallback_updates, fallback_updates=fallback_updates,
                    fallback_reasons=dict(self.fallbacks), nonconverged_updates=self.optimizer_stops,
                    rejected_audits=self.rejected_audits)
