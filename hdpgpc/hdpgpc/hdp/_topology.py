"""Ordered HDP transition rows on reduced simplices, not masked full rows.

v_i ~ Beta(1,gamma), beta_i=v_i prod_{l<i}(1-v_l).
r_i=beta_i/(beta_i+beta_{i+1})=v_i/(v_i+(1-v_i)v_{i+1}).
(stay_i,advance_i)|v ~ Dirichlet(alpha*r_i+kappa,alpha*(1-r_i)).
The initial state is zero. An unoccupied successor is retained at truncation.
Expected prior normalizers require deterministic Beta quadrature; the original
unrestricted HDP surrogate/optimizer is not valid for these normalized rows.
"""
from functools import lru_cache

import numpy as np
import torch
from scipy.linalg import eigh_tridiagonal
from scipy.integrate import quad_vec
from scipy.optimize import minimize
from scipy.special import betaln, digamma, gammaln, hyp2f1


def enabled(state):
    return getattr(state, 'transition_topology', 'ergodic') == 'left_to_right'


@lru_cache(maxsize=2048)
def _beta_rule(a, b, order):
    """Normalized Gauss-Jacobi rule without overflowing its normalization."""
    n = np.arange(order, dtype=float)
    total = a+b
    diagonal = np.empty(order)
    diagonal[0] = (a-b)/total
    diagonal[1:] = ((a-b)*(total-2)
        / ((2*n[1:]+total-2)*(2*n[1:]+total)))
    off = np.empty(order-1)
    off[0] = np.sqrt(4*a*b/(total*total*(total+1)))
    k = n[2:]
    z = 2*k+total-2
    off[1:] = np.sqrt(4*k*(k+a-1)*(k+b-1)*(k+total-2)
                              / (z*z*(z*z-1)))
    nodes, vectors = eigh_tridiagonal(diagonal, off)
    # Roundoff can put a node exactly at an endpoint for very concentrated q.
    nodes = np.clip((nodes+1)/2, np.finfo(float).eps, 1-np.finfo(float).eps)
    weights = vectors[0]**2
    return nodes, weights/weights.sum()


def _hypergeometric(A, B, C, nodes):
    values = hyp2f1(A, B, C, nodes)
    bad = ~np.isfinite(values)
    if bad.any() and A > 0 and B > 0:
        # scipy's continuation can return NaN for large C although the
        # positive power series converges rapidly (e.g. A=1,B=2,C=1002).
        x = nodes[bad]
        term, total = np.ones_like(x), np.ones_like(x)
        for n in range(1, 10001):
            with np.errstate(over='ignore', invalid='ignore'):
                term = term*((A+n-1)*(B+n-1)/((C+n-1)*n))*x
                total = total+term
            if np.all(np.abs(term) <= 1e-14*np.abs(total)):
                values[bad] = total
                break
            if not np.isfinite(total).all():
                break
    return values


@lru_cache(maxsize=8192)
def row_moments(a, b, c, d, alpha, kappa, order=128):
    """Return E[r] and E[log normalization] of one conditional row prior.

    Singular log(v) terms are integrated analytically. Only smooth remaining
    terms use quadrature. `order` is exposed privately for numerical audits.
    """
    # Integrating nonlinear functions of v/(v+(1-v)w) on a Cartesian
    # quadrature grid resolves its corner poorly at large concentrations.
    # Its 1-D density is Beta(a,min(b,c)) times a hypergeometric factor.
    r, wr = _beta_rule(float(a), float(min(b, c)), order)
    hyper = (_hypergeometric(c+d-b, a+c, a+c+d, r) if b >= c
             else _hypergeometric(a+b, d, a+c+d, r))
    with np.errstate(over='ignore', invalid='ignore'):
        weighted = wr*hyper
    if min(a, b, c, d) >= .5 and max(a, b, c, d) <= 100 and abs(b-c) < 1.5:
        # Near b=c the ratio density has a logarithmic endpoint. Adaptive
        # integration resolves it accurately without thousands of Jacobi nodes.
        power = min(b, c)
        constant = betaln(a+c, d)-betaln(a, b)-betaln(c, d)
        def integrand(value):
            value = np.clip(value, np.finfo(float).eps, 1-np.finfo(float).eps)
            h = (hyp2f1(c+d-b, a+c, a+c+d, value) if b >= c
                 else hyp2f1(a+b, d, a+c+d, value))
            density = np.exp(constant+(a-1)*np.log(value)
                             +(power-1)*np.log1p(-value)+np.log(h))
            return density*np.array([1., value, np.log(value),
                gammaln(1+alpha*value+kappa)+gammaln(1+alpha*(1-value)),
                np.log(alpha*value+kappa) if kappa else 0.])
        moments, error, info = quad_vec(integrand, 0., 1., epsabs=1e-9, epsrel=1e-10,
                                        limit=512, full_output=True)
        if not info.success or not np.isfinite(moments).all():
            raise FloatingPointError('Ordered transition-prior integration did not converge')
        mass, mean_r, log_r, ex_gamma, ex_log_x = moments/moments[0]
        mean_advance = 1-mean_r
        log_denominator = digamma(a)-digamma(a+b)-log_r
    elif np.isfinite(weighted).all() and (weighted >= 0).all() and weighted.sum() > 0:
        mass = weighted.sum()
        weighted = weighted/mass
        # Integrate the log(r) endpoint singularity analytically using h(0).
        # Since log(r) vanishes at r=1, no divergent h(1) is needed when b=c.
        log_r = (digamma(a)-digamma(a+min(b, c))
                 +float(wr @ ((hyper-1)*np.log(r))))/mass
        log_denominator = digamma(a)-digamma(a+b)-log_r
        mean_r, mean_advance = float(weighted @ r), float(weighted @ (1-r))
        ex_gamma = float(weighted @ (gammaln(1+alpha*r+kappa)+gammaln(1+alpha*(1-r))))
        ex_log_x = float(weighted @ np.log(alpha*r+kappa)) if kappa else None
    else:
        # Very asymmetric posteriors can overflow hyp2f1. Their ratio is
        # concentrated, where the direct two-dimensional rule is well behaved.
        v, wv = _beta_rule(float(a), float(b), order)
        w, ww = _beta_rule(float(c), float(d), order)
        denominator = v[:, None]+(1-v[:, None])*w[None]
        ratio = v[:, None]/denominator
        advance = (1-v[:, None])*w[None]/denominator
        weight = wv[:, None]*ww[None]
        average = lambda value: float(np.sum(weight*value))
        log_denominator = average(np.log(denominator))
        mean_r, mean_advance = average(ratio), average(advance)
        x, y = alpha*ratio+kappa, alpha*advance
        ex_gamma = average(gammaln(1+x)+gammaln(1+y))
        ex_log_x = average(np.log(x)) if kappa else None
    log_y = (np.log(alpha)+digamma(b)-digamma(a+b)
             +digamma(c)-digamma(c+d)-log_denominator)
    log_x = (ex_log_x if kappa else
             np.log(alpha)+digamma(a)-digamma(a+b)-log_denominator)
    normalizer = (gammaln(alpha+kappa)-ex_gamma
                  +log_x+log_y)
    return mean_r, normalizer, mean_advance


def shapes(rho, omega):
    rho, omega = np.asarray(rho, dtype=float), np.asarray(omega, dtype=float)
    return rho*omega, (1-rho)*omega


def prior_rows(state, rho=None, omega=None, *, order=128):
    rho = state.rho if rho is None else rho
    omega = state.omega if omega is None else omega
    a, b = shapes(_array(rho), _array(omega))
    rows, constants = [], []
    for i in range(len(a)):
        # The next stick beyond represented support keeps its prior q=p.
        c, d = (a[i+1], b[i+1]) if i+1 < len(a) else (1., float(state.gamma))
        ratio, constant, advance = row_moments(float(a[i]), float(b[i]), float(c), float(d),
                                     float(state.transAlpha), float(state.kappa), order)
        rows.append((state.transAlpha*ratio+state.kappa, state.transAlpha*advance))
        constants.append(constant)
    return np.asarray(rows), np.asarray(constants)


def _array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def allowed_counts(counts, count):
    counts = _array(counts)[:count, :count]
    mask = np.eye(count, dtype=bool) | np.eye(count, k=1, dtype=bool)
    if not np.isfinite(counts).all() or (counts < -1e-12).any():
        raise ValueError('Invalid ordered transition counts')
    if np.any(np.abs(counts[~mask]) > 1e-10):
        raise ValueError('Left-to-right counts contain forbidden transitions')
    result = np.zeros((count, 2))
    result[:, 0] = np.diag(counts)
    if count > 1:
        result[:-1, 1] = np.diag(counts, k=1)
    return result


def posterior_matrices(state, counts, rho=None, omega=None):
    rows, _ = prior_rows(state, rho, omega)
    size = len(rows)
    rows = rows+allowed_counts(counts, size)
    theta = torch.zeros((size+1, size+1), dtype=torch.float64, device=state.device)
    index = torch.arange(size, device=state.device)
    theta[index, index] = torch.as_tensor(rows[:, 0], device=state.device)
    theta[index, index+1] = torch.as_tensor(rows[:, 1], device=state.device)
    # A reserve bookkeeping row; no learned Dirichlet has zero concentrations.
    theta[-1, -1] = 1.
    start = theta.new_zeros(size+1)
    start[0] = 1.
    return theta, start


def expected_logs(theta, count):
    theta = torch.as_tensor(theta, dtype=torch.float64)
    result = theta.new_full((count, count), -torch.inf)
    index = torch.arange(count, device=theta.device)
    rows = torch.stack((theta[index, index], theta[index, index+1]), 1)
    if not bool((rows > 0).all()):
        raise ValueError('Ordered row posteriors require positive stay/advance parameters')
    logs = torch.digamma(rows)-torch.digamma(rows.sum(1))[:, None]
    result[index, index] = logs[:, 0]
    if count > 1:
        result[index[:-1], index[1:]] = logs[:-1, 1]
    return result


def start_logs(theta, count):
    result = torch.as_tensor(theta).new_full((count,), -torch.inf)
    result[0] = 0.
    return result


def _beta_kl(a, b, gamma):
    return np.sum(betaln(1., gamma)-betaln(a, b)+(a-1)*digamma(a)
        +(b-gamma)*digamma(b)-(a+b-1-gamma)*digamma(a+b))


def linear_terms(state, counts, rho=None, omega=None, theta=None, *, order=128):
    """E log p(v,pi,s)-E log q(v,pi), excluding chain entropy."""
    rho = state.rho if rho is None else rho
    omega = state.omega if omega is None else omega
    rows, constants = prior_rows(state, rho, omega, order=order)
    observed = allowed_counts(counts, len(rows))
    if theta is None:
        posterior = rows+observed
    else:
        theta = _array(theta)
        i = np.arange(len(rows))
        posterior = np.stack((theta[i, i], theta[i, i+1]), 1)
    log_pi = digamma(posterior)-digamma(posterior.sum(1))[:, None]
    entropy = (betaln(posterior[:, 0], posterior[:, 1])
        +(posterior.sum(1)-2)*digamma(posterior.sum(1))
        -((posterior-1)*digamma(posterior)).sum(1))
    a, b = shapes(_array(rho), _array(omega))
    return float(np.sum(constants+((rows+observed-1)*log_pi).sum(1)+entropy)
                 -_beta_kl(a, b, state.gamma))


def optimize_sticks(state, theta):
    """Numerical q(v) coordinate step for the ordered prior's own objective."""
    a, b = shapes(_array(state.rho), _array(state.omega))
    initial = np.log(np.concatenate((a, b)))
    counts = np.zeros((len(a), len(a)))
    def objective(parameters):
        left, right = np.exp(parameters).reshape(2, -1)
        omega = left+right
        try:
            return -linear_terms(state, counts, left/omega, omega, theta)
        except FloatingPointError:
            return np.inf
    result = minimize(objective, initial, method='L-BFGS-B',
        bounds=[(np.log(1e-3), np.log(1e8))]*len(initial),
        options={'maxiter': 40, 'ftol': 1e-10, 'maxls': 20})
    # An incomplete optimizer is useful only if its actual objective improved.
    chosen = result.x if np.isfinite(result.fun) and result.fun < objective(initial) else initial
    left, right = np.exp(chosen).reshape(2, -1)
    fine = linear_terms(state, counts, left/(left+right), left+right, theta, order=256)
    coarse = -objective(chosen)
    integration_error = abs(fine-coarse)
    reliable = np.isfinite(fine) and integration_error <= 1e-7+1e-10*max(1., abs(fine))
    if not reliable:
        # A quadrature artifact is never an accepted variational update.
        chosen = initial
        left, right = np.exp(chosen).reshape(2, -1)
    omega = torch.as_tensor(left+right, dtype=torch.float64, device=state.device)
    rho = torch.as_tensor(left/(left+right), dtype=torch.float64, device=state.device)
    return rho, omega, dict(success=bool(result.success), objective=float(objective(chosen)),
                           iterations=int(result.nit), quadrature_order=128,
                           quadrature_error_estimate=float(integration_error),
                           quadrature_accepted=bool(reliable))

