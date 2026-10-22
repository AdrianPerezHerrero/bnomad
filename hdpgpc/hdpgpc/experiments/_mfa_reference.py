"""Independent maximum-likelihood finite MFA reference for validation only.

Exact EM for component centroids/loadings, mixture weights and one shared
diagonal noise covariance. This implementation deliberately does not call
FactorAnalysisEmission or the HDP inference code. Formulas: Ghahramani &
Hinton, CRG-TR-96-1, Appendix B (1996, revised 1997).
https://www.cs.utoronto.ca/~hinton/absps/tr-96-1.pdf
"""
from __future__ import annotations

import math
from numbers import Integral

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits


class FiniteMFAReference:
    """Experimental finite, point-estimated MFA with fixed K and latent rank.

    Choose K independently of test labels. Several initializations are selected
    by training likelihood only. Noise floor is a declared ML constraint, not a
    covariance ridge or prior. No d-by-d covariance is formed during inference.
    """
    def __init__(self, n_components=3, n_factors=2, *, noise_initial=.4,
                 noise_floor=1e-8, max_iter=200, tol=1e-6, n_init=3, seed=0):
        for name, value in (('n_components', n_components), ('n_factors', n_factors),
                            ('max_iter', max_iter), ('n_init', n_init)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not (math.isfinite(noise_floor) and noise_floor > 0 and math.isfinite(tol) and tol > 0):
            raise ValueError('noise_floor and tol must be finite and positive')
        self.n_components, self.n_factors = int(n_components), int(n_factors)
        self.noise_initial, self.noise_floor = noise_initial, float(noise_floor)
        self.max_iter, self.tol, self.n_init, self.seed = int(max_iter), tol, int(n_init), int(seed)

    @staticmethod
    def _values(values):
        if np.iscomplexobj(values):
            raise ValueError('Reference observations must be real')
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2 or not x.shape[1] or not np.isfinite(x).all():
            raise ValueError('Reference observations must be a finite (N,d) matrix')
        return x

    def _initialize(self, x, seed):
        n, d = x.shape
        labels = KMeans(self.n_components, n_init=1, max_iter=100, random_state=seed).fit_predict(x)
        noise = np.broadcast_to(np.asarray(self.noise_initial, dtype=np.float64), (d,)).copy()
        if not np.isfinite(noise).all() or (noise <= 0).any():
            raise ValueError('noise_initial must be positive variances')
        self.noise_ = np.maximum(noise, self.noise_floor)
        self.means_ = np.zeros((self.n_components, d))
        self.loadings_ = np.zeros((self.n_components, d, self.n_factors))
        self.weights_ = np.zeros(self.n_components)
        for k in range(self.n_components):
            group = x[labels == k]
            if not len(group):
                raise ValueError('Reference initialization found an empty component')
            self.means_[k] = group.mean(0)
            self.weights_[k] = len(group) / n
            centered = group - self.means_[k]
            rank = min(self.n_factors, *centered.shape)
            _, singular, vh = randomized_svd(centered, n_components=rank, n_iter=4,
                                             random_state=(seed + k) % (2**32))
            variance = np.maximum(singular**2 / len(group) - self.noise_.mean(),
                                  max(float(np.var(x)) * .001, self.noise_floor))
            self.loadings_[k, :, :rank] = vh.T * np.sqrt(variance)

    def _component_terms(self, x, k):
        w = self.loadings_[k]
        centered = x - self.means_[k]
        precision = np.eye(self.n_factors) + w.T @ (w / self.noise_[:, None])
        chol = cho_factor(precision, lower=True)
        covariance = cho_solve(chol, np.eye(self.n_factors))
        mean = cho_solve(chol, ((centered / self.noise_) @ w).T).T
        residual = centered - mean @ w.T
        quadratic = (residual**2 / self.noise_).sum(1) + (mean**2).sum(1)
        logdet = np.log(self.noise_).sum() + 2 * np.log(np.diag(chol[0])).sum()
        evidence = -.5 * (x.shape[1] * math.log(2 * math.pi) + logdet + quadratic)
        return evidence, mean, covariance

    def _e_step(self, x):
        terms = [self._component_terms(x, k) for k in range(self.n_components)]
        with np.errstate(divide='ignore'):
            joint = np.column_stack([term[0] for term in terms]) + np.log(self.weights_)
        normalizers = logsumexp(joint, axis=1)
        return np.exp(joint - normalizers[:, None]), normalizers, terms

    def _m_step(self, x, resp, terms):
        n, d = x.shape
        mass = resp.sum(0)
        self.weights_ = mass / n
        residual_sum = np.zeros(d)
        for k, (_, mean, covariance) in enumerate(terms):
            if mass[k] == 0:
                continue  # exact zero-weight component: no heuristic revival
            u = np.column_stack((np.ones(n), mean))
            f = u.T @ (resp[:, k, None] * u)
            f[1:, 1:] += mass[k] * covariance
            g = x.T @ (resp[:, k, None] * u)
            b = cho_solve(cho_factor(f, lower=True), g.T).T
            self.means_[k], self.loadings_[k] = b[:, 0], b[:, 1:]
            residual_sum += ((resp[:, k, None] * x**2).sum(0) - 2 * (b*g).sum(1)
                             + ((b @ f) * b).sum(1))
        self.noise_ = np.maximum(residual_sum / n, self.noise_floor)

    def fit(self, observations):
        x = self._values(observations)
        if len(x) < self.n_components or self.n_factors > x.shape[1]:
            raise ValueError('Require N >= K and d >= factor rank')
        best, failures = None, []
        with threadpool_limits(limits=1):
            for initialization in range(self.n_init):
                try:
                    self._initialize(x, (self.seed + initialization) % (2**32))
                    resp, scores, terms = self._e_step(x)
                    history = [float(scores.sum())]
                    converged = False
                    for _ in range(self.max_iter):
                        self._m_step(x, resp, terms)
                        resp, scores, terms = self._e_step(x)
                        total = float(scores.sum())
                        tolerance = 1e-8 + 1e-10 * max(1., abs(total), abs(history[-1]))
                        if not np.isfinite(total) or total < history[-1] - tolerance:
                            raise FloatingPointError('Reference EM likelihood decreased')
                        gain = total - history[-1]
                        history.append(total)
                        if abs(gain) <= self.tol * max(1., abs(total)):
                            converged = True
                            break
                    if best is None or history[-1] > best[0]:
                        best = (history[-1], self.means_.copy(), self.loadings_.copy(),
                            self.noise_.copy(), self.weights_.copy(), resp.copy(), history,
                            converged, initialization)
                except (np.linalg.LinAlgError, FloatingPointError) as error:
                    failures.append(str(error))
        if best is None:
            raise FloatingPointError(f'All reference initializations failed: {failures}')
        (self.log_likelihood_, self.means_, self.loadings_, self.noise_, self.weights_,
         self.responsibilities_, history, self.converged_, self.initialization_) = best
        self.log_likelihood_history_ = np.asarray(history)
        self.n_iter_ = len(history) - 1
        self.labels_ = self.responsibilities_.argmax(1)
        self.failed_initializations_ = failures
        return self

    def score_samples(self, observations):
        if not hasattr(self, 'log_likelihood_'):
            raise RuntimeError('Fit the finite MFA reference first')
        x = self._values(observations)
        if x.shape[1] != self.means_.shape[1]:
            raise ValueError('Reference feature dimension mismatch')
        return self._e_step(x)[1]

    def predict_proba(self, observations):
        self.score_samples(observations)  # input/fitted checks
        return self._e_step(self._values(observations))[0]

    def bic(self, observations):
        """Conventional BIC heuristic; mixtures are singular statistical models."""
        x = self._values(observations)
        k, d, r = self.n_components, x.shape[1], self.n_factors
        dimension = k - 1 + k*d*(r + 1) - k*r*(r - 1)//2 + d
        return -2 * self.score_samples(x).sum() + dimension * math.log(len(x))
