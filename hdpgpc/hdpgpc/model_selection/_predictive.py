"""Cheap, frozen-parameter MAP validation, separate from bnomad inference.

Each mixture density is evaluated before conditioning on that observation.
Validation conditions a local state copy for the MAP component, retains fitted
parameters, and never births components. This is a screening approximation to
assignment uncertainty, not the native variational objective or an exact
switching-process marginal likelihood.
"""
import numpy as np
import torch
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.special import logsumexp


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


class GaussianNoise:
    """Factor observation noise once; use a small determinant for affine maps."""
    def __init__(self, covariance, C):
        self.R, self.C = array(covariance), array(C)
        self.diagonal = np.count_nonzero(self.R) == np.count_nonzero(np.diag(self.R))
        if self.diagonal:
            self.variance = np.diag(self.R)
            if np.any(self.variance <= 0):
                raise np.linalg.LinAlgError('Nonpositive observation noise')
            self.logdet = np.log(self.variance).sum()
            self.whitened_C = self.C/np.sqrt(self.variance[:, None])
        else:
            self.factor = cholesky(self.R, lower=True)
            self.logdet = 2*np.log(np.diag(self.factor)).sum()
            self.whitened_C = solve_triangular(self.factor, self.C, lower=True)

    def score(self, residual, covariance):
        residual = array(residual).reshape(-1)
        P = array(covariance)
        root = cholesky(.5*(P+P.T), lower=True)
        projected = self.whitened_C @ root
        small = np.eye(len(P))+projected.T @ projected
        factor = cholesky(.5*(small+small.T), lower=True)
        whitened = (residual/np.sqrt(self.variance) if self.diagonal else
                    solve_triangular(self.factor, residual, lower=True))
        solution = cho_solve((factor, True), projected.T @ whitened)
        remainder = whitened-projected @ solution
        # Avoid subtracting nearly equal large energies at small Sigma.
        quadratic = remainder @ remainder+solution @ solution
        logdet = self.logdet+2*np.log(np.diag(factor)).sum()
        return -.5*(len(residual)*np.log(2*np.pi)+logdet+quadratic)


def _gaussian_score(residual, covariance):
    covariance = array(covariance)
    factor = cholesky(.5*(covariance+covariance.T), lower=True)
    residual = array(residual).reshape(-1)
    return -.5*(len(residual)*np.log(2*np.pi)+2*np.log(np.diag(factor)).sum()
                +residual @ cho_solve((factor, True), residual))


def validate_predictive(model, observations, grid):
    """Score a chronological suffix without changing the fitted model."""
    from ..emissions._fixed_observation import AffineGaussianObservation, _AffineConditioner
    ids = model.cluster_ids_.astype(int)
    labels = model.labels_
    slots = {int(k): j for j, k in enumerate(ids)}
    hmm = model.config.hdp.hmm_switch
    if hmm:
        transition = array(model._backend.transTheta)[np.ix_(ids, ids)]
        if np.any(transition.sum(1) <= 0):
            raise ValueError('No represented validation successor')
        transition = transition/transition.sum(1, keepdims=True)
        weights = np.zeros(len(ids)); weights[slots[int(labels[-1])]] = 1.
    else:
        weights = np.array([(labels == k).sum() for k in ids], dtype=float)
        weights /= weights.sum()
    states, models, noise, conditioners = [], [], [], []
    for k in ids:
        states.append([]); models.append([]); noise.append([]); conditioners.append([])
        for output, state in enumerate(model.posterior(int(k))):
            gp = model._backend.gpmodels[output][int(k)]
            mean = gp.cond_to_cuda(gp.cond_to_torch(np.array(state.filtered_mean))).reshape(-1, 1)
            covariance = gp.cond_to_cuda(gp.cond_to_torch(np.array(state.filtered_covariance)))
            states[-1].append((mean, covariance)); models[-1].append(gp)
            fixed = getattr(gp, 'fixed_observation_model', None)
            if type(fixed) is AffineGaussianObservation:
                noise[-1].append(GaussianNoise(gp.Sigma[-1], fixed.C))
                conditioners[-1].append(_AffineConditioner(fixed, mean, gp.Sigma[-1]))
            else:
                noise[-1].append(None); conditioners[-1].append(None)
    losses, assignments = [], []
    physical = model.config.dynamics.clock == 'physical-clock'
    for observation in observations:
        prior = weights @ transition if hmm else weights
        component_scores, predictions = [], []
        for j, k in enumerate(ids):
            predictions.append([]); score = 0.
            for output, gp in enumerate(models[j]):
                mean, covariance = states[j][output]
                A, Q, C, R = gp.A[-1], gp.Gamma[-1], gp.C[-1], gp.Sigma[-1]
                mean = A @ mean
                if gp.constant_input:
                    mean = mean+gp.b[-1]
                covariance = A @ covariance @ A.T+Q
                covariance = .5*(covariance+covariance.T)
                predictions[-1].append((mean, covariance))
                y = observation[:, output]
                if noise[j][output] is not None:
                    fixed = gp.fixed_observation_model
                    residual = y-fixed.offset-array(C @ mean).reshape(-1)
                    score += noise[j][output].score(residual, covariance)
                else:
                    coordinates = gp.cond_to_cuda(gp.cond_to_torch(grid))
                    projected, R_y = gp._observation_distribution(coordinates, gp.x_basis, C @ mean, R)
                    P_y = gp._observation_latent_cov(coordinates, covariance, C)
                    score += _gaussian_score(y-array(projected).reshape(-1), R_y+P_y)
            component_scores.append(score)
        log_prior = np.full(len(prior), -np.inf)
        np.log(prior, out=log_prior, where=prior > 0)
        joint = log_prior+component_scores
        losses.append(float(-logsumexp(joint)/observation.size))
        selected = int(np.argmax(joint)); assignments.append(int(ids[selected]))
        if physical:
            states = predictions
        for output, gp in enumerate(models[selected]):
            mean, covariance = predictions[selected][output]
            y = gp.cond_to_cuda(gp.cond_to_torch(observation[:, output])).reshape(-1, 1)
            conditioner = conditioners[selected][output]
            if conditioner is not None:
                mean, covariance = conditioner.condition(mean, covariance, y, responsibility=1.)
            else:
                eye = torch.eye(len(mean), dtype=mean.dtype, device=mean.device)
                coordinates = gp.cond_to_cuda(gp.cond_to_torch(grid))
                mean, covariance = gp.gp.posterior(mean, covariance, y, eye, torch.zeros_like(eye),
                    gp.C[-1], gp.Sigma[-1], x_train=coordinates, x_warped=coordinates, h=1.)
            states[selected][output] = (mean.detach(), covariance.detach())
        if hmm:
            weights = np.zeros(len(ids)); weights[selected] = 1.
    if not np.isfinite(losses).all():
        raise ValueError('Nonfinite validation predictive density')
    return {'nll_per_feature': float(np.mean(losses)), 'losses': losses,
            'assignments': assignments}


def gap_diagnostics(model, horizon=128):
    """Read-only covariance growth under fixed fitted matrices, without clipping."""
    radii, growth = [], []
    for k in model.cluster_ids_:
        for output, state in enumerate(model.posterior(int(k))):
            gp = model._backend.gpmodels[output][int(k)]
            A, Q = array(gp.A[-1]), array(gp.Gamma[-1])
            radii.append(float(np.abs(np.linalg.eigvals(A)).max()))
            P = np.array(state.filtered_covariance)
            initial = max(float(np.trace(P)), np.finfo(float).tiny)
            with np.errstate(over='ignore', invalid='ignore'):
                for _ in range(horizon):
                    P = A @ P @ A.T+Q
                    if not np.isfinite(P).all():
                        break
            ratio = float(np.trace(P)/initial)
            growth.append(ratio if np.isfinite(ratio) else None)
    return {'horizon_ticks': horizon, 'spectral_radii': radii,
            'step_unit': ('global_sampling_ticks' if model.config.dynamics.clock == 'physical-clock'
                          else 'retained_cluster_observations'),
            'latent_covariance_trace_growth': growth,
            'scope': 'Fixed fitted matrices; a diagnostic, not a stability constraint or ranking penalty'}
