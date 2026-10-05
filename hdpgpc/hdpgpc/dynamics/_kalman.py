"""Original GP-conditioned Kalman and two distinct smoothing recurrences."""

import numpy as np
import torch


def filter_posterior(backend, mean_prior, cov_prior, y_train, A, Gamma, C, Sigma, warped_mean=None, x_train=None,
              x_warped=None, embedding=True, h=1.0):
    """Compute posterior distribution using an observation as an input and returns
    the mean and the covariance centered into the set of basis points.
    Assuming we follow the system described by:
        f_t = A f_{t-1} + Gamma
        y_t = C f_t + Sigma

    Parameters
    ----------
    mean_prior : array-like of shape (s_samples) prior mean on basis vectors

    cov_prior : matrix-like of shape (sxs_samples) prior covariance matrix on basis vectors

    y_train : array-like of shape (n_samples) target values paired with x_basis.

    A: matrix-like of shape (sxs_samples) matrix associated with the linear transformation of the dynamical model.

    Gamma: matrix-like of shape (sxs_samples) matrix represents the noise of projection of the previous mean.

    C: matrix-like of shape (sxs_samples) matrix associated with the projection of the latent variable to the observations.

    Sigma: matrix-like of shape (sxs_samples) matrix associated with noise of the observations (alpha simpler case).

    Returns
    -------
    mean_post :  array-like of shape (s_samples) posterior mean on basis vectors

    cov_post : matrix-like of shape (sxs_samples) posterior covariance matrix on basis vectors
    """

    y = y_train
    x_basis_mean = torch.matmul(A, mean_prior)
    x_basis_cov = cov_prior
    x_basis_ = backend.x_basis
    if x_train is None:
        x_train = backend.cond_to_torch(np.atleast_2d(np.arange(0, y_train.shape[0], dtype=np.float64)).T)
    if x_warped is None:
        x_warped = x_train.detach()
    if backend.cuda and torch.cuda.is_available():
        x_basis_cov = x_basis_cov.cuda()
        x_basis_mean = x_basis_mean.cuda()
        y = y.cuda()
        x_train = x_train.cpu()
        x_warped = x_warped.cpu()
        x_basis_ = x_basis_.cpu()

    l = x_basis_mean.shape[0]
    id = torch.eye(l)
    if torch.cuda.is_available() and backend.cuda:
        id = id.cuda()
    jitter = 1e-4 * id
    K_X_X = backend.cond_to_torch(backend.kernel(x_basis_, x_basis_))
    # K_X_X = backend.cond_to_torch(backend.kernel(backend.x_basis))
    K_Xs_X = backend.cond_to_torch(backend.kernel(x_warped, x_basis_))
    if backend.cuda and torch.cuda.is_available():
        K_X_X = K_X_X.cuda()
        K_Xs_X = K_Xs_X.cuda()
    if torch.equal(x_warped, x_basis_):
        K_cov = id
    else:
        K_cov = torch.linalg.solve((K_X_X + jitter).T, K_Xs_X.T).T
    P_t = A @ x_basis_cov @ A.T + Gamma
    # First step where initial covariance is needed as a prior, using initial Sigma computed by the GP optim.
    if torch.equal(cov_prior, K_X_X):
        P_t = x_basis_cov
        f_star, cov_f = backend.pred_dist(x_warped, backend.x_basis,
                                          C @ x_basis_mean, Sigma)
    else:
        mean = C @ x_basis_mean
        f_star, cov_f = backend.pred_dist(x_warped, backend.x_basis, mean, Sigma)
    if h < 1.0:
        # A fractional assignment supplies h times the observation precision,
        # including GP interpolation uncertainty.
        cov_f = cov_f / h
    # Forward equations of Kalman.
    KC = K_cov @ C
    K_t = torch.linalg.solve((KC @ P_t @ KC.T + cov_f).T,
                             (KC @ P_t.T)).T
    mean_post = x_basis_mean + K_t @ (y - f_star)
    # Joshep form
    IKC = id - K_t @ KC
    cov_post = IKC @ P_t @ IKC.T + K_t @ cov_f @ K_t.T
    return mean_post, cov_post


def smooth_sequence(backend, A_prior, Gamma_prior, means, covars):
    """
    Compute backward mean and cov incorporating full sequence
    information to previous latent states.

    Parameters
    ----------
    A_prior: matrix (sxs_samples)
        Matrix that determine linear transformation.
    Gamma_prior: matrix (sxs_samples)
        Matrix of error projection of linear transformation.
    means : ordered array of arrays (txs_samples)
        List of means computed till moment
    covars : ordered array of matrix (txsxs_samples)
        List of covariance computed till moment


    Returns
    -------
    Lists of new means (txs_samples) and covars (txsxs_samples).

    """
    T = len(means)
    for t in range(T - 2, -1, -1):
        A_prior_ = A_prior[t] if t < len(A_prior) else A_prior[-1]
        Gamma_prior_ = Gamma_prior[t] if t < len(Gamma_prior) else Gamma_prior[-1]
        P_t = torch.linalg.multi_dot([A_prior_, covars[t], A_prior_.T]) + Gamma_prior_
        J_t = torch.linalg.solve(P_t.T, A_prior_ @ covars[t].T).T
        means[t] = means[t] + torch.matmul(J_t, (means[t + 1] - torch.matmul(A_prior_, means[t])))
        eye = torch.eye(covars[t].shape[0], dtype=covars[t].dtype, device=covars[t].device)
        projection = eye - J_t @ A_prior_
        reverse_conditional = (projection @ covars[t] @ projection.T
                               + J_t @ Gamma_prior_ @ J_t.T)
        covars[t] = reverse_conditional + J_t @ covars[t + 1] @ J_t.T
    return means, covars


def smooth_pair(backend, A_prior, Gamma_prior, means, covars):
    """
    Compute backward mean and cov incorporating full sequence
    information to previous latent states.

    Parameters
    ----------
    A_prior: matrix (sxs_samples)
        Matrix that determine linear transformation.
    Gamma_prior: matrix (sxs_samples)
        Matrix of error projection of linear transformation.
    means : ordered array of arrays (txs_samples)
        List of means computed till moment
    covars : ordered array of matrix (txsxs_samples)
        List of covariance computed till moment


    Returns
    -------
    Lists of new means (txs_samples) and covars (txsxs_samples).

    """
    T = len(means)
    for t in range(T - 2, -1, -1):
        P_t = torch.linalg.multi_dot([A_prior, covars[t], A_prior.T]) + Gamma_prior
        J_t = torch.linalg.solve(P_t.T, torch.matmul(A_prior, covars[t].T)).T
        means[t] = means[t] + torch.matmul(J_t, (means[t + 1] - torch.matmul(A_prior, means[t])))
        eye = torch.eye(covars[t].shape[0], dtype=covars[t].dtype, device=covars[t].device)
        projection = eye - J_t @ A_prior
        reverse_conditional = (projection @ covars[t] @ projection.T
                               + J_t @ Gamma_prior @ J_t.T)
        covars[t] = reverse_conditional + J_t @ covars[t + 1] @ J_t.T
    return means, covars
