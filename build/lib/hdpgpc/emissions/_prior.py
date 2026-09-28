# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 18:56:03 2021

@author: adrianperez
"""

import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from scipy.linalg import cholesky, cho_solve
import scipy.optimize
from sklearn.utils import check_random_state
import warnings
import torch
import gpytorch
from ._torch_models import ExactGPModel, ProjectedGPModel, VarProjectedGPModel
from ..diagnostics._plots import print_hyperparams
from tqdm import trange
from ..dynamics._models import _LINEAR
from ..dynamics._learning import estimate_lds_parameters

torch.set_default_dtype(torch.float64)


# Definition class of an Iterative Gaussian Process. Takes a kernel as parameters from the
# sklearn gaussian_process class.
class GaussianProcessPrior:
    """Gaussian Process from an iterative point of view, can compute posterior and
    update them recursively from new observations. It works by storing the posterior distribution
    and mixing a standard GP of 0 mean and kernel covariance function with posterior
    mean and posterior covariance between inputs.

        Parameters
        ----------
        kernel : kernel from sklearn class for covariance computation.

        x_basis : array-like of shape (s_samples) domain points where is wanted
        to focus the learning.

        alpha : positive double or array-like of shape (n_samples) noise associated with subyacent process.

        Returns
        -------
        self : returns an instance of self.
        """

    def __init__(self, kernel, x_basis, cuda=False, verbose=True):
        self.kernel = kernel
        self.x_basis = torch.tensor(x_basis) if not type(x_basis) is torch.Tensor else x_basis
        self.alpha_ini = 0.0
        self.gamma_ini = 0.0
        self.zero = self.cond_to_torch([0.0])
        if type(x_basis) is torch.Tensor:
            if x_basis.is_cuda:
                self.K_X_X = torch.tensor(kernel(x_basis.cpu()), requires_grad=False)
            else:
                self.K_X_X = torch.tensor(kernel(x_basis), requires_grad=False)
        else:
            self.K_X_X = torch.tensor(kernel(x_basis), requires_grad=False)
        self.K_inv = 0.0  # np.linalg.inv(self.K_X_X + alpha_ini)
        self.optimizer = "fmin_l_bfgs_b"
        self.fitted = False
        self.log_marginal_fitted_value = 0.0
        self.cuda = cuda
        self.verbose = verbose
        self.disable = not verbose
        if self.cuda:
            self.zero = self.zero.cuda()
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def posterior(self, mean_prior, cov_prior, y_train, A, Gamma, C, Sigma, warped_mean=None, x_train=None,
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
        return _LINEAR.filter(self, mean_prior, cov_prior, y_train, A, Gamma, C, Sigma,
                              warped_mean=warped_mean, x_train=x_train, x_warped=x_warped,
                              embedding=embedding, h=h)

    def projection_matrix(self, x_basis=None, x_train=None):
        """Compute projection matrix using the optimised GP:
            K_t = K_{t,t_n}K_{t_n,t_n}^{-1}

        Parameters
        ----------
        x_basis : array-like of shape (s_samples) basis vectors

        x_train : array-like of shape (s*_samples) train vector

        Returns
        -------
        K_nn_inv :  matrix-like of shape (s*xs_samples) projection matrix
        """
        if x_basis is None:
            x_basis_ = self.x_basis
        else:
            x_basis_ = x_basis
        if x_train is None:
            x_train_ = self.x_train
        else:
            x_train_ = x_train
        if self.cuda and torch.cuda.is_available():
            x_basis_ = x_basis_.cpu()
            x_train_ = x_train.cpu()
        if torch.equal(x_basis_, x_train_):
            K_nn_inv = torch.eye(x_basis.shape[0])
            if self.cuda and torch.cuda.is_available():
                K_nn_inv = K_nn_inv.cuda()
        else:
            x_basis_ = self.cond_to_numpy(x_basis_)
            x_train_ = self.cond_to_numpy(x_train_)
            jitter = 1e-4
            K_mn = self.cond_to_torch(self.kernel(x_basis_, x_train_))
            K_nn = self.cond_to_torch(self.kernel(x_train_, x_train_)) + jitter * torch.eye(x_train_.shape[0])
            if self.cuda and torch.cuda.is_available():
                K_mn = K_mn.cuda()
                K_nn = K_nn.cuda()
            K_nn_inv = torch.linalg.solve(K_nn.T, K_mn.T).T
        return K_nn_inv

    def project_y(self, x_train, y, cov, C, Sigma, x_basis=None):
        """
        Compute projection of observations into basis dimension.

        Parameters
        -----------
        x_train: domain of the observations
        y: observations
        x_basis : domain of output
        Returns:
        y_p: projected observations using GP
        y_p_cov: projected covariance of observations
        """
        if x_basis is None:
            x_basis_ = self.x_basis
        else:
            x_basis_ = x_basis
        if x_train is None:
            x_train_ = self.x_train
        else:
            x_train_ = x_train
        if self.cuda and torch.cuda.is_available():
            x_basis_ = x_basis_.cpu()
            x_train_ = x_train.cpu()
        if torch.equal(x_basis_, x_train_):
            y_p = y
            y_p_cov = Sigma
            if self.cuda and torch.cuda.is_available():
                y_p = y_p.cuda()
                y_p_cov = y_p_cov.cuda()
        else:
            x_basis_ = self.cond_to_numpy(x_basis_)
            x_train_ = self.cond_to_numpy(x_train_)
            jitter = 1e-4
            K_mn = self.cond_to_torch(self.kernel(x_basis_, x_train_))
            K_nn = self.cond_to_torch(self.kernel(x_train_, x_train_)) + jitter * torch.eye(x_train_.shape[0])
            K_mm = self.cond_to_torch(self.kernel(x_basis_, x_basis_)) + jitter * torch.eye(x_basis_.shape[0])
            if self.cuda and torch.cuda.is_available():
                K_mn = K_mn.cuda()
                K_nn = K_nn.cuda()
                K_mm = K_mm.cuda()
            K_nn_inv = torch.linalg.solve(K_nn.T, torch.matmul(C, K_mn).T).T
            y_p = torch.linalg.multi_dot([K_nn_inv, y])
            y_p_cov = Sigma
        return y_p, y_p_cov

    def backward(self, A_prior, Gamma_prior, means, covars):
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
        return _LINEAR.smooth(self, A_prior, Gamma_prior, means, covars)

    def backward_notrange(self, A_prior, Gamma_prior, means, covars):
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
        return _LINEAR.smooth_pair(self, A_prior, Gamma_prior, means, covars)

    def new_params_LDS(self, A_prior, Gamma_prior, C_prior, Sigma_prior, y_samples, means, covars, model='dynamic'):
        """Compute new matrices of linear tranformation using ML approach following
        the model described by:
            f_t = A f_{t-1} + Gamma
            y_t = C f_t + Sigma
        Parameters
        ----------
        A_priors : list of matrix (txsxs_samples)
            Previous matrices of linear transformation.
        Gamma_priors : list of matrix (txsxs_samples)
            Previous matrices of projection noise.
        C_priors : list of matrix (txsxs_samples)
            Previous matrices observations projection.
        Sigma_priors : list of matrix (txsxs_samples)
            Previous matrices observation projection noise.
        y_samples : ordered array of arrays (txs_samples)
            List of ovservations received till moment.
        means : ordered array of arrays (txs_samples)
            List of means computed till moment
        covars : ordered array of matrix (txsxs_samples)
            List of covariance computed till moment
        Returns
        -------
        Matrices (sxs_samples) new matrices of linear transformation model
        which maximize likelihood of the model (A,Gamma, C, Sigma).
        """
        return estimate_lds_parameters(self, A_prior, Gamma_prior, C_prior, Sigma_prior,
                                       y_samples, means, covars, model=model)

    def pred_dist(self, x_post, x_fixed, mean_prior, Sigma):
        x_post_mean = self.compute_mean(x_post)
        x_train_mean = self.compute_mean(x_fixed)
        x_f = x_fixed
        x_p = x_post

        if torch.cuda.is_available() and self.cuda:
            x_f = x_f.cpu()
            x_p = x_p.cpu()

        if torch.equal(x_f, x_p):
            return mean_prior, Sigma

        ker = self.kernel.clone_with_theta(self.kernel.theta)
        x_f_np = self.cond_to_numpy(x_f)
        x_p_np = self.cond_to_numpy(x_p)

        K_X_X = self.cond_to_torch(ker(x_f_np, x_f_np))
        K_X_Xs = self.cond_to_torch(ker(x_f_np, x_p_np))
        K_Xs_Xs = self.cond_to_torch(self.kernel(x_p_np))

        if torch.cuda.is_available() and self.cuda:
            K_X_X = K_X_X.cuda()
            K_X_Xs = K_X_Xs.cuda()
            K_Xs_Xs = K_Xs_Xs.cuda()

        n = K_X_X.shape[0]
        m = K_Xs_Xs.shape[0]
        id_n = torch.eye(n, device=K_X_X.device, dtype=K_X_X.dtype)
        id_m = torch.eye(m, device=K_Xs_Xs.device, dtype=K_Xs_Xs.dtype)

        jitter = 1e-4 * torch.mean(torch.diag(Sigma).abs()).clamp_min(torch.finfo(Sigma.dtype).eps)
        L = torch.linalg.cholesky(0.5 * (K_X_X + K_X_X.T) + jitter * id_n)

        # Solve K^{-1} K_X_Xs
        K_solve = torch.cholesky_solve(K_X_Xs, L)  # (n,m)

        delta = mean_prior - x_train_mean
        f_star = x_post_mean + K_solve.T @ delta

        if torch.all(torch.isclose(torch.diag(Sigma), torch.mean(torch.diag(Sigma)))):
            cov_f = torch.mean(torch.diag(Sigma)) * id_m
        else:
            cov_f = K_Xs_Xs - K_X_Xs.T @ K_solve + K_solve.T @ Sigma @ K_solve
            cov_f = 0.5 * (cov_f + cov_f.T) + 1e-6 * id_m

        return f_star, cov_f

    def pred_latent_dist(self, x_post, x_fixed, mean_prior, cov_prior):
        """Compute posterior latent distribution using a Cholesky-based solve."""
        x_post_mean = self.compute_mean(x_post)
        x_train_mean = self.compute_mean(x_fixed)

        x_f = x_fixed
        x_p = x_post

        if torch.equal(x_f, x_p):
            f_star = mean_prior
            cov_f = cov_prior
        else:
            if torch.cuda.is_available() and self.cuda:
                x_f = x_f.cpu()
                x_p = x_p.cpu()

            x_f = self.cond_to_numpy(x_f)
            x_p = self.cond_to_numpy(x_p)

            K_X_X = self.cond_to_torch(self.kernel(x_f, x_f))
            K_X_Xs = self.cond_to_torch(self.kernel(x_f, x_p))
            K_Xs_X = self.cond_to_torch(self.kernel(x_p, x_f))
            K_Xs_Xs = self.cond_to_torch(self.kernel(x_p, x_p))

            if torch.cuda.is_available() and self.cuda:
                K_X_X = K_X_X.cuda()
                K_X_Xs = K_X_Xs.cuda()
                K_Xs_X = K_Xs_X.cuda()
                K_Xs_Xs = K_Xs_Xs.cuda()

            eye = torch.eye(K_X_X.shape[0], device=K_X_X.device, dtype=K_X_X.dtype)

            # Same jitter scale as original code
            cov = K_X_X + 1e-4 * eye

            # Cholesky factorization of cov
            L = torch.linalg.cholesky(cov)

            delta = mean_prior - x_train_mean

            # f_star = x_post_mean + K_Xs_X @ cov^{-1} @ delta
            sol_delta = torch.cholesky_solve(delta, L)
            f_star = x_post_mean + K_Xs_X @ sol_delta

            # term_data = K_Xs_X @ cov^{-1} @ K_X_Xs
            sol_K = torch.cholesky_solve(K_X_Xs, L)
            term_data = K_Xs_X @ sol_K

            # term_prior = K_Xs_X @ cov^{-1} @ cov_prior @ cov^{-T} @ K_X_Xs
            # First solve cov^T Z = K_X_Xs, but since cov is symmetric after jitter,
            # cov^{-T} = cov^{-1}, so reusing cholesky_solve is consistent here.
            middle = cov_prior @ sol_K
            sol_middle = torch.cholesky_solve(middle, L)
            term_prior = K_Xs_X @ sol_middle

            cov_f = K_Xs_Xs - term_data + term_prior

        return f_star, cov_f

    def sample_y(self, f_mean, f_cov, C, Sigma, n_samples=1, random_state=0):
        """Draw samples from Gaussian process and evaluate at X.

        Parameters
        ----------
        f_mean : mean of the distribution (s-array) paired with x_basis

        f_cov : covariance of the distribution (sxs-array)

        C: matrix (sxs_array) which represent projection over mean to observations.

        Sigma: matrix (sxs_array) which represent noise associated with observations.

        n_samples : int, default=1
            The number of samples drawn from the Gaussian process

        random_state : int, RandomState instance or None, default=0
            Determines random number generation to randomly draw samples.
            Pass an int for reproducible results across multiple function
            calls.
            See :term: `Glossary <random_state>`.

        Returns
        -------
        y_samples : ndarray of shape (n_samples_X, [n_output_dims], n_samples)
            Values of n_samples samples drawn from Gaussian process and
            evaluated at query points.
        """
        f_mean = self.cond_to_numpy(f_mean)
        f_cov = self.cond_to_numpy(f_cov)
        C = self.cond_to_numpy(C)
        Sigma = self.cond_to_numpy(Sigma)

        rng = check_random_state(random_state)
        mean = np.dot(C, f_mean)
        cov = np.dot(C, np.dot(f_cov, C.T)) + Sigma
        if f_mean.ndim == 1:
            y_samples = rng.multivariate_normal(mean, cov, n_samples).T
        else:
            y_samples =\
                [rng.multivariate_normal(mean[:, i], cov,
                                         n_samples).T[:, np.newaxis]
                 for i in range(f_mean.shape[1])]
            y_samples = np.hstack(y_samples)
        return y_samples

    from ._training import fit_torch

    from ._training import fit

    # Revisar computo verosimilitud para una muestra mas amplia.
    from ._training import log_likelihood

    from ._training import log_marginal_likelihood

    def KL_divergence(self, mean1, cov1, mean2, cov2):
        """
        Compute symmetric Kullback-Leibler divergence for two
        Gaussian distributions.

        Parameters
        ----------
        mean1 : array like of s_samples
            Mean of the first distribution
        cov1 : matrix like of sxs_samples
            Covariance matrix of first distribution
        mean2 : array like of s_samples
            Mean of the second distribution
        cov2 : matrix like of sxs_samples
            Covariance matrix of second distribution

        Returns
        -------
        Positive real value KL-divergence of distributions.

        """
        # Second term
        invcov1 = torch.linalg.inv(cov1)
        invcov2 = torch.linalg.inv(cov2)
        cov = torch.matmul(invcov2, cov1) + torch.matmul(invcov1, cov2)
        tr = (torch.trace(cov) - 2 * len(cov)) / 4

        # First term
        difmean = mean1 - mean2
        sumcov = invcov1 + invcov2
        prod = torch.matmul(sumcov, difmean)
        first = torch.sum(prod * difmean) / 4

        # Total
        dist = first + tr

        return dist.item()

    def compute_mean(self, x):
        """Compute prior mean of an input vector x

        Parameters
        ----------
        x : array-like of shape (m_samples) inputs.

        Returns
        -------
        mean : vector with prior mean over the inputs x
        """
        # Now assuming a 0 mean prior.
        if torch.cuda.is_available() and self.cuda:
            return torch.tensor(np.atleast_2d(np.repeat(0.0, x.shape[0])).T, requires_grad=False).cuda()
        else:
            return torch.tensor(np.atleast_2d(np.repeat(0.0, x.shape[0])).T, requires_grad=False)

    # Private function of optimization used for hyperparameter computation.
    from ._training import _constrained_optimization

    # Standard plot function
    def plotGP(self, numberFigure, x_post, y_pred, gamma, x_train=None, y_train=None, sigma=None, title=False,
               label_model=0, labels=False):
        x_post = self.cond_to_numpy(x_post)
        y_pred = self.cond_to_numpy(y_pred)
        gamma = self.cond_to_numpy(gamma)
        x_train = self.cond_to_numpy(x_train)
        y_train = self.cond_to_numpy(y_train)
        sigma = self.cond_to_numpy(sigma)

        plt.figure(numberFigure, figsize=(12, 6))
        if title == True:
            tit = 'Gaussian process with example:' + str(numberFigure)
            if not label_model == 0:
                tit = 'Gaussian process example:' + str(numberFigure - label_model) + ' with model:' + str(
                    int(label_model / 1000))
            plt.title(tit)
        if x_train is not None and y_train is not None:
            plt.plot(x_train, y_train, 'k.', markersize=2, label='Observations QRS complex')
        plt.plot(x_post, y_pred, 'b-', label='Prediction')
        plt.fill(np.concatenate([x_post, x_post[::-1]]),
                 np.concatenate([y_pred - 1.9600 * gamma,
                                 (y_pred + 1.9600 * gamma)[::-1]]),
                 alpha=.35, fc='b', ec='None', label='95% confidence interval')
        if sigma is not None:
            plt.fill(np.concatenate([x_post, x_post[::-1]]),
                     np.concatenate([y_pred - 1.9600 * sigma,
                                     (y_pred + 1.9600 * sigma)[::-1]]),
                     alpha=.2, fc='cyan', ec='None')
        if labels == True:
            plt.xlabel('$t$')

    # Robust computing of determinant
    def log_det(self, name, M):
        # Compute using formula log(det(C))=2*trace(log(L)) where L is cholesky lower decomposition
        M = self.cond_to_torch(M)
        id = torch.eye(M.shape[0])
        if torch.cuda.is_available() and self.cuda:
            M = M.cuda()
            id = id.cuda()
        try:
            # Compute order of magnitude
            od = torch.floor(torch.log10(torch.max(torch.diag(M))))
            # Compute the inverse to "normalize matrix"
            k = 10 ** -od
            # Apply reescale to the matrix
            M_aux = torch.mul(k, M)
            L = torch.linalg.cholesky(M_aux)
            log_diag = torch.log(torch.diag(L))
            # Apply reescale again to return to main order
            det = 2 * sum(log_diag) - len(M) * torch.log(k)
        except RuntimeError:
            od = torch.floor(torch.log10(torch.max(torch.diag(M))))
            k = 10 ** -od
            M_aux = torch.mul(k, M)
            jitter = torch.mul(1e-6, id)
            with warnings.catch_warnings(record=True) as w:
                # Cause all warnings to always be triggered.
                warnings.simplefilter("always")
                det = torch.log(torch.linalg.det(M_aux + jitter)) - len(M) * torch.log(k)
                # if w is not None:
                #     print("Stop")
        except ValueError:
            return torch.from_numpy(-np.inf)
        return det

    # Robust computing of the inverse
    def inv_r(self, name, M):
        M = self.cond_to_torch(M)
        if torch.cuda.is_available() and self.cuda:
            M = M.cuda()
        try:
            # Compute order of magnitude
            id = torch.eye(M.shape[0])
            if torch.cuda.is_available() and self.cuda:
                id = id.cuda()
                id.requires_grad = False
            od = torch.floor(torch.log10(torch.max(torch.diag(M))))
            # Compute the inverse to "normalize matrix"
            k = 10 ** -od
            # Apply reescale to the matrix
            M_aux = torch.mul(k, M)
            inv_aux = torch.linalg.solve(M_aux, id)
            # Apply reescale again to return to main order
            inv = torch.mul(k, inv_aux)
        except RuntimeError:
            inv = torch.linalg.solve(M, id)
        return inv

    def assign_alpha_ini(self, alpha, gamma):
        self.alpha_ini = alpha
        self.gamma_ini = gamma
        # self.covariance_kernel = WhiteKernel(noise_level=alpha*0.2, noise_level_bounds=(1e-10, 1e+1))

    def cond_to_numpy(self, x):
        if x is None:
            x = x
        elif type(x) is torch.Tensor:
            x = x.clone().detach().numpy()
        return x

    def cond_to_torch(self, x):
        if x is None:
            x = x
        elif type(x) is not torch.Tensor:
            x = torch.tensor(x, requires_grad=False)
        return x
