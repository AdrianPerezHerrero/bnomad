"""Emission optimizer objectives and fitting schedules; preserve stopping/order."""
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
from sklearn.gaussian_process.kernels import ConstantKernel, Product, RBF, Sum, WhiteKernel
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
def fit_torch(self, x, y, alpha_ini, gamma_ini, reduced_points=False, verbose=False):
    """Optimize the parameters for a RBF kernel of the GP using one sample.
     Parameters
    ----------
    x : array (s_array) x train of the example y

    y : array (s_array) example y

    Returns
    -------
    fitted : True of False depending on the optimization succesful or not
    """
    if not self.fitted:
        if verbose:
            print("\n Fitting_GP: \n")
        x_ = x.detach().T[0]
        y_ = y.detach().T[0]
        x_basis = self.x_basis.T[0].detach().clone()
        if self.cuda and torch.cuda.is_available():
            x_basis = x_basis.cuda()
        if not (isinstance(self.kernel, Sum)
                and isinstance(self.kernel.k1, Product)
                and isinstance(self.kernel.k1.k1, ConstantKernel)
                and isinstance(self.kernel.k1.k2, RBF)
                and isinstance(self.kernel.k2, WhiteKernel)):
            raise ValueError('GP fitting supports ConstantKernel * RBF + WhiteKernel; '
                             'use fit_gp=False to keep another supplied sklearn kernel fixed')
        alpha_ini_bounds = self.kernel.k2.noise_level_bounds
        lengthscale_bounds = self.kernel.k1.k2.length_scale_bounds
        outputscale_bounds = self.kernel.k1.k1.constant_value_bounds
        for name, bounds in (('white noise', alpha_ini_bounds),
                             ('lengthscale', lengthscale_bounds),
                             ('outputscale', outputscale_bounds)):
            if isinstance(bounds, str) or bounds[0] >= bounds[1]:
                raise ValueError(f'GP fitting requires a nonzero {name} bound interval; '
                                 'use fit_gp=False for fixed kernel parameters')
        lik = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.Interval(*alpha_ini_bounds))
        if reduced_points or not torch.equal(x_basis, x_):
            gp = ProjectedGPModel(x_, y_, lik, x_basis)
            scale_kernel = gp.covar_module.base_kernel
        else:
            gp = ExactGPModel(x_, y_, lik)
            scale_kernel = gp.covar_module

        scale_kernel.base_kernel.register_constraint(
            'raw_lengthscale', gpytorch.constraints.Interval(*lengthscale_bounds))
        scale_kernel.register_constraint(
            'raw_outputscale', gpytorch.constraints.Interval(*outputscale_bounds))

        def inside(value, bounds):
            lower, upper = map(float, bounds)
            width = upper - lower
            lower_margin = min(width / 4, max(abs(lower) * 1e-6, np.finfo(float).eps))
            upper_margin = min(width / 4, max(abs(upper) * 1e-6, np.finfo(float).eps))
            return float(np.clip(value, lower + lower_margin, upper - upper_margin))

        scale_kernel.base_kernel.lengthscale = inside(
            self.kernel.k1.k2.length_scale, lengthscale_bounds)
        scale_kernel.outputscale = inside(
            self.kernel.k1.k1.constant_value, outputscale_bounds)
        lik.noise = inside(self.kernel.k2.noise_level, alpha_ini_bounds)

        lik.train()
        gp.train()

        training_iter = 4000

        if not reduced_points and not torch.equal(x_basis, x_):
            optimizer = torch.optim.Adam([{'params': gp.covar_module.base_kernel.parameters(), 'lr': 0.05},
                                          {'params': gp.likelihood.parameters()}], lr=0.05)
            training_iter = 2000
        elif hasattr(gp.covar_module, 'inducing_points'):
            optimizer = torch.optim.Adam([{'params': gp.covar_module.inducing_points, 'lr': 0.1},
                                          {'params': gp.covar_module.base_kernel.parameters(), 'lr': 0.1},
                                          {'params': gp.likelihood.parameters(), 'lr': 0.1}])
            training_iter = 5000
        else:
            optimizer = torch.optim.Adam(gp.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, gp)

        losses = []

        # CUDA OPTION
        if torch.cuda.is_available() and self.cuda:
            gp = gp.cuda()
            lik = lik.cuda()

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = gp(x_)
            # Calc loss and backprop gradients
            loss = -mll(output, y_)
            loss.backward()
            losses.append(loss.item())
            if verbose and i % 500 == 0:
                print('Iter %d/%d - Loss: %.3f' % (
                    i + 1, training_iter, loss.item()))
            optimizer.step()
            if len(losses) > 1000:
                # if np.isclose(np.sum(np.subtract(losses[-10:], losses[-11:-1])), 0, atol=1e-4):
                if np.isclose(np.sum(np.subtract(losses[-10:], losses[-11:-1])), 0, atol=1e-4):
                    break

        gp.eval()
        lik.eval()

        if verbose:
            print_hyperparams(gp, self.cuda)

        if type(gp) is ExactGPModel:
            if hasattr(self.kernel.k1, "k1"):
                self.kernel.k1.k1.theta = np.log(np.array([gp.covar_module.outputscale.item()]))
            if hasattr(self.kernel.k1, "k2"):
                self.kernel.k1.k2.theta = np.log(
                    np.array([gp.covar_module.base_kernel.lengthscale.item()]))
            else:
                self.kernel.k1.theta = np.log(np.array([gp.covar_module.base_kernel.lengthscale.item()]))
            self.kernel.k2.theta = np.log(np.array([lik.noise.item()]))
        elif type(gp) is ProjectedGPModel:
            self.x_basis, _ = torch.sort(torch.atleast_2d(x_basis).T, axis=0)
            # Check if some points collapsed
            if reduced_points:
                x_basis_clean = torch.clone(self.x_basis.detach())
                collapsed = False
                for i in torch.arange(self.x_basis.shape[0] - 1):
                    if self.x_basis[i + 1] - self.x_basis[i] < torch.log(
                            gp.covar_module.base_kernel.base_kernel.lengthscale):
                        print("Inducing point removed, ", str(self.x_basis[i]))
                        x_basis_clean = x_basis_clean[x_basis_clean != self.x_basis[i]]
                        collapsed = True
                if collapsed:
                    self.x_basis = torch.atleast_2d(x_basis_clean).T
            if self.cuda:
                self.x_basis = self.x_basis.cuda()
            if hasattr(self.kernel.k1, "k1"):
                self.kernel.k1.k1.theta = np.log(np.array([gp.covar_module.base_kernel.outputscale.item()]))
            if hasattr(self.kernel.k1, "k2"):
                self.kernel.k1.k2.theta = np.log(
                    np.array([gp.covar_module.base_kernel.base_kernel.lengthscale.item()]))
            else:
                self.kernel.k1.theta = np.log(
                    np.array([gp.covar_module.base_kernel.base_kernel.lengthscale.item()]))
            self.kernel.k2.theta = np.log(np.array([lik.noise.item()]))
        elif type(gp) is VarProjectedGPModel:
            self.x_basis = torch.clone(torch.sort(gp.variational_strategy.inducing_points, axis=0)[0]).detach()
            if hasattr(self.kernel.k1, "k1"):
                self.kernel.k1.k1.theta = np.log(np.array([gp.covar_module.outputscale.item()]))
            if hasattr(self.kernel.k1, "k2"):
                self.kernel.k1.k2.theta = np.log(np.array([gp.covar_module.base_kernel.lengthscale.item()]))
            else:
                self.kernel.k1.theta = np.log(np.array([gp.covar_module.base_kernel.lengthscale.item()]))
            self.kernel.k2.theta = np.log(np.array([lik.noise.item()]))
        if torch.cuda.is_available() and self.cuda:
            x_ = self.cond_to_numpy(self.x_basis.cpu())
            self.K_X_X = self.cond_to_torch(self.kernel(x_, x_)).cuda()
        else:
            x_ = self.cond_to_numpy(self.x_basis)
            self.K_X_X = self.cond_to_torch(self.kernel(x_, x_))
        self.K_inv = self.inv_r("kernelMat", self.K_X_X)
        self.fitted = True
        try:
            id = torch.eye(self.x_basis.shape[0])
            alph_ = self.cond_to_torch(self.kernel.k2.noise_level)
            gam_ = self.cond_to_torch(gamma_ini)
            if torch.cuda.is_available() and self.cuda:
                id = id.cuda()
                alph_ = alph_.cuda()
                gam_ = gam_.cuda()
            alph_ = torch.mul(alph_, id)
            gam_ = torch.mul(gam_, id)
            self.assign_alpha_ini(alph_, gam_)
        except AttributeError:
            self.assign_alpha_ini(alpha_ini, gamma_ini)
    return self.fitted


def fit(self, x, y, alpha_ini, gamma_ini, n_restarts_optimizer, random_state=None, valid=True):
    """Compute optimal hyperparameter for kernel maximizing marginal likelihood.

    Parameters
    ----------
    x_post : array-like of shape (m_samples) target inputs.

    y_mean : array-like of shape (s_samples) mean of basis vectors.

    n_restarts_optimizer : int number of points from where is started the search
    of the hyperparameters

    random_state : int for reproducibility of results

    Returns
    -------
    self : an instance of self with hyperparameter stablished on kernel

    parameters : return a double or array of doubles with optimal hyperparameters of kernel.

    log_marginal_likelihood_value : double indicating value of log marginal likelihood for
    computed hyperparameters.
    """
    x = self.cond_to_numpy(x)
    y = self.cond_to_numpy(y)
    alpha_ini = self.cond_to_numpy(alpha_ini)
    gamma_ini = self.cond_to_numpy(gamma_ini)

    if self.fitted == False:
        # Normalize y_target.
        y_train_mean = np.mean(y, axis=0)
        y_train_std = np.std(y, axis=0)

        # Remove mean and make unit variance
        if not y_train_std == 0:
            y = (y - y_train_mean) / y_train_std
            alpha_ini = alpha_ini / y_train_std ** 2
        else:
            y = y - y_train_mean
            alpha_ini = alpha_ini

        # Choose hyperparameters based on maximizing the log-marginal
        # likelihood (potentially starting from several initial values)
        def obj_func(theta, eval_gradient=True):
            if eval_gradient:
                lml, grad = self.log_marginal_likelihood(x, y, alpha_ini,
                                                         theta, eval_gradient=True, clone_kernel=True)
                return -lml, -grad
            else:
                return -self.log_marginal_likelihood(x, y, alpha_ini, theta,
                                                     clone_kernel=True)

            # First optimize starting from theta specified in kernel

        optima = [(self._constrained_optimization(obj_func,
                                                  self.kernel.theta,
                                                  self.kernel.bounds))]

        rng = check_random_state(random_state)
        # Additional runs are performed from log-uniform chosen initial
        # theta
        if n_restarts_optimizer > 0:
            if not np.isfinite(self.kernel.bounds).all():
                raise ValueError(
                    "Multiple optimizer restarts (n_restarts_optimizer>0) "
                    "requires that all bounds are finite.")
            bounds = self.kernel.bounds
            for iteration in range(n_restarts_optimizer):
                theta_initial =\
                    rng.uniform(bounds[:, 0], bounds[:, 1])
                optima.append(
                    self._constrained_optimization(obj_func, theta_initial,
                                                   bounds))
        # Select result from run with minimal (negative) log-marginal
        # likelihood and transforming it into theta which is in log form.
        lml_values = list(map(itemgetter(1), optima))
        self.kernel.theta = optima[np.argmin(lml_values)][0]
        # self.kernel._check_bounds_params()

        self.log_marginal_likelihood_value = -np.min(lml_values)

        # Recompute values of the GP model in terms of new kernel
        if type(self.x_basis) is torch.Tensor:
            if self.x_basis.is_cuda:
                x_ = self.x_basis.cpu()
                self.K_X_X = self.cond_to_torch(self.kernel(x_, x_))
            else:
                self.K_X_X = self.cond_to_torch(self.kernel(self.x_basis, self.x_basis))
        else:
            self.K_X_X = self.cond_to_torch(self.kernel(self.x_basis, self.x_basis))
        # self.K_inv = np.linalg.inv(self.K_X_X)
        self.K_inv = self.cond_to_torch(self.inv_r("K_X_X", self.K_X_X))
        if torch.cuda.is_available() and self.cuda:
            self.K_X_X = self.K_X_X.cuda()
            self.K_X_X.requires_grad = False
            self.K_inv = self.K_inv.cuda()
            self.K_inv.requires_grad = False
        # Assign alpha_ini to the model
        try:
            self.assign_alpha_ini(self.kernel.k2.noise_level ** 2, gamma_ini)
        except AttributeError:
            self.assign_alpha_ini(alpha_ini, gamma_ini)
        if valid:
            self.fitted = True
    return self.kernel.theta, self.log_marginal_likelihood_value


def log_likelihood(self, t0, t1, means, covars, A=None, Gamma=None, y=None, C=None, Sigma=None):
    """
    Return log transformed likelihood of the full joint distribution
    depending on what inputs were given.

    Parameters
    ----------
    t0: integer time of starting sequence whose we want to compute likelihood.
    t1: integer time of ending sequence whose we want to compute likelihood.
    means : list of arrays-like with shape (txs_samples)
        Means of the distributions generated by the iteration of GP
    covars : list of matrix-like with shape (txsxs_samples)
        Covariance matrices of the distributions generated by the iteration of GP
    A : matrix-like with shape (sxs_samples)
        Matrix that characterize the linear model
    Gamma: matrix-like with shape (sxs_samples)
        Matrix of noise associated with linear model
    C: matrix-like with shape (sxs_samples)
        Matrix of linear projection between latent state and observations
    Sigma: matrix-like with shape (sxs_samples)
        Matrix of noise associated with observations
    y : list of array-like with shape (txs_samples), optional
        If included the likelihood is computed also in terms of the likelihood of the observations.
        The default is None.

    Returns
    -------
    Negative scalar, log-likelihood of the model with given parameters.

    """
    # Ensure y is a tensor
    if type(y) is list:
        if type(y[0]) is torch.Tensor:
            y = y
    else:
        y = self.cond_to_torch(y)
    # If all None then simplest system
    if A is None:
        A = torch.eye(means[0].shape[0])
    if Gamma is None:
        Gamma = torch.zeros(covars[0].shape)
    if C is None:
        C = torch.eye(means[0].shape[0])
    if Sigma is None:
        Sigma = torch.zeros(covars[0].shape)
    n = means[0].shape[0]
    sum_0 = 0.0
    if t0 == 0:
        if isinstance(y, list):
            sum_0 = self.log_marginal_likelihood(self.x_basis, y[0], self.alpha_ini, self.kernel.theta,
                                                 clone_kernel=True)
        else:
            sum_0 = self.log_marginal_likelihood(self.x_basis, y, self.alpha_ini, self.kernel.theta,
                                                 clone_kernel=True)
    sum_0 = torch.tensor(sum_0)
    sum_1 = torch.tensor(0.0)
    T = t1 - t0
    zer = torch.zeros(covars[0].shape)
    if torch.cuda.is_available() and self.cuda:
        sum_0 = sum_0.cuda()
        sum_1 = sum_1.cuda()
        zer = zer.cuda()
        zer.requires_grad = False
    if not torch.allclose(Gamma, zer):
        if t1 > 1:
            C_t = Gamma
            det = self.log_det('Gamma', C_t)
            C_t_inv = self.inv_r('Gamma', C_t)
            for t in range(max(t0, 1), t1):
                exp_t_t = covars[t] + torch.matmul(means[t], means[t].T)
                sum_1 = sum_1 - torch.linalg.multi_dot([means[t + 1].T, C_t_inv, means[t + 1]])\
                        + 2 * torch.linalg.multi_dot([means[t + 1].T, C_t_inv, C, means[t]])\
                        - torch.trace(torch.linalg.multi_dot([C.T, C_t_inv, C, exp_t_t])) - det
            sum_1 = sum_1 - T * n * np.log(2 * np.pi)
            sum_1 = 0.5 * sum_1
    lik = sum_0 + sum_1
    # lik = sum_1
    if y is not None:
        sum_2 = torch.tensor(0.0)
        var = Sigma
        det = self.log_det('Sigma', var)
        var_inv = self.inv_r('Sigma', var)
        for t in range(t0, t1 + 1):
            exp_t_t = covars[t] + torch.matmul(means[t], means[t].T)
            if isinstance(y, list):
                sum_2 = sum_2 - torch.linalg.multi_dot([y[t].T, var_inv, y[t]])\
                        + 2 * torch.linalg.multi_dot([y[t].T, var_inv, C, means[t]])\
                        - torch.trace(torch.linalg.multi_dot([C.T, var_inv, C, exp_t_t])) - det
            else:
                sum_2 = sum_2 - torch.linalg.multi_dot([y.T, var_inv, y])\
                        + 2 * torch.linalg.multi_dot([y.T, var_inv, C, means[t]])\
                        - torch.trace(torch.linalg.multi_dot([C.T, var_inv, C, exp_t_t])) - det
        sum_2 = sum_2 - (T + 1) * n * np.log(2 * torch.pi)
        sum_2 = 0.5 * sum_2
        lik = lik + sum_2
    return lik


def log_marginal_likelihood(self, x_train, y_train, alpha_ini, theta=None, eval_gradient=False,
                            clone_kernel=True):
    """Returns log-marginal likelihood of theta for training data.

    Parameters
    ----------
    x_train : array-like of shape (s_samples) inputs associated with training inputs y

    y_train : array-like of shape (s_samples) observations of the sample

    theta : array-like of shape (n_kernel_params,) default=None
        Kernel hyperparameters for which the log-marginal likelihood is
        evaluated. If None, the precomputed log_marginal_likelihood
        of ``self.kernel_.theta`` is returned.

    eval_gradient : bool, default=False
        If True, the gradient of the log-marginal likelihood with respect
        to the kernel hyperparameters at position theta is returned
        additionally. If True, theta must not be None.

    clone_kernel : bool, default=True
        If True, the kernel attribute is copied. If False, the kernel
        attribute is modified, but may result in a performance improvement.

    Returns
    -------
    log_likelihood : float
        Log-marginal likelihood of theta for training data.

    log_likelihood_gradient : ndarray of shape (n_kernel_params,), optional
        Gradient of the log-marginal likelihood with respect to the kernel
        hyperparameters at position theta.
        Only returned when eval_gradient is True.
    """
    if torch.cuda.is_available() and self.cuda:
        x_train = x_train.cpu()
        y_train = y_train.cpu()
    x_train = self.cond_to_numpy(x_train)
    y_train = self.cond_to_numpy(y_train)
    theta = self.cond_to_numpy(theta)
    if theta is None:
        if eval_gradient:
            raise ValueError(
                "Gradient can only be evaluated for theta!=None")

    if clone_kernel:
        kernel = self.kernel.clone_with_theta(theta)
    else:
        kernel = self.kernel
        kernel.theta = theta

    if eval_gradient:
        K, K_gradient = kernel(x_train, eval_gradient=True)
    else:
        K = kernel(x_train)

    # K += alpha_ini
    try:
        L = cholesky(K, lower=True)  # Line 2
    except np.linalg.LinAlgError:
        return (-np.inf, np.zeros_like(theta))\
            if eval_gradient else -np.inf

    # Support multi-dimensional output of self.y_train_
    if y_train.ndim == 1:
        y_train = y_train[:, np.newaxis]

    alpha = cho_solve((L.dot(L.T), True), y_train)  # Line 3
    log_likelihood = -1 / 2 * y_train.T.dot(alpha) - np.log(np.diag(L)).sum() - K.shape[0] / 2 * np.log(2 * np.pi)

    if eval_gradient:  # compare Equation 5.9 from GPML
        tmp = np.einsum("ik,jk->ijk", alpha, alpha)  # k: output-dimension
        tmp -= cho_solve((L, True), np.eye(K.shape[0]))[:, :, np.newaxis]
        log_likelihood_gradient_dims =\
            0.5 * np.einsum("ijl,jik->kl", tmp, K_gradient)
        log_likelihood_gradient = log_likelihood_gradient_dims.sum(-1)

    if eval_gradient:
        return log_likelihood, log_likelihood_gradient
    else:
        return log_likelihood


def _constrained_optimization(self, obj_func, initial_theta, bounds):
    if self.optimizer == "fmin_l_bfgs_b":
        opt = {'maxiter': 50000, 'disp': False}
        opt_res = scipy.optimize.minimize(
            obj_func, initial_theta, method="L-BFGS-B", jac=True,
            bounds=bounds, options=opt)
        # _check_optimize_result("lbfgs", opt_res)
        theta_opt, func_min = opt_res.x, opt_res.fun
    elif self.optimizer == "CG":
        opt_res = scipy.optimize.minimize(
            obj_func, initial_theta, method="CG", jac=True)  # , options=opt)
        theta_opt, func_min = opt_res.x, opt_res.fun
    elif callable(self.optimizer):
        theta_opt, func_min =\
            self.optimizer(obj_func, initial_theta, bounds=bounds)
    else:
        raise ValueError("Unknown optimizer %s." % self.optimizer)

    return theta_opt, func_min

