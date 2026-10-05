"""Cluster birth, pruning, copying and initialization policies.

Functions operate on the explicitly supplied mutable inference state.
Numerical expressions and mutation order are preserved from the executable specification.
"""

from copy import deepcopy

import numpy as np
import torch
import pandas as pd
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from scipy.special._ufuncs import psi as digamma, gammaln
from torchmetrics.audio import SignalNoiseRatio
from tqdm import trange
import pickle as plk
from ..emissions import _gaussian_process as gp
from ..alignment import _components as _alignment
from ..hdp._variational import HDPVariational
from ..hdp._optimizer import find_optimum_multiple_tries, kvec
from ..numerics._normalization import max_shift_log_scores
from ..hdp._entropy import state_entropy_terms, pair_entropy_terms


def keep_last_all(self):
    """ Method to reset full model and save space
    """
    for ld in range(self.n_outputs):
        for gp in self.gpmodels[ld]:
            gp.reinit_LDS(save_last=True)
            gp.reinit_GP(save_last=True, save_index=True)


def set_default_options(self, kernel, ini_sigma, ini_gamma, ini_outputscale, bound_sigma, bound_gamma,
                        bound_noise_warp, annealing, method_compute_warp,
                        model_type, recursive_warp, warp_updating, inducing_points, estimation_limit,
                        free_deg_MNIV):
    """ Default options definition.
    """
    self.kernel_def = kernel.clone_with_theta(kernel.theta)
    self.ini_sigma_def = ini_sigma
    self.ini_gamma_def = ini_gamma
    self.ini_outputscale_def = ini_outputscale
    self.bound_sigma_def = bound_sigma
    self.bound_gamma_def = bound_gamma
    self.bound_sigma_warp_def = bound_noise_warp
    self.annealing_def = annealing
    self.method_compute_warp_def = method_compute_warp
    self.model_type_def = model_type
    self.recursive_warp_def = recursive_warp
    self.warp_updating_def = warp_updating
    self.inducing_points_def = inducing_points
    self.estimation_limit_def = estimation_limit
    self.free_deg_MNIV = free_deg_MNIV


def get_default_options(self):
    return self.kernel_def, self.ini_sigma_def, self.ini_gamma_def, self.ini_outputscale_def, self.bound_sigma_def,\
        self.bound_gamma_def, self.bound_sigma_warp_def, self.annealing_def, self.method_compute_warp_def,\
        self.model_type_def, self.recursive_warp_def, self.warp_updating_def, self.inducing_points_def,\
        self.estimation_limit_def, self.free_deg_MNIV


def create_gp_default(self, i=None, fitted_kernel=None):
    """ Create a GP default when a birth happens.
    """
    if i is None or len(self.wp_sys[0]) <= i:
        kernel, ini_sigma, ini_gamma, ini_outputscale, bound_sigma, bound_gamma, bound_noise_warp, annealing, method_compute_warp,\
            model_type, recursive_warp, warp_updating, inducing_points, estimation_limit, free_deg_MNIV = self.get_default_options()
        kernel = fitted_kernel if fitted_kernel is not None else kernel
        kernel = kernel.clone_with_theta(kernel.theta)
        gp_ = gp.GaussianProcessEmission(kernel, self.x_basis_ini, annealing=annealing, bayesian=self.bayesian_params,
                           cuda=self.cuda,
                           inducing_points=inducing_points, estimation_limit=estimation_limit,
                           free_deg_MNIV=free_deg_MNIV, verbose=self.verbose,
                           lds_update=getattr(self, 'lds_update', 'conjugate'),
                           lds_step_size=getattr(self, 'lds_step_size', 0.2))
        if model_type == 'static':
            cond = gp_.GPR_static(ini_sigma)
        elif model_type == 'dynamic':
            cond = gp_.GPR_dynamic(ini_gamma, ini_sigma)
        else:
            print("Chosen model should be dynamic or static.")
        gp_.initial_conditions(ini_mean=None, ini_cov=None,
                               ini_A=cond[0], ini_Gamma=cond[1], ini_C=cond[2], ini_Sigma=cond[3])
        if fitted_kernel is not None:
            gp_.fitted = True
        # self.wp_sys.append(_alignment.configured(self).create_warper(self.x_basis_warp[0], self.noise_warp, bound_noise_warp,
        #                                       recursive=recursive_warp, cuda=self.cuda, mode=self.mode_warp))
        if self.cuda and torch.cuda.is_available():
            gp_.model_to_cuda()
            for ld in range(self.n_outputs):
                self.wp_sys[ld][-1].warp_gp.model_to_cuda()
        self.bound_sigma.append(bound_sigma)
        self.bound_gamma.append(bound_gamma)
        self.bound_sigma_warp.append(bound_noise_warp)
        self.annealing.append(annealing)
        self.recursive_warp.append(recursive_warp)
        self.warp_updating.append(warp_updating)
        self.model_type.append(model_type)
        self.x_basis.append(self.x_basis_ini)
        self.bound_sigma.append(bound_sigma)
        self.bound_sigma_warp.append(bound_noise_warp)
        self.inducing_points.append(inducing_points)
        self.estimation_limit.append(estimation_limit)
        for t, y_ in enumerate(self.y[:-1]):
            gp_.condition_observation(t, y_, 0, posterior=False)
        return gp_
    else:
        kernel, ini_sigma, ini_gamma, ini_outputscale, bound_sigma, bound_gamma, bound_noise_warp, annealing, method_compute_warp,\
            model_type, recursive_warp, warp_updating, inducing_points, estimation_limit, free_deg_MNIV = self.get_default_options()
        kernel = fitted_kernel if fitted_kernel is not None else kernel
        kernel = kernel.clone_with_theta(kernel.theta)
        gp_ = gp.GaussianProcessEmission(kernel, self.x_basis_ini, annealing=annealing, bayesian=self.bayesian_params,
                           cuda=self.cuda, inducing_points=inducing_points, estimation_limit=estimation_limit,
                           free_deg_MNIV=free_deg_MNIV, verbose=self.verbose,
                           lds_update=getattr(self, 'lds_update', 'conjugate'),
                           lds_step_size=getattr(self, 'lds_step_size', 0.2))
        if model_type == 'static':
            cond = gp_.GPR_static(ini_sigma)
        elif model_type == 'dynamic':
            cond = gp_.GPR_dynamic(ini_gamma, ini_sigma)
        else:
            print("Chosen model should be dynamic or static.")
        gp_.initial_conditions(ini_mean=None, ini_cov=None,
                               ini_A=cond[0], ini_Gamma=cond[1], ini_C=cond[2], ini_Sigma=cond[3])
        if fitted_kernel is not None:
            gp_.fitted = True
        for ld in range(self.n_outputs):
            self.wp_sys[ld][i] = _alignment.configured(self).create_warper(self.x_basis_warp[0], self.noise_warp, bound_noise_warp,
                                                recursive=recursive_warp, cuda=self.cuda, mode=self.mode_warp)
        if self.cuda and torch.cuda.is_available():
            gp_.model_to_cuda()
            self.wp_sys[i].warp_gp.model_to_cuda()
        self.bound_sigma[i] = bound_sigma
        self.bound_gamma[i] = bound_gamma
        self.bound_sigma_warp[i] = bound_noise_warp
        self.annealing[i] = annealing
        self.recursive_warp[i] = recursive_warp
        self.warp_updating[i] = warp_updating
        self.model_type[i] = model_type
        self.x_basis[i] = self.x_basis_ini
        self.bound_sigma[i] = bound_sigma
        self.bound_sigma_warp[i] = bound_noise_warp
        self.inducing_points[i] = inducing_points
        self.estimation_limit[i] = estimation_limit
        for t, y_ in enumerate(self.y[:-1]):
            gp_.condition_observation(t, y_, 0, posterior=False)
        return gp_


def create_wp_sys_default(self):
    kernel, ini_sigma, ini_gamma, ini_outputscale, bound_sigma, bound_gamma, bound_noise_warp, annealing, method_compute_warp,\
        model_type, recursive_warp, warp_updating, inducing_points, estimation_limit, free_deg_MNIV = self.get_default_options()
    return _alignment.configured(self).create_warper(self.x_basis_warp[0], self.noise_warp, bound_noise_warp, recursive=recursive_warp,
                          cuda=self.cuda, mode=self.mode_warp)


def refill(self, resp, respPair, startStateCount, transStateCount, q, q_lat, snr):
    """ Redistribute the responsibility for not letting empty clusters and generate a new model if needed.
    """
    resp_per_group = torch.sum(resp, axis=0)
    if self.verbose:
        print("Group responsibility estimated: " + str(resp_per_group.detach().cpu().numpy().astype(np.int64)),
              flush=True)
    if self.hard_assignments:
        occupied = resp_per_group >= 1.0
    else:
        occupied = torch.bincount(torch.argmax(resp, dim=1), minlength=resp.shape[1]) > 0
    if torch.any(~occupied[:-1]):
        # Case where last is filled and exists an empty group
        if occupied[-1]:
            resp, respPair = self.refill_resp(resp, respPair,
                                             occupied=None if self.hard_assignments else occupied)
        else:
            if self.verbose:
                print("Empty group detected, new iteration.\n")
            return resp, respPair, q, q_lat, snr, True
    return resp, respPair, q, q_lat, snr, False


def reorder(self, resp, respPair, q, q_lat):
    """ Reorder the responsibility to have a sorted assignation.
    """
    resp_per_group = torch.sum(resp, axis=0)
    reorder = torch.argsort(resp_per_group, descending=True)
    resp = resp[:, reorder]
    respPair = respPair[:, reorder, :]
    respPair = respPair[:, :, reorder]
    q = q[:, reorder]
    q_lat = q_lat[:, reorder]
    M = self.M
    gpmodels_temp = [[] * M] * self.n_outputs
    wp_sys = [[] * M] * self.n_outputs
    for ld in range(self.n_outputs):
        for i in range(M):
            gpmodels_temp[ld].append(self.gpmodels[ld][reorder[i]])
            wp_sys[ld].append(self.wp_sys[ld][reorder[i]])
    self.gpmodels = gpmodels_temp
    self.wp_sys = wp_sys
    return resp, respPair, q, q_lat, reorder


def new_group(self, resp, respPair, q, q_lat, snr):
    M = resp.shape[1]
    n_samples = self.T
    resp_ = self.cond_cuda(torch.zeros((n_samples, M + 1)))
    resp_[:, :-1] = resp
    resp = resp_
    respPair_ = self.cond_cuda(torch.zeros((n_samples, M + 1, M + 1)))
    respPair_[:, :-1, :-1] = respPair
    respPair = respPair_
    q_ = self.cond_cuda(torch.zeros((n_samples, M + 1, self.n_outputs)))
    q_[:, :-1, :] = q
    q = q_
    q_lat_ = self.cond_cuda(torch.zeros((n_samples, M + 1, self.n_outputs)))
    q_lat_[:, :-1, :] = q_lat
    q_lat = q_lat_
    snr_ = self.cond_cuda(
        torch.zeros((n_samples, M + 1, self.n_outputs)) - np.abs(torch.min(snr, dim=1)[0])[:, np.newaxis] * 2.0)
    snr_[:, :-1, :] = snr
    snr = snr_
    return resp, respPair, q, q_lat, snr


def remove_last_group(self, resp, respPair, q, q_lat, snr):
    resp = resp[:, :-1]
    respPair = respPair[:, :-1, :-1]
    q = q[:, :-1, :]
    q_lat = q_lat[:, :-1, :]
    snr = snr[:, :-1, :]
    return resp, respPair, q, q_lat, snr


def redefine_default(self, x_trains, y_trains, resp):
    """ Method to compute Sigma and Gamma from a batch of examples and assign it to initial values.
    """
    if self.verbose:
        print("Redefining default LDS priors.")
    if self.estimation_limit_def is None:
        n_f = y_trains.shape[0] - 1
    else:
        n_f = min(self.estimation_limit_def, y_trains.shape[0] - 1) if (
                self.estimation_limit != np.PINF) else y_trains.shape[0] - 1

    samples = y_trains[:n_f][:, :10, 0].T
    samples_ = y_trains[1:n_f + 1][:, :10, 0].T
    var_y_y = torch.median(torch.diag(torch.linalg.multi_dot([(samples - torch.mean(samples, dim=1)[:, np.newaxis]),
                                                              (samples - torch.mean(samples, dim=1)[:,
                                                                         np.newaxis]).T])) / n_f)  # * 0.15
    var_y_y_ = torch.median(
        torch.diag(torch.linalg.multi_dot([(samples_ - samples), (samples_ - samples).T])) / n_f)  # * 0.15
    kernel, ini_sigma, ini_gamma, ini_outputscale, bound_sigma, bound_gamma, bound_noise_warp, annealing, method_compute_warp,\
        model_type, recursive_warp, warp_updating, inducing_points, estimation_limit, free_deg_MNIV = self.get_default_options()

    ini_Sigma = var_y_y * 0.02
    ini_Gamma = var_y_y_ * 0.025

    bound_sigma = (ini_Sigma.item() * 1e-5, ini_Sigma.item() * 2.0)
    bound_gamma = (ini_Gamma.item() * 1e-5, ini_Gamma.item() * 2.0)

    if self.verbose:
        print("-----------Reestimated ------------", flush=True)
        print("Sigma: ", ini_Sigma)
        print("Gamma: ", ini_Gamma)
        print("-----------------------------", flush=True)
    kernel = (ConstantKernel(ini_outputscale, (ini_outputscale, ini_outputscale * 5.0)) *
              RBF(self.ini_lengthscale[0], self.bound_lengthscale[0]) + WhiteKernel(bound_sigma[0], bound_sigma))
    self.set_default_options(kernel, ini_Sigma, ini_Gamma, ini_outputscale, bound_sigma, bound_gamma,
                             bound_noise_warp, annealing, method_compute_warp,
                             model_type, recursive_warp, warp_updating, inducing_points, estimation_limit,
                             free_deg_MNIV)
    for ld in range(self.n_outputs):
        for m in range(len(self.gpmodels[ld])):
            self.gpmodels[ld][m] = self.create_gp_default()


def update_initial_sigma(self):
    """ Method to iteratively update initial sigma (not works so well)
    """
    estimator_sig, estimator_gam = 0.0, 0.0
    coef = 0.9
    for ld in range(self.n_outputs):
        for gp in self.gpmodels[ld]:
            estimator_sig = estimator_sig + torch.mean(torch.diag(gp.Sigma[-1]))
            estimator_gam = estimator_gam + torch.mean(torch.diag(gp.Gamma[-1]))
    estimator_sig = estimator_sig / (self.M * self.n_outputs)
    estimator_gam = estimator_gam / (self.M * self.n_outputs)
    self.ini_sigma_def = self.ini_sigma_def * coef + estimator_sig * (1 - coef)
    self.ini_gamma_def = self.ini_gamma_def * coef + estimator_gam * (1 - coef)
    for ld in range(self.n_outputs):
        for gp in self.gpmodels[ld]:
            gp.Sigma_def = gp.Sigma_def * coef + torch.eye(gp.Sigma_def.shape[0]) * estimator_sig * (1 - coef)
            gp.Gamma_def = gp.Gamma_def * coef + torch.eye(gp.Gamma_def.shape[0]) * estimator_gam * (1 - coef)


def return_model_of_sample(self, n_sample):
    for i, g in enumerate(self.gpmodels[0]):
        if n_sample in g.indexes:
            return i


def selected_gpmodels(self):
    selgp = 0
    for gp in self.gpmodels[0]:
        if len(gp.indexes) > 0:
            selgp = selgp + 1
    return list(range(selgp))


def gpmodel_deepcopy(self, gpmodel):
    gp_ = gp.GaussianProcessEmission(gpmodel.gp.kernel.clone_with_theta(gpmodel.gp.kernel.theta), gpmodel.x_basis.clone(),
                       annealing=gpmodel.annealing, bayesian=gpmodel.bayesian,
                       cuda=gpmodel.cuda, inducing_points=gpmodel.inducing_points,
                       estimation_limit=gpmodel.estimation_limit,
                       free_deg_MNIV=gpmodel.free_deg_MNIV, verbose=self.verbose,
                       lds_update=getattr(gpmodel, 'lds_update', 'conjugate'),
                       lds_step_size=getattr(gpmodel, 'lds_step_size', 0.2))
    gp_.y_train = gpmodel.y_train.copy()
    gp_.x_train = gpmodel.x_train.copy()
    gp_.f_star = gpmodel.f_star.copy()
    gp_.f_star_sm = gpmodel.f_star_sm.copy()
    gp_.cov_f = gpmodel.cov_f.copy()
    gp_.cov_f_sm = gpmodel.cov_f_sm.copy()
    gp_.y_var = gpmodel.y_var.copy()
    gp_.var = gpmodel.var.copy()
    gp_.A = gpmodel.A.copy()
    gp_.Gamma = gpmodel.Gamma.copy()
    gp_.C = gpmodel.C.copy()
    gp_.Sigma = gpmodel.Sigma.copy()
    gp_.gp.assign_alpha_ini(gp_.Sigma[0], gp_.Gamma[0])
    gp_.likelihood = gpmodel.likelihood.copy()
    gp_.N = gpmodel.N
    gp_.indexes = gpmodel.indexes.copy()
    gp_.fitted = gpmodel.fitted
    gp_.ini_cov_def = gpmodel.ini_cov_def
    gp_.A_def, gp_.Gamma_def = gpmodel.A_def, gpmodel.Gamma_def
    gp_.C_def, gp_.Sigma_def = gpmodel.C_def, gpmodel.Sigma_def
    gp_.internal_params = deepcopy(gpmodel.internal_params)
    gp_.observation_params = deepcopy(gpmodel.observation_params)
    gp_.ini_kernel_theta = gpmodel.ini_kernel_theta
    gp_.free_deg_MNIV = gpmodel.free_deg_MNIV
    return gp_

