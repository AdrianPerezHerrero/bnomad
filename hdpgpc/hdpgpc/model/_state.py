#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 8 10:33:40 2022
@author: adrian.perez
"""

from hdpgpc.emissions import _gaussian_process as gp
from hdpgpc.alignment import GaussianProcessWarp
from ..alignment import _components as _alignment
from ..hdp._variational import HDPVariational
from hdpgpc.hdp._optimizer import find_optimum_multiple_tries, kvec

import numpy as np

np.random.seed(42)
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from scipy.special._ufuncs import psi as digamma
from scipy.special._ufuncs import gammaln
import pandas as pd
from torchmetrics.audio import SignalNoiseRatio
from tqdm import trange
import pickle as plk
import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)

from hdpgpc.numerics._normalization import max_shift_log_scores
from hdpgpc.hdp._entropy import state_entropy_terms, pair_entropy_terms


class ModelState:
    """Private mutable scientific state with directly bound component operations.

    Algorithm definitions live in their owning modules, not in this state class.
    Bindings are descriptors to those exact functions (no copied bodies/wrappers).
    The public model is HDPGPC; this class is not a compatibility constructor.
    """
    def __init__(self, config, *, schedule=None):
        from ..config._models import _initialization_options, resolved_lds_policies
        options = _initialization_options(config, schedule=schedule)
        x_basis = options['x_basis']
        M = options['M']
        n_outputs = options['n_outputs']
        x_basis_warp = options['x_basis_warp']
        kernels = options['kernels']
        model_type = options['model_type']
        ini_lengthscale = options['ini_lengthscale']
        bound_lengthscale = options['bound_lengthscale']
        ini_gamma = options['ini_gamma']
        ini_sigma = options['ini_sigma']
        ini_outputscale = options['ini_outputscale']
        bound_sigma = options['bound_sigma']
        bound_gamma = options['bound_gamma']
        bound_noise_warp = options['bound_noise_warp']
        reest_conditions = options['reest_conditions']
        noise_warp = options['noise_warp']
        recursive_warp = options['recursive_warp']
        warp_updating = options['warp_updating']
        method_compute_warp = options['method_compute_warp']
        mode_warp = options['mode_warp']
        verbose = options['verbose']
        annealing = options['annealing']
        hmm_switch = options['hmm_switch']
        max_models = options['max_models']
        batch = options['batch']
        check_var = options['check_var']
        bayesian_params = options['bayesian_params']
        cuda = options['cuda']
        inducing_points = options['inducing_points']
        estimation_limit = options['estimation_limit']
        reestimate_initial_params = options['reestimate_initial_params']
        n_explore_steps = options['n_explore_steps']
        self.offline_representer_selection = options['offline_representer_selection']
        free_deg_MNIV = options['free_deg_MNIV']
        self.lds_update = options['lds_update']
        self.lds_scoring = options['lds_scoring']
        self.lds_step_size = options['lds_step_size']
        self.observation_scope = options['scope']
        self.observation_C_initial = options['C']
        self.observation_Sigma_initial = options['Sigma']
        self.fixed_observation_grid = (config.observation.fixed_grid if
                                     config.observation.fixed_grid is not None else
                                     config.observation.pca_grid)
        self.fixed_observation_offset = (config.observation.fixed_offset if
                                     config.observation.fixed_offset is not None else
                                     config.observation.pca_mean)
        self.fixed_latent_covariance = (config.observation.fixed_latent_covariance if
                                     config.observation.fixed_latent_covariance is not None else
                                     config.observation.pca_variances)
        self.lds_policies = resolved_lds_policies(config)
        self.custom_lds_policies = any((config.observation.C_policy,
                                        config.observation.Sigma_policy,
                                        config.dynamics.A_policy,
                                        config.dynamics.Gamma_policy))
        if self.custom_lds_policies:
            c_mode = self.lds_policies['C'].mode
            s_mode = self.lds_policies['Sigma'].mode
            self.observation_scope = ('shared' if c_mode == 'shared' or s_mode.startswith('shared')
                                      else 'fixed' if c_mode == s_mode == 'fixed' else 'cluster')
        share_gp = options['share_gp']
        use_snr = options['use_snr']
        reduce_outputs = options['reduce_outputs']
        reduce_outputs_ratio = options['reduce_outputs_ratio']
        hdp_hyp = options['hdp_hyp']
        alignment = options['alignment']
        if M is None:
            M = 1
        self.M = M
        from ..config._verbosity import normalize_verbosity
        self.verbosity = normalize_verbosity(verbose)
        self.progress = self.verbosity in ('progress', 'debug')
        self.verbose = self.verbosity == 'debug'
        self.hard_assignments = options['hard_assignments']
        self.fit_gp = options['fit_gp']
        self.birth_lds_kl_per_dimension = options['birth_lds_kl_per_dimension']
        self.actual_state = 0
        self.cuda = cuda
        self.device = 'cuda' if self.cuda else 'cpu'
        self.n_outputs = n_outputs
        self.alignment = alignment

        # Default cases
        # Be careful at using []*M because this replicates the object.
        if not isinstance(x_basis, list):
            x_basis = [x_basis] * M
        if x_basis_warp is None:
            x_basis_warp = x_basis
        elif not isinstance(x_basis_warp, list):
            x_basis_warp = [x_basis_warp] * M
        if not isinstance(bound_sigma, list) and not isinstance(bound_sigma, np.ndarray):
            bound_sigma = [bound_sigma] * M
        if not isinstance(bound_gamma, list) and not isinstance(bound_gamma, np.ndarray):
            bound_gamma = [bound_gamma] * M
        if not isinstance(bound_noise_warp, list) and not isinstance(bound_noise_warp, np.ndarray):
            bound_noise_warp = [bound_noise_warp] * M
        if not isinstance(ini_lengthscale, list) and not isinstance(ini_lengthscale, np.ndarray):
            ini_lengthscale = [ini_lengthscale] * M
        if not isinstance(ini_outputscale, list) and not isinstance(ini_outputscale, np.ndarray):
            ini_outputscale = [ini_outputscale] * M
        if not isinstance(bound_lengthscale, list) and not isinstance(bound_lengthscale, np.ndarray):
            bound_lengthscale = [bound_lengthscale] * M
        if not isinstance(inducing_points, list) and not isinstance(inducing_points, np.ndarray):
            inducing_points = [inducing_points] * M
        if not isinstance(estimation_limit, list) and not isinstance(estimation_limit, np.ndarray):
            estimation_limit = [estimation_limit] * M
        if not isinstance(ini_gamma, list) and not isinstance(ini_gamma, np.ndarray):
            ini_gamma = [ini_gamma] * M
        if not isinstance(ini_sigma, list) and not isinstance(ini_sigma, np.ndarray):
            ini_sigma = [ini_sigma] * M
        if not isinstance(model_type, list) and not isinstance(model_type, np.ndarray):
            model_type = [model_type] * M
        if not isinstance(annealing, list) and not isinstance(annealing, np.ndarray):
            annealing = [annealing] * M
        if not isinstance(warp_updating, list) and not isinstance(warp_updating, np.ndarray):
            warp_updating = [warp_updating] * M
        if not isinstance(recursive_warp, list) and not isinstance(recursive_warp, np.ndarray):
            recursive_warp = [recursive_warp] * M
        if ini_outputscale[0] is None:
            ini_outputscale = ini_sigma.copy()
        if kernels is None:
            kernels = []
            # WhiteNoise always last for stability.
            for m in range(M):
                # Outputscale should be bounded if a learning with few t basis will be performed
                kernels.append(ConstantKernel(ini_outputscale[m], (ini_outputscale[m], ini_outputscale[m] * 5.0)) * RBF(
                    ini_lengthscale[m], bound_lengthscale[m])
                               + WhiteKernel(bound_sigma[m][0], bound_sigma[m]))
                # kernels.append(RBF(ini_lengthscale[m], bound_lengthscale[m])
                #                + WhiteKernel(ini_sigma[m], bound_sigma[m]))
        # Save default options to generate new model
        self.set_default_options(kernels[0], ini_sigma[0], ini_gamma[0], ini_outputscale[0], bound_sigma[0],
                                 bound_gamma[0], bound_noise_warp[0],
                                 annealing[0], method_compute_warp, model_type[0], recursive_warp[0], warp_updating[0],
                                 inducing_points[0],
                                 estimation_limit[0], free_deg_MNIV)
        # Define some characteristics of the model with an initial M decided
        self.ini_lengthscale = ini_lengthscale
        self.bound_lengthscale = bound_lengthscale
        self.share_gp = share_gp
        # self.static_factor = ini_sigma[0] / (ini_sigma[0] + ini_gamma[0])
        # self.dynamic_factor = ini_gamma[0] / (ini_sigma[0] + ini_gamma[0])
        self.static_factor = ini_sigma[0] / ini_sigma[0]
        self.dynamic_factor = ini_gamma[0] / ini_gamma[0]
        self.bound_sigma = bound_sigma
        self.bound_gamma = bound_gamma
        self.bound_sigma_warp = bound_noise_warp
        self.annealing = annealing
        self.hmm_switch = hmm_switch
        self.method_compute_warp = method_compute_warp
        self.recursive_warp = recursive_warp
        self.model_type = model_type
        self.warp_updating = warp_updating
        self.max_models = max_models
        self.batch = batch
        self.use_snr = use_snr
        self.reduce_outputs = reduce_outputs
        self.reduce_outputs_ratio = reduce_outputs_ratio
        self.check_var = check_var
        self.bayesian_params = bayesian_params
        self.x_basis_warp = x_basis_warp
        self.inducing_points = inducing_points
        self.estimation_limit = estimation_limit
        self.reestimate_initial_params = reestimate_initial_params
        self.n_explore_steps = n_explore_steps
        self.free_deg_MNIV = free_deg_MNIV
        self.train_elbo = []
        self.resp_assigned = []
        self.f_ind_old = torch.zeros(M, device=self.device).long()
        # Define reestimation conditions
        self.min_samples = reest_conditions[0]
        self.max_samples = reest_conditions[1]
        self.div_samples = reest_conditions[2]
        # Define the parameters associated with samples and warp
        self.T = 0
        self.y = []
        self.y_w = []
        self.x_w = []
        # self.x_warped = []
        self.liks = []
        self.noise_warp = noise_warp
        self.mode_warp = mode_warp
        self.wp_sys = [[] for _ in range(n_outputs)]
        for ld in range(self.n_outputs):
            for m in range(M):
                self.wp_sys[ld].append(_alignment.configured(self).create_warper(x_basis_warp[m], noise_warp, bound_noise_warp[m],
                                                      recursive=recursive_warp[m], bayesian=self.bayesian_params,
                                                      cuda=self.cuda, mode=self.mode_warp))

        # Define the parameters of the state model:
        self.x_basis = x_basis
        self.x_basis_ini = x_basis[0].copy()
        self.x_train = []
        self.y_train = torch.tensor([])

        # Initial vector following a uniform distribution in log form
        # self.pi = [np.log(np.repeat(1 / M, M))]

        # Uniform matrix describing the initial transitions between states.
        self.trans_A = [np.log(np.matrix([[1 / M] * M] * M))]

        # Messages associated with state-space model
        # self.alpha = []
        # self.beta = []

        # Variational local parameters
        self.h = []
        self.q = []
        self.fmsg = None
        self.margPrObs = None

        # Variational global parameters
        self.rho = []
        self.omega = []
        self.theta = []
        self.transTheta = []
        self.startTheta = []

        # Warp caches (initialized lazily too, but better explicit)
        self._warp_cache_state = None  # (N, n_outputs) int64: last state_idx used
        self._warp_cache_ref = None  # (N, n_outputs) int64: last f_ind_old[state_idx] used
        self._y_trains_w_cache = None  # (N, T, n_outputs) warped y
        # self.x_w and self.liks_w already exist in your class; we reuse them as caches too.

        self.hdp.set_hyperparameters(hdp_hyp)

        # Model associated with each state
        # Default hyperparameters of GP defining the model.
        self.gpmodels = [[] for _ in range(n_outputs)]
        for ld in range(n_outputs):
            for m in range(M):
                # Create GP
                if self.bayesian_params:
                    gp_ = gp.GaussianProcessEmission(kernels[m], x_basis[m], annealing=self.annealing[m],
                                       bayesian=self.bayesian_params, cuda=self.cuda,
                                       fit_gp=self.fit_gp,
                                       inducing_points=inducing_points[m],
                                       estimation_limit=estimation_limit[m], free_deg_MNIV=self.free_deg_MNIV,
                                       verbose=self.verbose, lds_update=self.lds_update,
                                       lds_scoring=self.lds_scoring,
                                       lds_step_size=self.lds_step_size)
                else:
                    gp_ = gp.GaussianProcessEmission(kernels[m], x_basis[m], annealing=self.annealing[m], cuda=self.cuda,
                                       fit_gp=self.fit_gp,
                                       inducing_points=inducing_points[m], estimation_limit=estimation_limit[m],
                                       free_deg_MNIV=self.free_deg_MNIV, verbose=self.verbose,
                                       lds_update=self.lds_update, lds_scoring=self.lds_scoring,
                                       lds_step_size=self.lds_step_size)
                # Initiate GP
                if model_type[m] == 'static':
                    cond = gp_.GPR_static(ini_sigma[m])
                elif model_type[m] == 'dynamic':
                    cond = gp_.GPR_dynamic(ini_gamma[m], ini_sigma[m])
                else:
                    print("Chosen model should be dynamic or static.")
                self.initialize_lds_emission(gp_, cond)
                self.gpmodels[ld].append(gp_)

        self.observation_policies = []
        if self.observation_scope != 'cluster':
            from ..dynamics._observation import ObservationPolicy
            for ld in range(n_outputs):
                template = self.gpmodels[ld][0]
                initial_C = (self.observation_C_initial if self.observation_C_initial is not None
                             else template.C[0])
                initial_Sigma = (self.observation_Sigma_initial if self.observation_Sigma_initial is not None
                                 else template.Sigma[0])
                if self.custom_lds_policies:
                    policy = ObservationPolicy(self.observation_scope, initial_C, initial_Sigma,
                                               self.free_deg_MNIV,
                                               self.lds_policies['C'], self.lds_policies['Sigma'])
                else:
                    policy = ObservationPolicy(self.observation_scope, initial_C, initial_Sigma,
                                               self.free_deg_MNIV)
                self.observation_policies.append(policy)
                for emission in self.gpmodels[ld]:
                    policy.attach(emission)

        self.transition_policies = []
        if (self.lds_policies['A'].mode == 'shared'
                or self.lds_policies['Gamma'].mode.startswith('shared')):
            from ..dynamics._transition import TransitionPolicy
            for ld in range(n_outputs):
                template = self.gpmodels[ld][0]
                owner = TransitionPolicy(template.A[0], template.Gamma[0],
                                         self.free_deg_MNIV, self.lds_policies['A'],
                                         self.lds_policies['Gamma'])
                self.transition_policies.append(owner)
                for emission in self.gpmodels[ld]:
                    owner.attach(emission)

        self.init_global_params(len(x_basis), self.M)
        if self.cuda:
            self.full_model_to_torch()
            self.full_model_to_cuda()

    from ..model._clusters import (
        keep_last_all, set_default_options, get_default_options, create_gp_default, create_wp_sys_default, refill, reorder, new_group, remove_last_group, redefine_default, update_initial_sigma, return_model_of_sample, selected_gpmodels, gpmodel_deepcopy
    )

    def initialize_lds_emission(self, emission, initial):
        if self.custom_lds_policies:
            emission.lds_pair_policies = {
                'transition': (self.lds_policies['A'], self.lds_policies['Gamma']),
                'observation': (self.lds_policies['C'], self.lds_policies['Sigma'])}
            initial = list(initial)
            for index, name in enumerate(('A', 'Gamma', 'C', 'Sigma')):
                value = self.lds_policies[name].value
                if value is not None:
                    initial[index] = torch.as_tensor(value, dtype=torch.float64)
                elif name in ('A', 'C') and self.lds_policies[name].mode == 'fixed':
                    initial[index] = torch.eye(initial[index].shape[0], dtype=torch.float64)
        initial_covariance = None
        if self.fixed_observation_grid is not None:
            from ..emissions._fixed_observation import AffineGaussianObservation
            latent_covariance = np.asarray(self.fixed_latent_covariance)
            if latent_covariance.ndim == 1:
                latent_covariance = np.diag(latent_covariance)
            if not hasattr(self, 'fixed_observation_model'):
                self.fixed_observation_model = AffineGaussianObservation(
                    torch.as_tensor(initial[2]).detach().cpu().numpy(),
                    torch.as_tensor(initial[3]).detach().cpu().numpy(),
                    np.asarray(self.fixed_observation_grid),
                    np.asarray(self.fixed_observation_offset),
                    latent_covariance)
            emission.fixed_observation_model = self.fixed_observation_model
            emission.gp.fixed_observation_model = self.fixed_observation_model
            emission.direct_observation_grid = torch.as_tensor(
                self.fixed_observation_grid, dtype=torch.float64)
            emission.direct_observation_offset = torch.as_tensor(
                self.fixed_observation_offset, dtype=torch.float64).reshape(-1, 1)
            emission.gp.direct_observation_grid = emission.direct_observation_grid
            emission.gp.direct_observation_offset = emission.direct_observation_offset
            initial_covariance = torch.as_tensor(latent_covariance, dtype=torch.float64)
            emission.gp.direct_initial_covariance = initial_covariance
        emission.initial_conditions(ini_mean=None, ini_cov=initial_covariance,
                                    ini_A=initial[0], ini_Gamma=initial[1],
                                    ini_C=initial[2], ini_Sigma=initial[3])

    def update_shared_observation(self, responsibilities, *, online=False):
        if self.observation_scope == 'shared':
            for ld, policy in enumerate(self.observation_policies):
                policy.update(self.gpmodels[ld], responsibilities, online=online,
                              current_index=self.T - 1 if online else None)
        for ld, policy in enumerate(self.transition_policies):
            policy.update(self.gpmodels[ld], responsibilities, online=online,
                          current_index=self.T - 1 if online else None)

    from ..inference._messages import (
        _safe_exp, variational_local_terms, LogLik, forward, backward, coupled_state_coef, sum_coupled_state_coef, compute_h, baum_welch, normalize_log
    )

    from ..emissions._weights import (
        signaltonoise, rolling_snr, weight_mean, reduce_num_outputs, compute_snr_ini, compute_snr, normalize_snr, compute_joint_xy_q
    )

    from ..inference._scoring import (
        compute_q_elbo, full_LDS_elbo, estimate_new, compute_q
    )

    from ..model._conversion import (
        cond_to_torch, cond_to_numpy, cond_cuda, cond_cpu, cond_detach, full_model_to_numpy, full_model_to_torch, full_model_to_cuda, full_model_to_cpu, save_swgp
    )

    from ..inference._online import (
        _init_resp_cache_cold_start, _ensure_fast_cache_online, _update_q_lat_tail, _append_hard_step
    )

    from ..inference._offline import (
        variational_local_terms_batch, estimate_q_first, estimate_q_all, classify_batch, fit_labels
    )

    @property
    def hdp(self):
        """HDP variational component over the single native state store."""
        return HDPVariational(self)


    def reset_warp_cache(self):
        return _alignment.configured(self).reset_warp_cache(self)


    def init_global_params(self, d_dim, M):
        return self.hdp.init_global_params(d_dim, M)


    def reinit_global_params(self, M, transStateCount_, startStateCount_):
        return self.hdp.reinit_global_params(M, transStateCount_, startStateCount_)


    def temp_reinit_global_params(self, M, transStateCount_, startStateCount_, rho=None, omega=None):
        return self.hdp.temp_reinit_global_params(M, transStateCount_, startStateCount_, rho, omega)


    def create_initrho(self, M):
        return self.hdp.create_initrho(M)


    def _calcThetaPost(self, transStateCount, startStateCount, M, rho):
        return self.hdp._calcThetaPost(transStateCount, startStateCount, M, rho)


    def _calcThetaFull(self, transStateCount, startStateCount, M=None, rho=None, kappa=None):
        return self.hdp._calcThetaFull(transStateCount, startStateCount, M, rho, kappa)


    def compute_Pi(self):
        """ Compute transition matrix
        """
        return self.hdp.compute_Pi()


    def rho2beta(self, rho, returnSize='K+1'):
        return self.hdp.rho2beta(rho, returnSize)


    def beta2rho(self, beta, K):
        ''' Get rho vector that can deterministically produce provided beta.

        Returns
        ------
        rho : 1D array, size K
            Each entry rho[k] >= 0.
        '''
        return self.hdp.beta2rho(beta, K)


    def compute_warp_actual_state(self, x_trains, y_trains, q=None, q_lat=None, snr=None):
        """ Compute warp for every sample with an assignation to a gpmodel in the actual state of the model.
        """
        return _alignment.configured(self).compute_warp_actual_state(self, x_trains, y_trains, q, q_lat, snr)


    def compute_warp_actual_state_amtgp(self, x_trains, y_trains, q=None, q_lat=None, snr=None):
        """ Compute warp for every sample with an assignation to a gpmodel in the actual state of the model.
        """
        return _alignment.configured(self).compute_warp_actual_state_amtgp(self, x_trains, y_trains, q, q_lat, snr)


    def elbo_Linears(self, resp, respPair, post=False, one_sample=False):
        """ Compute ELBO for linear terms. HDP.
        """
        return self.hdp.elbo_Linears(resp, respPair, post, one_sample)


    def refill_resp(self, resp, respPair=None, occupied=None):
        """ Refill HDP parameters if a model is reasignated.
        """
        return self.hdp.refill_resp(resp, respPair, occupied=occupied)


    def calcELBO_LinearTerms(self, rho, omega, alpha, startAlpha, kappa, gamma, transTheta, startTheta, startStateCount,
                             transStateCount):
        """ Method to compute HPD linear terms.
        """
        return self.hdp.calcELBO_LinearTerms(rho, omega, alpha, startAlpha, kappa, gamma, transTheta, startTheta, startStateCount, transStateCount)


    def calcELBO_NonlinearTerms(self, resp, respPair):
        """ Method to compute H[q].
        """
        return self.hdp.calcELBO_NonlinearTerms(resp, respPair)


    def calc_Hstart(self, resp, eps=1e-30):
        return HDPVariational.calc_Hstart(self, resp, eps)


    def calc_Htable(self, respPair, eps=1e-30):
        return HDPVariational.calc_Htable(self, respPair, eps)


    def L_top(self, rho, omega, alpha, startAlpha, kappa, gamma):
        return self.hdp.L_top(rho, omega, alpha, startAlpha, kappa, gamma)


    def c_Dir(self, AMat, arem=None):
        ''' Evaluate cumulant function of the Dir distribution
        When input is vectorized, we compute sum over all entries.
        Returns
        -------
        c : scalar real
        '''
        return HDPVariational.c_Dir(self, AMat, arem)


    def find_optimum_rhoOmega(self, startTheta=None, transTheta=None, rho=None, omega=None, M=None):
        ''' Performs numerical optimization of rho and omega for M-step update.

                Note that the optimizer forces rho to be in [EPS, 1-EPS] for
                the sake of numerical stability

                Returns
                -------
                rho : 1D array, size K
                omega : 1D array, size K
                Info : dict of information about optimization.
                '''
        return self.hdp.find_optimum_rhoOmega(startTheta, transTheta, rho, omega, M)


    def compute_warp_y(self, x_train, y, strategie='standard', force_model=None, gpmodel=None, i=None, ld=0):
        """
        Computation of warps following some strategies and functions proposed on warp_models

        Parameters
        ----------
        x_train : Array
            Domain of new observations.
        y : Array
            New observations vector.
        strategie : String, optional
            Describes the strategie to compute the warps of the model. The default is 'standard'.

        Returns
        -------
        y_w : Array.
            Vector describing warped sample respect to the models.
        x_w : Array.
            Vector describing the applied warp.
        liks : Double
            Likelihood of the array assuming some hyperparameters of noise of the GP warp.

        """
        return _alignment.configured(self).compute_warp_y(self, x_train, y, strategie, force_model, gpmodel, i, ld)


    def warp_batch_by_resp_amtgp(
            self,
            x_trains: torch.Tensor,
            y_trains: torch.Tensor,
            resp_temp: torch.Tensor,  # (N,M)
            gpmodels=None,
            f_ind_old: torch.Tensor = None,
            train_iter: int = 50,
            batch_size: int = 128,
    ):
        return _alignment.configured(self).warp_batch_by_resp_amtgp(self, x_trains, y_trains, resp_temp, gpmodels, f_ind_old, train_iter, batch_size)


    def warp_batch_by_resp_amtgp_cached(
            self,
            x_trains,
            y_trains,
            resp_temp,  # (N,M)
            f_ind_old=None,
            train_iter: int = 50,
            batch_size: int = 128,
    ):

        return _alignment.configured(self).warp_batch_by_resp_amtgp_cached(self, x_trains, y_trains, resp_temp, f_ind_old, train_iter, batch_size)


    def select_assigned_warp(self, y_trains_w, resp):
        # y_trains_w: (N,T,D,M)
        return _alignment.configured(self).select_assigned_warp(self, y_trains_w, resp)


    def compute_trans_A(self, M):
        return self.hdp.compute_trans_A(M)


    def compute_trans_pi(self, M, pi):
        return HDPVariational.compute_trans_pi(self, M, pi)


    @property
    def online(self):
        from ..inference import OnlineInference
        return OnlineInference(self)

    @property
    def offline(self):
        from ..inference import OfflineInference
        return OfflineInference(self)
