#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native cluster/output GP emission and coupled LDS trajectory.
Prior/prediction operations live in _prior; fitting in _training; LDS in dynamics.
Original algorithms by Adrian Perez Herrero, retained without numerical changes."""
from ._prior import GaussianProcessPrior
import numpy as np
import torch
from tqdm import trange
import math
from bisect import bisect_right
from ..dynamics._models import _LINEAR, _STATIC
from ..dynamics import _trajectory
from ..dynamics._distributions import matrix_normal_inv_wishart, inv_wishart
dtype = torch.float64
torch.set_default_dtype(dtype)

class GaussianProcessEmission:
    """Model that sum up all the information to characterise a GPR_iterative_model.
        Parameters
        ----------
        kernel : kernel from sklearn class for covariance computation.

        x_basis : array-like of shape (s_samples) domain points where is wanted
        to focus the learning.

        ini_Sigma : positive double or array-like of shape (n_samples) noise associated with subyacent process.
        Returns
        -------
        self : returns an instance of self.
        """

    def __init__(self, kernel, x_basis, annealing=True, bayesian=False, cuda=False, inducing_points=False,
                 estimation_limit=None, free_deg_MNIV=5, verbose=True,
                 lds_update='conjugate_sequential', lds_step_size=0.2,
                 lds_scoring='final', fit_gp=True):
        self.gp = GaussianProcessPrior(kernel, x_basis, cuda=cuda, verbose=verbose)
        self.x_basis = self.cond_to_torch(x_basis)
        self.x_train = []
        self.y_train = []
        self.f_star = []
        self.f_star_sm = []
        self.cov_f = []
        self.cov_f_sm = []
        self.y_var = []
        self.var = []
        self.A = []
        self.Gamma = []
        self.C = []
        self.Sigma = []
        self.likelihood = []
        self.N = 0
        self.indexes = []
        self.K = self.cond_to_torch(kernel(self.cond_to_cpu(x_basis), self.cond_to_cpu(x_basis)))
        self.annealing = annealing
        self.bayesian = bayesian
        self.cuda = cuda
        self.inducing_points = inducing_points
        if bayesian:
            self.internal_params = None
            self.observation_params = None
        if estimation_limit is None:
            estimation_limit = np.inf
        self.estimation_limit = estimation_limit
        self.fit_gp = fit_gp
        self.fitted = not fit_gp
        self.gp.fitted = not fit_gp
        self.A_def, self.Gamma_def, self.C_def, self.Sigma_def = None, None, None, None
        self.ini_cov_def = None
        self.ini_kernel_theta = self.gp.kernel.theta

        if self.cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.sq_lat_last = torch.zeros(1, device=self.device)
        self.free_deg_MNIV = free_deg_MNIV
        self.lds_update = lds_update
        self.lds_scoring = lds_scoring
        self.lds_step_size = lds_step_size
        self.verbose = verbose
        self.disable = not verbose

    def _eye_cached(self, n, device, dtype):
        if not hasattr(self, "_eye_cache"):
            self._eye_cache = {}
        key = (n, str(device), dtype)
        if key not in self._eye_cache:
            self._eye_cache[key] = torch.eye(n, device=device, dtype=dtype)
        return self._eye_cache[key]

    def _chol_spd(self, M, jitter_scale=1e-8, *, context='covariance'):
        M = 0.5 * (M + M.T)
        eye = self._eye_cached(M.shape[0], M.device, M.dtype)
        diag_mean = torch.mean(torch.diag(M).abs()).clamp_min(torch.finfo(M.dtype).eps)
        try:
            # Preserve the usual numerical path unless factorization fails.
            return torch.linalg.cholesky(M + jitter_scale * diag_mean * eye)
        except torch.linalg.LinAlgError as error:
            last_error = error

        # A nearly singular covariance may need more than the usual jitter.
        # Use the matrix scale when the diagonal alone is unusually small.
        scale = torch.maximum(diag_mean, M.abs().amax())
        for multiplier in (10.0, 1e2, 1e3, 1e4, 1e5, 1e6):
            relative_jitter = jitter_scale * multiplier
            try:
                factor = torch.linalg.cholesky(M + relative_jitter * scale * eye)
            except torch.linalg.LinAlgError as error:
                last_error = error
                continue
            recoveries = getattr(self, '_chol_jitter_recoveries', 0) + 1
            self._chol_jitter_recoveries = recoveries
            if self.verbose and (recoveries & (recoveries - 1)) == 0:
                print(f'Cholesky recovered for {context} with relative jitter '
                      f'{relative_jitter:.0e} (recoveries={recoveries})', flush=True)
            return factor

        raise torch.linalg.LinAlgError(
            f'Cholesky failed for {context} after relative jitter up to '
            f'{jitter_scale * 1e6:.0e}; covariance may be indefinite or non-finite'
        ) from last_error

    def _log2pi(self, ref):
        return torch.tensor(math.log(2.0 * math.pi), device=ref.device, dtype=ref.dtype)

    def _observation_latent_cov(self, x_post, latent_cov, C):
        """Project C Var[f] C.T onto the grid used by an observation score."""
        latent_cov = torch.as_tensor(latent_cov, dtype=C.dtype, device=C.device)
        projected = C @ latent_cov @ C.T
        x_post = self.cond_to_torch(x_post).to(self.x_basis.device)
        if getattr(self, 'direct_observation_offset', None) is not None:
            if not torch.equal(x_post.reshape(-1, 1), self.direct_observation_grid):
                raise ValueError('PCA emission requires its calibrated observation grid')
            return projected
        if torch.equal(x_post, self.x_basis):
            return projected

        x_basis_np = self.x_basis.detach().cpu().numpy()
        x_post_np = x_post.detach().cpu().numpy()
        K_ff = torch.as_tensor(self.gp.kernel(x_basis_np, x_basis_np),
                               dtype=C.dtype, device=C.device)
        K_fp = torch.as_tensor(self.gp.kernel(x_basis_np, x_post_np),
                               dtype=C.dtype, device=C.device)
        projection = torch.cholesky_solve(K_fp, self._chol_spd(K_ff)).T
        return projection @ projected @ projection.T

    def _observation_params(self, i=None, params=None):
        """Return the scoring state covariance and selected observation map."""
        if params is not None:
            return params[1], params[2]
        if i is None:
            return self.cov_f_sm[-1], self.C[-1]
        if not self.indexes:
            C = self.C[-1] if self.lds_scoring == 'final' else self.C[0]
            return self.cov_f_sm[0], C
        if i < 0 or len(self.indexes) <= i:
            return self.cov_f_sm[-1], self.C[-1]
        if self.lds_scoring == 'historical' and self.estimation_limit > i:
            return self.cov_f_sm[i], self.get_params(i)[2]
        return self.cov_f_sm[i], self.C[-1]

    def _gaussian_score_shared_cov(self, Y, mean, cov, latent_cov=None):
        """
        Expected Gaussian observation log density on a shared grid.

        Y    : (B,T,1) or (B,T)
        mean : (T,1) or (T,)
        cov  : (T,T) observation noise covariance
        latent_cov : (T,T) projected latent posterior covariance
        returns: (B,)
        """
        if Y.ndim == 3:
            Y2 = Y[..., 0].T  # (T,B)
        else:
            Y2 = Y.T  # (T,B)

        if mean.ndim == 2:
            m = mean[:, 0:1]  # (T,1)
        else:
            m = mean.unsqueeze(1)  # (T,1)

        diff = Y2 - m  # (T,B)
        L = self._chol_spd(cov)
        alpha = torch.cholesky_solve(diff, L)
        q = diff.shape[0]
        logdet = 2.0 * torch.log(torch.diag(L)).sum()
        trace = 0.0
        if latent_cov is not None:
            trace = torch.trace(torch.cholesky_solve(latent_cov, L))
        return -0.5 * (torch.sum(diff * alpha, dim=0) + trace
                       + logdet + q * self._log2pi(diff))

    def initial_conditions(self, ini_mean=None, ini_cov=None,
                           ini_A=None, ini_Gamma=None, ini_C=None, ini_Sigma=None):
        """
        Incorporate initial conditions to the model, if none is specified default
        model is a dynamic one.
        Parameters
        ----------
        ini_mean : array like
            DESCRIPTION. Initial mean of the model. The default is 0.
        ini_cov : matrix like
            DESCRIPTION. Initial covariance of the model, should be computed
            using hyperparameters kernel stimation. The default is None.
        ini_A : matrix like
            DESCRIPTION. Projection step matrix. The default is Id.
        ini_Gamma : matrix like
            DESCRIPTION. Noise associated with the subsequent process. The default is 0.01.
        ini_C : matrix like
            DESCRIPTION. Projection observations matrix. The default is Id.
        ini_Sigma : matrix like
            DESCRIPTION. Noise associated with observations. The default is 0.5**2.

        Returns
        -------
        None.

        """
        # Incorporate initial parameters
        if ini_mean is None:
            self.f_star.append(self.compute_mean())
            self.f_star_sm.append(self.compute_mean())
        else:
            self.f_star.append(self.cond_to_torch(ini_mean))
            self.f_star_sm.append(self.cond_to_torch(ini_mean))
        if ini_cov is None:
            self.cov_f.append(self.cond_to_torch(self.K))
            self.cov_f_sm.append(self.cond_to_torch(self.K))
            self.ini_cov_def = self.K.clone()
        else:
            self.cov_f.append(self.cond_to_torch(ini_cov))
            self.cov_f_sm.append(self.cond_to_torch(ini_cov))
            self.ini_cov_def = ini_cov
        if ini_A is None and ini_Gamma is None and ini_C is None and ini_Sigma is None:
            ini_A, ini_Gamma, ini_C, ini_Sigma = self.GPR_dynamic()

        self.A.append(ini_A)
        self.Gamma.append(ini_Gamma)
        self.C.append(ini_C)
        self.Sigma.append(ini_Sigma)
        self.A_def = self.cond_to_cuda(self.cond_to_torch(ini_A))
        self.Gamma_def = self.cond_to_cuda(self.cond_to_torch(ini_Gamma))
        self.C_def = self.cond_to_cuda(self.cond_to_torch(ini_C))
        self.Sigma_def = self.cond_to_cuda(self.cond_to_torch(ini_Sigma))
        if not self.fit_gp:
            self.gp.assign_alpha_ini(self.Sigma_def, self.Gamma_def)
        self.var.append(np.atleast_2d(np.diag(ini_Gamma)).T)
        self.y_var.append(np.atleast_2d(np.diag(ini_Sigma)).T)
        # DEGREES OF FREEDOM AT LEAST 3
        if self.bayesian:
            pair_policies = getattr(self, 'lds_pair_policies', None)
            if pair_policies is not None:
                from ..dynamics._parameter_posterior import make_pair_prior
                self.internal_params = make_pair_prior(
                    ini_A, ini_Gamma, self.free_deg_MNIV, *pair_policies['transition'])
                self.observation_params = make_pair_prior(
                    ini_C, ini_Sigma, self.free_deg_MNIV, *pair_policies['observation'])
            else:
                self.internal_params = matrix_normal_inv_wishart(ini_A, np.eye(ini_A.shape[0]),  self.free_deg_MNIV, ini_Gamma)
            if pair_policies is None and torch.all(ini_Gamma == torch.zeros(ini_Gamma.shape)):
                self.observation_params = inv_wishart(self.free_deg_MNIV, ini_Sigma, ini_C)
            elif pair_policies is None:
                self.observation_params = matrix_normal_inv_wishart(ini_C, np.eye(ini_C.shape[0]), self.free_deg_MNIV, ini_Sigma)
        if self.lds_scoring == 'historical':
            self.lds_internal_score_history = []
            self.lds_observation_score_history = []
            _trajectory.record_lds_parameter_score(self)


    def GPR_static(self, ini_Sigma=None):
        """ Define static conditions using initial Sigma as a double.
        """
        return _STATIC.initial_parameters(self.x_basis, sigma=ini_Sigma)

    def GPR_dynamic(self, gamma=None, sigma=None):
        """ Define dynamic conditions using initial Sigma as a double.
        """
        return _LINEAR.initial_parameters(self.x_basis, sigma=sigma, gamma=gamma)

    def fit_kernel_params(self, x_train, y, alpha_ini, gamma_ini, valid=True):
        """ Optimize RBF kernel hyperparameters
        """
        if not self.fit_gp:
            raise ValueError('GP hyperparameter fitting is disabled for this emission')
        alph_ = self.cond_to_torch(alpha_ini[0][0])
        gam_ = self.cond_to_torch(gamma_ini[0][0])
        if valid:
            fitted = self.gp.fit_torch(self.cond_to_torch(x_train), self.cond_to_torch(y), alph_, gam_,
                                       reduced_points=self.inducing_points, verbose=self.verbose)
        noise = self.gp.kernel.get_params()["k2__noise_level"]
        noise = alph_
        self.x_basis = self.cond_to_cuda(self.cond_to_torch(self.gp.x_basis))
        if getattr(self, 'lds_pair_policies', None) is None:
            self.Sigma[-1] = self.cond_to_cuda(self.cond_to_torch(torch.mul(noise, torch.eye(len(self.x_basis), device=noise.device))))
            self.Sigma_def = torch.clone(self.Sigma[-1])
            self.y_var[-1] = self.cond_to_cuda(self.cond_to_torch(np.atleast_2d(np.diag(self.cond_to_cpu(self.Sigma[-1]))).T))
            self.C[-1] = self.cond_to_cuda(torch.eye(len(self.x_basis)))
            self.A[-1] = self.cond_to_cuda(torch.eye(len(self.x_basis)))
            self.Gamma[-1] = self.cond_to_cuda(self.cond_to_torch(torch.mul(torch.mean(torch.diag(self.Gamma[-1])),
                                                                            torch.eye(len(self.x_basis), device=self.device))))
        self.f_star[-1] = self.cond_to_cuda(self.cond_to_torch(self.compute_mean()))
        self.f_star_sm[-1] = self.cond_to_cuda(self.cond_to_torch(self.compute_mean()))
        ini_cov = self.cond_to_cuda(self.cond_to_torch(
            self.gp.kernel(self.cond_to_numpy(self.cond_to_cpu(self.x_basis)),
                           self.cond_to_numpy(self.cond_to_cpu(self.x_basis)))))# + self.Sigma[-1]
        self.ini_cov_def = ini_cov
        self.cov_f[-1] = ini_cov
        self.cov_f_sm[-1] = ini_cov
        if self.bayesian and getattr(self, 'lds_pair_policies', None) is None:
            self.observation_params.set_scale(self.cond_to_cuda(self.cond_to_torch(torch.mul(noise, torch.eye(len(self.x_basis), device=noise.device)))))
            self.observation_params.m_mean = self.C[-1]
            self.internal_params.set_scale(self.Gamma[-1])
            self.internal_params.m_mean = self.A[-1]
        if self.lds_scoring == 'historical':
            _trajectory.record_lds_parameter_score(self, replace_last=True)
        self.fitted = True
        print("---Kernel estimated---")
        print(self.gp.kernel)
        return self.x_basis, ini_cov

    def log_lik_sample(self, y):
        """Returns log-likelihood of a single sample using the last state of the model.
        """
        lik_post = self.gp.log_likelihood(self.N, self.N, self.f_star_sm, self.cov_f_sm, self.A[-1], self.Gamma[-1], y,
                                          self.C[-1], self.Sigma[-1])
        return lik_post

    def log_sq_error(self, x_train, y, mean=None, cov=None, C=None, Sigma=None, i=None, proj=False, first=False):
        """Expected Gaussian observation log density at a model iteration."""
        y = self.cond_to_cuda(self.cond_to_torch(y)).to(torch.float64)
        mean = self.cond_to_torch(mean)
        cov = self.cond_to_torch(cov)
        C = self.cond_to_torch(C)
        Sigma = self.cond_to_torch(Sigma)

        if x_train is None:
            x_train = (self.direct_observation_grid
                       if getattr(self, 'direct_observation_offset', None) is not None
                       else self.x_basis)

        if mean is None:
            params = None
        else:
            params = [mean, cov, C, Sigma]

        if i is not None:
            f_star, cov_f = self.observe(x_train, i, params, proj=proj,
                                         final_lds=params is None and self.lds_scoring == 'final',
                                         smoothed=params is None)
        else:
            f_star, cov_f = self.step_forward_last(x_train, params)

        if first:
            score_sigma = (params[3] if params is not None else
                           self.Sigma[-1] if self.lds_scoring == 'final' else self.Sigma[0])
            ini_noise = torch.mean(torch.diag(score_sigma)) * 1e-2
            cov_f = cov_f + ini_noise * self._eye_cached(len(x_train), cov_f.device, cov_f.dtype)

        if y.ndim == 1:
            y = y.unsqueeze(-1)
        if f_star.ndim == 1:
            f_star = f_star.unsqueeze(-1)

        diff = y - f_star
        L = self._chol_spd(cov_f)
        alpha = torch.cholesky_solve(diff, L)
        q = diff.shape[0]
        latent_cov, C_mat = self._observation_params(i, params)
        output_latent_cov = self._observation_latent_cov(x_train, latent_cov, C_mat)
        trace = torch.trace(torch.cholesky_solve(output_latent_cov, L))
        logdet = 2.0 * torch.log(torch.diag(L)).sum()
        err = -0.5 * (torch.sum(diff * alpha) + trace
                      + logdet + q * self._log2pi(diff))
        return err

    def log_lat_error(self, i, h_ini):
        """Expected Gaussian state log density plus its chain entropy."""
        mean_cur = self.f_star_sm[i + 1].reshape(-1, 1)
        cov_cur = self.cov_f_sm[i + 1]
        if i == 0:
            # The filtering code starts at the GP prior, without an A/Gamma step.
            prior_mean = self.f_star_sm[0].reshape(-1, 1)
            prior_cov = self.cov_f_sm[0] * h_ini
            residual = mean_cur - prior_mean
            residual_cov = cov_cur
            L_cur = self._chol_spd(cov_cur)
            logdet_conditional = 2.0 * torch.log(torch.diag(L_cur)).sum()
            noise_cov = prior_cov
        else:
            mean_prev = self.f_star_sm[i].reshape(-1, 1)
            filtered_prev = self.cov_f[i]
            # The pair covariance and entropy describe the saved q trajectory
            # under its historical transition in either scoring mode.
            A_q = self.A[i] if i < len(self.A) else self.A[-1]
            Gamma_q = self.Gamma[i] if i < len(self.Gamma) else self.Gamma[-1]
            if self.lds_scoring == 'final':
                noise_cov, A = self.Gamma[-1], self.A[-1]
            else:
                noise_cov, A = Gamma_q, A_q
            predicted_cov = A_q @ filtered_prev @ A_q.T + Gamma_q
            gain = torch.cholesky_solve(A_q @ filtered_prev,
                                        self._chol_spd(predicted_cov)).T
            # Build the RTS pair posterior from positive covariance sums.
            # This avoids a near-singular forward Schur complement subtraction.
            eye = self._eye_cached(filtered_prev.shape[0], filtered_prev.device,
                                   filtered_prev.dtype)
            projection = eye - gain @ A_q
            reverse_conditional = (projection @ filtered_prev @ projection.T
                                   + gain @ Gamma_q @ gain.T)
            cov_prev_pair = reverse_conditional + gain @ cov_cur @ gain.T
            residual = mean_cur - A @ mean_prev
            residual_projection = eye - A @ gain
            residual_cov = (residual_projection @ cov_cur @ residual_projection.T
                            + A @ reverse_conditional @ A.T)
            L_cur = self._chol_spd(cov_cur)
            L_reverse = self._chol_spd(reverse_conditional,
                                       context='latent reverse conditional')
            L_prev = self._chol_spd(cov_prev_pair)
            logdet_conditional = 2.0 * (
                torch.log(torch.diag(L_cur)).sum()
                + torch.log(torch.diag(L_reverse)).sum()
                - torch.log(torch.diag(L_prev)).sum())

        L_noise = self._chol_spd(noise_cov)
        logdet_noise = 2.0 * torch.log(torch.diag(L_noise)).sum()
        quadratic = torch.sum(residual * torch.cholesky_solve(residual, L_noise))
        trace = torch.trace(torch.cholesky_solve(residual_cov, L_noise))
        dimension = residual.shape[0]
        expected_log_transition = -0.5 * (quadratic + trace + logdet_noise
                                          + dimension * self._log2pi(residual))
        # H[q(f_1)] or H[q(f_t | f_{t-1})] completes the Gaussian latent term.
        entropy = 0.5 * (dimension * (1.0 + self._log2pi(residual))
                         + logdet_conditional)
        return expected_log_transition + entropy

    def condition_observation(self, index, x_train, y, x_warped=None, h=1.0, posterior=True, embedding=True, include_index=False):
        """ Method to include sample in the model, compute the posterior and add the data.
        """
        if posterior:
            self.N = self.N + 1
            self.indexes.append(index)
            self.x_train.append(x_train)
            self.y_train.append(self.cond_to_torch(y))
            f_star_, cov_f_ = self.gp.posterior(self.f_star_sm[-1], self.cov_f_sm[-1], self.y_train[-1], self.A[-1],
                                                self.Gamma[-1], self.C[-1], self.Sigma[-1], x_train=x_train,
                                                x_warped=x_warped, embedding=embedding, h=h)
            self.f_star.append(f_star_)
            self.f_star_sm.append(f_star_)
            self.cov_f.append(cov_f_)
            self.cov_f_sm.append(cov_f_)
            return self.f_star_sm[-1], self.cov_f_sm[-1]
        else:
            if include_index:
                self.indexes.append(index)
                self.x_train.append(x_train)
                self.y_train.append(self.cond_to_torch(y))
                f_star_, cov_f_ = self.f_star_sm[-1], self.cov_f_sm[-1]
                self.f_star.append(f_star_)
                self.f_star_sm.append(f_star_)
                self.cov_f.append(cov_f_)
                self.cov_f_sm.append(cov_f_)
            return self.f_star_sm[-1], self.cov_f_sm[-1]

    def include_weighted_sample(self, index, x_train, x_warped, y, h, snr=None):
        """Method to include the sample depending on the responsibility h.
        """
        y = self.cond_to_cuda(self.cond_to_torch(y))
        x_train = self.cond_to_cuda(self.cond_to_torch(x_train))
        new_x_basis = self.x_basis
        if h > 0.0:
            if self.N == 0 and not self.fitted:
                if torch.allclose(torch.from_numpy(self.gp.kernel.theta), torch.from_numpy(self.ini_kernel_theta)):
                    new_x_basis, _ = self.fit_kernel_params(x_train, y, self.Sigma[-1], self.Gamma[-1], valid=True)
                else:
                    new_x_basis, _ = self.fit_kernel_params(x_train, y, self.Sigma[-1], self.Gamma[-1], valid=False)
            if snr is not None:
                if snr > 0.5:
                    self.condition_observation(index, x_train, y, x_warped, h=h)
                else:
                    self.condition_observation(index, x_train, y, x_warped, posterior=False, include_index=True)
            else:
                self.condition_observation(index, x_train, y, x_warped, h=h)
        else:
            self.condition_observation(index, x_train, y, x_warped, posterior=False)
        return new_x_basis

    def include_weighted_sample_flow(self, index, x_train, grid_samples, y, h):
        """Variational Gaussian state update averaged over sampled input warps.

        With q(g) fixed, E_q(g)[log p(y | f, g)] is quadratic in the basis
        state f.  Averaging its precision and linear natural parameter gives
        the optimal Gaussian coordinate update for this observation.
        """
        from ..alignment._gp_flow import _column, _kernel_torch

        if h <= 0:
            return self.x_basis
        device = self.f_star_sm[-1].device
        dtype = torch.float64
        grids = torch.as_tensor(grid_samples, dtype=dtype, device=device)
        if grids.ndim == 2:
            grids = grids.unsqueeze(-1)
        observation = _column(y, device=device)
        mean_grid = grids.mean(0)
        new_basis = self.x_basis
        if self.N == 0 and not self.fitted:
            valid = torch.allclose(torch.from_numpy(self.gp.kernel.theta),
                                   torch.from_numpy(self.ini_kernel_theta))
            new_basis, _ = self.fit_kernel_params(mean_grid, observation,
                                                   self.Sigma[-1], self.Gamma[-1],
                                                   valid=valid)

        basis = _column(self.x_basis, device=device)

        a = torch.as_tensor(self.A[-1], dtype=dtype, device=device)
        gamma = torch.as_tensor(self.Gamma[-1], dtype=dtype, device=device)
        c = torch.as_tensor(self.C[-1], dtype=dtype, device=device)
        sigma = torch.as_tensor(self.Sigma[-1], dtype=dtype, device=device)
        previous_mean = _column(self.f_star_sm[-1], device=device)
        previous_cov = torch.as_tensor(self.cov_f_sm[-1], dtype=dtype, device=device)
        first = self.N == 0
        prior_mean = previous_mean if first else a @ previous_mean
        prior_cov = previous_cov if first else a @ previous_cov @ a.T + gamma
        eye_basis = torch.eye(len(basis), dtype=dtype, device=device)
        prior_cov = 0.5 * (prior_cov + prior_cov.T) + 1e-6 * eye_basis
        prior_chol = torch.linalg.cholesky(prior_cov)
        prior_precision = torch.cholesky_inverse(prior_chol)

        kernel = self.gp.kernel
        kbb = _kernel_torch(kernel, basis, basis)
        kbb_chol = torch.linalg.cholesky(0.5 * (kbb + kbb.T) + 1e-4 * eye_basis)
        eye_y = torch.eye(len(observation), dtype=dtype, device=device)
        natural_precision = prior_precision.clone()
        natural_mean = prior_precision @ prior_mean
        weight = float(h) / len(grids)
        constant_noise = bool(torch.all(torch.isclose(
            torch.diagonal(sigma), torch.diagonal(sigma).mean())))

        for grid in grids:
            kbg = _kernel_torch(kernel, basis, grid)
            projection = torch.cholesky_solve(kbg, kbb_chol).T
            measurement = projection @ c
            if constant_noise:
                residual_cov = torch.diagonal(sigma).mean() * eye_y
            else:
                kgg = _kernel_torch(kernel, grid)
                residual_cov = (kgg - kbg.T @ projection.T
                                + projection @ sigma @ projection.T)
            residual_cov = 0.5 * (residual_cov + residual_cov.T) + 1e-6 * eye_y
            residual_chol = torch.linalg.cholesky(residual_cov)
            inv_measurement = torch.cholesky_solve(measurement, residual_chol)
            inv_observation = torch.cholesky_solve(observation, residual_chol)
            natural_precision = natural_precision + weight * measurement.T @ inv_measurement
            natural_mean = natural_mean + weight * measurement.T @ inv_observation

        posterior_chol = torch.linalg.cholesky(0.5 * (natural_precision + natural_precision.T))
        posterior_cov = torch.cholesky_inverse(posterior_chol)
        posterior_mean = posterior_cov @ natural_mean
        self.N += 1
        self.indexes.append(index)
        self.x_train.append(mean_grid.detach())
        self.y_train.append(observation.detach())
        self.f_star.append(posterior_mean.detach())
        self.f_star_sm.append(posterior_mean.detach())
        self.cov_f.append(posterior_cov.detach())
        self.cov_f_sm.append(posterior_cov.detach())
        return new_basis

    @staticmethod
    def _active_weighted_indices(resp):
        """Drop only a numerically negligible tail of fractional mass.

        This bounds omitted responsibility by 1e-8 of the cluster total,
        while retaining every one-hot assignment in the legacy path.
        """
        active = torch.nonzero(resp > 0, as_tuple=False).squeeze(1)
        if active.numel() == 0:
            return active
        weights = resp[active]
        if weights.sum() < 1e-8:
            return active[:0]
        if active.numel() == 1:
            return active
        order = torch.argsort(weights)
        sorted_indices = active[order]
        omitted = torch.cumsum(weights[order], dim=0)
        retained = sorted_indices[omitted > weights.sum() * 1e-8]
        return torch.sort(retained).values

    def full_pass_weighted_flow(self, x_trains, y_trains, resp, grid_samples,
                                q=None, q_lat=None, snr=None):
        """Offline pass using one MC warp posterior per observation."""
        if self.lds_update == 'conjugate_batch':
            raise NotImplementedError('conjugate_batch does not support GP-flow input warps')
        active = self._active_weighted_indices(resp)
        if active.numel() == 0:
            return q, q_lat
        dynamic = bool(torch.count_nonzero(self.Gamma[-1]))
        batch_lds = self.lds_update == 'conjugate_batch'
        for index in active.tolist():
            h = float(resp[index].item())
            self.include_weighted_sample_flow(index, x_trains[index],
                                               grid_samples[index],
                                               y_trains[index], h)
            if batch_lds:
                _trajectory.hold_lds_params(self)
            elif dynamic:
                self.backwards_pair(h)
                self.bayesian_new_params(h)
        if dynamic:
            self.backwards()
        if batch_lds:
            self.bayesian_batch_params(resp, grid_samples=grid_samples)
        mean_grids = torch.stack([samples.mean(0) for samples in grid_samples])
        return (self.compute_sq_err_all(mean_grids, y_trains),
                self.compute_q_lat_all(mean_grids))

    def full_pass_weighted(self, x_trains, y_trains, resp, q=None, q_lat=None, snr=None):
        """Full forward pass method used in the offline scheme."""
        if len(torch.nonzero(self.Gamma[-1])) < 1:
            model_type = 'static'
        else:
            model_type = 'dynamic'

        active = self._active_weighted_indices(resp)
        if active.numel() == 0:
            return q, q_lat

        batch_lds = self.lds_update == 'conjugate_batch'
        if batch_lds and not self.inducing_points and any(not torch.equal(x_trains[index], self.x_basis)
                             for index in active.tolist()):
            raise NotImplementedError('conjugate_batch requires a common observation grid')
        for index in active.tolist():
            h = float(resp[index].item())
            self.include_weighted_sample(
                index,
                x_trains[index],
                x_trains[index],
                y_trains[index],
                h
            )
            if batch_lds:
                _trajectory.hold_lds_params(self)
            elif model_type == 'dynamic':
                self.backwards_pair(h)
                self.bayesian_new_params(h)

        if model_type == 'dynamic':
            self.backwards()
        if batch_lds:
            self.bayesian_batch_params(resp)

        q_ = self.compute_sq_err_all(x_trains, y_trains)
        q_lat_ = self.compute_q_lat_all(x_trains)
        return q_, q_lat_

    def reinit_GP(self, save_last=False, save_index=False):
        """ Method to reinitiate GP parameters. Can save some of them.
        """
        if save_last:
            self.y_var = [self.y_var[0],self.y_var[-1]]
            self.var = [self.var[0],self.var[-1]]
            self.f_star = [self.f_star[0],self.f_star[-1]]
            self.f_star_sm = [torch.clone(self.f_star[0]),torch.clone(self.f_star[-1])]
            self.cov_f = [self.cov_f[0],self.cov_f[-1]]
            self.cov_f_sm = [self.cov_f_sm[0],self.cov_f_sm[-1]]
            self.y_train = []
            self.x_train = []
            if not save_index:
                self.indexes = [0]

        else:
            self.y_var = self.y_var[:1]
            self.var = self.var[:1]
            self.f_star = self.f_star[:1]
            self.f_star_sm = self.f_star[:1].copy()
            self.cov_f = [torch.clone(self.ini_cov_def)]
            self.cov_f_sm = [torch.clone(self.ini_cov_def)]
            self.indexes = []
            self.y_train = []
            self.x_train = []
        self.likelihood = []
        self.N = 0


    def reinit_LDS(self, save_last=False, save_last_diag=False, return_likelihood=False,
                   *, warm_start=False):
        """ Method to reinitiate LDS parameters. Can save some of them.
        """
        if self.lds_update == 'conjugate_batch' and self.N > 0 and warm_start:
            # Carry the last fit into the next offline E-step, while resetting
            # the MNIW posterior to the original prior for its next batch step.
            filter_start = (self.A[-1], self.Gamma[-1], self.C[-1], self.Sigma[-1])
            result = _trajectory.reinit_LDS(self, False, save_last_diag, return_likelihood)
            self.A[0], self.Gamma[0], self.C[0], self.Sigma[0] = filter_start
            return result
        if self.lds_update == 'conjugate_batch':
            return _trajectory.reinit_LDS(self, False, save_last_diag, return_likelihood)
        return _trajectory.reinit_LDS(self, save_last, save_last_diag, return_likelihood)

    def bayesian_batch_params(self, resp, grid_samples=None):
        """Update the cluster LDS once from its offline smoothed trajectory."""
        return _trajectory.bayesian_batch_params(self, resp, grid_samples)

    def return_LDS_param_likelihood(self, first=False):
        """ Method to compute the likelihood of LDS parameters over the prior.
        """
        return _trajectory.return_LDS_param_likelihood(self, first)

    def compute_sq_err_all(self, x_trains, y_trains, no_first=False):
        """Method to compute the squared error over all provided examples y_trains."""
        n_samps = x_trains.shape[0]
        device = x_trains.device if torch.is_tensor(x_trains) else x_trains[0].device
        sq_err = torch.zeros(n_samps, device=device, dtype=torch.float64)

        if len(self.indexes) == 0:
            return sq_err

        idx_t = torch.as_tensor(self.indexes, device=device, dtype=torch.long)
        sample_ids = torch.arange(n_samps, device=device, dtype=torch.long)

        pos_of_sample = torch.full((n_samps,), -1, device=device, dtype=torch.long)
        pos_of_sample[idx_t] = torch.arange(idx_t.numel(), device=device, dtype=torch.long)

        exact_mask = pos_of_sample >= 0
        closest_pos0 = torch.searchsorted(idx_t, sample_ids, right=True) - 1

        # State zero is the prior; state p + 1 has assimilated member p.
        # A nonmember uses the most recent member, or the prior if none exists.
        i_vals = torch.where(
            exact_mask,
            pos_of_sample + 1,
            closest_pos0 + 1,
        )
        first_mask = exact_mask & (i_vals == 1) & (not no_first)

        # Fast path only if all samples share the same x-grid
        shared_grid = torch.equal(x_trains, x_trains[0:1].expand_as(x_trains))
        if shared_grid:
            x0 = x_trains[0]
            group_code = i_vals * 2 + first_mask.to(i_vals.dtype)

            for code in torch.unique(group_code):
                ids = torch.nonzero(group_code == code, as_tuple=False).squeeze(1)
                i_cur = int((code // 2).item())
                first_cur = bool((code % 2).item())

                f_star, cov_f = self.observe(x0, i_cur, params=None, proj=False,
                                             final_lds=self.lds_scoring == 'final',
                                             smoothed=True)
                if first_cur:
                    score_sigma = self.Sigma[-1] if self.lds_scoring == 'final' else self.Sigma[0]
                    ini_noise = torch.mean(torch.diag(score_sigma)) * 1e-2
                    cov_f = cov_f + ini_noise * self._eye_cached(len(x0), cov_f.device, cov_f.dtype)

                latent_cov, C_mat = self._observation_params(i_cur)
                output_latent_cov = self._observation_latent_cov(x0, latent_cov, C_mat)
                sq_err[ids] = self._gaussian_score_shared_cov(
                    y_trains[ids], f_star, cov_f, output_latent_cov)

            return sq_err

        # Fallback for irregular x-grids
        i_vals_cpu = i_vals.cpu().tolist()
        first_mask_cpu = first_mask.cpu().tolist()

        for index in trange(n_samps, desc="Compute_sq_error", disable=self.disable):
            sq_err[index] = self.log_sq_error(
                x_trains[index],
                y_trains[index],
                i=i_vals_cpu[index],
                first=first_mask_cpu[index],
            )

        return sq_err

    def compute_q_lat_all(self, x_trains, h_ini=1.0):
        """Compute each included state's Gaussian prior and entropy contribution.
        """
        sq_err = torch.zeros(x_trains.shape[0], device=x_trains[0].device)
        if self.N == 0:
            return sq_err
        if len(torch.nonzero(self.Gamma[-1]))< 1:
            return sq_err
        for j, index in enumerate(self.indexes):
            sq_err[index] = self.log_lat_error(j, h_ini)
        return sq_err

    def posterior_weighted(self, x_train, y, h, t=None):
        """ Method to compute the posterior depending on the responsibility h.
        """
        y = self.cond_to_torch(y)
        x_train = self.cond_to_torch(x_train)
        if h > 0.0:
            if t is not None and len(self.indexes) > t:
                f_star_sm_, cov_f_sm_ = self.f_star[t], self.cov_f[t]
                A, Gamma, C, Sigma = self.get_params(t)
            else:
                f_star_sm_ = self.f_star[-1]
                cov_f_sm_ = self.cov_f[-1]
                A = self.A[-1]
                Gamma = self.Gamma[-1]
                C = self.C[-1]
                Sigma = self.Sigma[-1]
            observation_noise = (Sigma if getattr(self, 'direct_observation_offset', None)
                                 is not None else Sigma / h)
            f_star_, cov_f_ = self.gp.posterior(f_star_sm_, cov_f_sm_, y, A,
                                                Gamma / h, C, observation_noise,
                                                x_train=x_train, h=h)
        else:
            f_star_ = torch.clone(self.f_star[-1])
            cov_f_ = torch.clone(self.cov_f[-1])
        return f_star_, cov_f_

    def find_closest_lower(self, t):
        """Method to compute the closest last sample added to the model.
        """
        #List is assumed sorted
        lst = self.indexes
        idx = bisect_right(lst, t)
        if idx:
            return idx-1
        else:
            return 0

    def step_forward_last(self, x_post, params=None):
        """ Compute the observation over new x_post.
        """
        if params is None:
            C = self.C[-1]
            A = self.A[-1]
            Gamma = self.Gamma[-1]
            Sigma = self.Sigma[-1]
            mean = self.f_star_sm[-1]
            cov = self.cov_f_sm[-1]
        else:
            A = self.A[-1]
            Gamma = self.Gamma[-1]
            mean = params[0]
            cov = params[1]
            C = params[2]
            Sigma = params[3]
        mean = torch.linalg.multi_dot([C, mean])
        Sigma = Sigma# + torch.linalg.multi_dot([C, Gamma, C.T])
        x_basis = self.x_basis
        return self._observation_distribution(x_post, x_basis, mean, Sigma)

    def _observation_distribution(self, x_post, x_basis, mean, Sigma):
        """Use the direct PCA decoder or the original GP grid projection."""
        offset = getattr(self, 'direct_observation_offset', None)
        if offset is None:
            return self.gp.pred_dist(x_post, x_basis, mean, Sigma)
        x_post = self.cond_to_torch(x_post).reshape(-1, 1)
        if not torch.equal(x_post, self.direct_observation_grid):
            raise ValueError('PCA emission requires its calibrated observation grid')
        return mean.reshape(-1, 1) + offset, Sigma

    def _project_process_noise(self, Gamma, C):
        if getattr(self, 'direct_observation_offset', None) is not None:
            return C @ Gamma @ C.T
        return Gamma

    def observe_last(self, x_post):
        """ Compute the last observation distribution over new x_post.
        """
        C = self.C[-1]
        Sigma = self.Sigma[-1]
        mean = torch.matmul(C, self.f_star_sm[-1])
        x_basis = self.x_basis
        return self._observation_distribution(x_post, x_basis, mean, Sigma)

    def observe(self, x_post, t, params=None, proj=False, *, final_lds=False,
                smoothed=False):
        """
        Method to resample the emission GP distribution at time t or using the parameters given.
        :param x_post: time set to resample
        :param t: step n where to resample the model
        :param params: tuple (mean, cov, C, Sigma)
        :param final_lds: use final C and Sigma with the state at t
        :param smoothed: use the saved smoothed state mean for ELBO scoring
        """
        states = self.f_star_sm if smoothed else self.f_star
        if final_lds and params is None:
            # Preserve the state associated with t while evaluating every
            # observation under the cluster's final C and Sigma.
            state_index = t if 0 <= t < len(states) else -1
            mean = self.C[-1] @ states[state_index]
            Sigma = self.Sigma[-1]
            if proj:
                Sigma = Sigma + self._project_process_noise(self.Gamma[-1], self.C[-1])
            return self._observation_distribution(x_post, self.x_basis, mean, Sigma)
        if params is None:
            #Case when model is not initialited
            if len(self.indexes) == 0:
                C = self.C[0]
                Sigma = self.Sigma[0]
                mean = torch.matmul(C, states[0])
            #Case when computing error with last (predict)
            elif len(self.indexes) <= t:
                C = self.C[-1]
                Sigma = self.Sigma[-1]
                A = self.A[-1]
                Gamma = self.Gamma[-1]
                mean = torch.linalg.multi_dot([C, states[-1]])
            elif self.estimation_limit <= t:
                C = self.C[-1]
                Sigma = self.Sigma[-1]
                if proj:
                    Sigma = Sigma + self._project_process_noise(self.Gamma[-1], C)
                mean = torch.matmul(C, states[t])
            else:
                A, Gamma, C, Sigma = self.get_params(t)
                if proj:
                    Sigma = Sigma + self._project_process_noise(Gamma, C)
                mean = torch.matmul(C, states[t])
        else:
            mean = params[0]
            Sigma = params[3]
            mean = torch.matmul(params[2], mean)
        x_basis = self.x_basis
        return self._observation_distribution(x_post, x_basis, mean, Sigma)

    def get_params(self, t):
        """Method to return params on a specific iteration of the model
        """
        rest_len = len(self.C)
        ind = t if t < rest_len else -1
        return self.A[ind], self.Gamma[ind], self.C[ind], self.Sigma[ind]

    def resample_latent_mean(self, x_post, t=None, params=None):
        """Method to resample the latent process on a specific iteration of the model
        """
        if params is None:
            if t is None or t> len(self.indexes):
                mean = self.f_star_sm[-1]
                cov = self.cov_f_sm[-1]
            else:
                mean = self.f_star_sm[t]
                cov = self.cov_f_sm[t]
        else:
            mean = params[0]
            cov = params[1]
        x_basis = self.x_basis
        return self.gp.pred_latent_dist(x_post, x_basis, mean, cov)

    def backwards(self, h=1.0):
        """Method to compute backward recursion weighted by the responsibility.
        """
        return _trajectory.backwards(self, h)

    def backwards_pair(self, h, snr=None):
        """ Fast method to compute the last two backward iterations.
        """
        return _trajectory.backwards_pair(self, h, snr)

    def smoother_weighted(self, x_train, y, h):
        """ Method to compute the conditioned posterior and distribution if a sample is added.
        """
        f_star_aux, cov_f_aux = self.posterior_weighted(x_train, y, h)
        means = self.f_star.copy()
        means.append(f_star_aux)
        covs = self.cov_f.copy()
        covs.append(cov_f_aux)
        C = self.C.copy()
        C.append(self.C[-1])
        Sigma = self.Sigma.copy()
        Sigma.append(self.Sigma[-1])
        return means, covs, C, Sigma

    def smoother_weighted_index(self, x_train, y, h, t):
        """ Method to return the conditioned posterior and LDS parameters of a specific iteration of the model.
        """
        f_star_aux, cov_f_aux = self.posterior_weighted(x_train, y, h, t)
        A, Gamma, C, Sigma = self.get_params(t)
        return f_star_aux, cov_f_aux, C, Sigma

    def new_params(self, batch=None, reestimate=True, model_type='dynamic', verbose=True, check_var=False):
        """ Maximum Likelihood computation of the new LDS params.
        """
        return _trajectory.new_params(self, batch, reestimate, model_type, verbose, check_var)

    def check_bound_sigma(self, S):
        """ Method to check variance bounds (not actually used)
        """
        bounds = np.exp(self.gp.kernel.bounds[0]) ** 2
        for i in range(S.shape[0]):
            if S[i][i] < bounds[0]:
                S[i][i] = bounds[0]
            elif S[i][i] > bounds[1]:
                S[i][i] = bounds[1]
        return S

    def new_params_weighted(self, h, batch=None, reestimate=True, model_type='dynamic', min_samples=1, max_samples=6,
                            div_samples=15, verbose=True, check_var=False):
        """ Method to compute iteratively LDS params conditioned on the responsibility.
        """
        return _trajectory.new_params_weighted(self, h, batch, reestimate, model_type, min_samples, max_samples, div_samples, verbose, check_var)

    def revise_constraint_noise_step(self, Sigma_prior, Sigma_new):
        """ Method of natural gradient for covariance estimation (not actually used)
        """
        # Condition to ensure a minimal variance is required (occurs when two samples
        # are so similar and we assume a very certain model)
        if np.trace(Sigma_new) / np.trace(Sigma_prior) < 0.3 or np.trace(Sigma_new) / np.trace(Sigma_prior) > 2.0:
            # If reduces a third or doubles their medium diagonal variance value we assume is a strong assumption.
            Sigma_new = (Sigma_new + Sigma_prior) / 2
        return Sigma_new

    def KL_divergence(self, t, gpmodel, t_gp, smoothed=True, x_bas=None):
        """ Method to compute the Kullback-Leibler divergence over every iteration of two LDS models.
        """
        l1 = t
        l2 = t_gp
        if smoothed:
            f_m1 = self.f_star_sm[l1 + 1]
            f_m2 = gpmodel.f_star_sm[l2 + 1]
            cov_m1 = self.cov_f_sm[l1 + 1]
            cov_m2 = gpmodel.cov_f_sm[l2 + 1]
        else:
            f_m1 = self.f_star[l1 + 1]
            f_m2 = gpmodel.f_star[l2 + 1]
            cov_m1 = self.cov_f[l1 + 1]
            cov_m2 = gpmodel.cov_f[l2 + 1]
        if self.estimation_limit <= t:
            t = -1
        if gpmodel.estimation_limit <= t_gp:
            t_gp = -1
        if len(self.Gamma) == 0:
            return self.gp.KL_divergence(f_m1, cov_m1, f_m2, cov_m2)
        elif torch.all(self.Gamma[-1] == 0):
            return self.gp.KL_divergence(f_m1, cov_m1, f_m2, cov_m2)
        else:
            if x_bas is not None and not torch.equal(x_bas, self.x_basis):
                mean1, cov1 = self.observe(x_bas, t, params=[f_m1, cov_m1, self.C[t], self.Sigma[t]])
                mean2, cov2 = gpmodel.observe(x_bas, t_gp, params=[f_m2, cov_m2, gpmodel.C[t_gp], gpmodel.Sigma[t_gp]])
            else:
                mean1 = torch.matmul(self.C[t], f_m1)
                mean2 = torch.matmul(gpmodel.C[t_gp], f_m2)
                cov1 = torch.linalg.multi_dot([self.C[t], cov_m1, self.C[t].T]) + self.Sigma[t]
                cov2 = torch.linalg.multi_dot([gpmodel.C[t_gp], cov_m2, gpmodel.C[t_gp].T]) + gpmodel.Sigma[t_gp]
            return self.gp.KL_divergence(mean1, cov1, mean2, cov2)

    def compute_mean(self):
        """ Method to resample mean (usually zero mean)
        """
        return self.gp.compute_mean(self.x_basis)

    def plot_last(self, num_model):
        """ Predefined method to plot last iteration of the model.
        """
        if len(self.indexes) == 0:
            y = np.repeat(0, len(self.x_basis))
            x_train = np.repeat(0, len(self.x_basis))
        else:
            y = self.y_train[-1]
            x_train = self.x_train[-1]

        self.gp.plotGP(len(self.indexes) + num_model, self.x_basis,
                       np.dot(self.C[-1], self.f_star[-1]), np.sqrt(self.var[-1]),
                       x_train, y, np.sqrt(self.y_var[-1]),
                       title=True, label_model=num_model, labels=True)

    def sample_last(self, num_samples=1, random_state=0):
        """ Method to resample last GP as a distribution.
        """
        if getattr(self, 'direct_observation_offset', None) is not None:
            mean = ((self.C[-1] @ self.f_star_sm[-1]).reshape(-1, 1)
                    + self.direct_observation_offset).reshape(-1)
            covariance = (self.C[-1] @ self.cov_f_sm[-1] @ self.C[-1].T
                          + self.Sigma[-1])
            draws = np.random.default_rng(random_state).multivariate_normal(
                mean.detach().cpu().numpy(), covariance.detach().cpu().numpy(),
                size=num_samples)
            return [draw.copy() for draw in np.atleast_2d(draws)]
        samples = self.gp.sample_y(self.f_star_sm[-1], self.cov_f_sm[-1], self.C[-1], self.Sigma[-1], num_samples,
                                   random_state).T
        rav_samples = []
        for i in range(num_samples):
            rav_samples.append(samples[i][0])
        return rav_samples

    def reduce_noise_matrix(self, x_basis=None, x_train=None):
        return self.gp.projection_matrix(x_basis, x_train)

    def project_observation_to_basis(self, y, x_train):
        """Map a waveform to the inducing grid for the LDS observation update."""
        observation = self.cond_to_cuda(self.cond_to_torch(y)).reshape(-1, 1)
        if getattr(self, 'direct_observation_offset', None) is not None:
            if not torch.equal(self.cond_to_torch(x_train).reshape(-1, 1),
                               self.direct_observation_grid):
                raise ValueError('PCA emission requires its calibrated observation grid')
            return self.C[-1].T @ (observation - self.direct_observation_offset)
        if torch.equal(self.x_basis, x_train):
            return observation
        if not self.inducing_points:
            raise NotImplementedError('LDS observation projection requires inducing_points=True')
        projection = self.reduce_noise_matrix(self.x_basis, x_train)
        return projection.to(device=observation.device, dtype=observation.dtype) @ observation

    def bayesian_new_params(self, h, model_type='dynamic', full_data=False, q=None, force=False,
                            snr=1.0, first_observation=False):
        """ Method to compute the variational Bayesian step for LDS parameters. Can deal with dynamic or static models.
            Can use full data batch n-step estimation or 1-step estimation.
        """
        if self.lds_update == 'conjugate_batch':
            raise ValueError('conjugate_batch updates require an offline full pass')
        return _trajectory.bayesian_new_params(
            self, h, model_type, full_data, q, force, snr,
            first_observation=first_observation)


    #Methods to perform the conversion from NumPy to torch and from Cuda to cpu. Still have to solve this.
    def cond_to_numpy(self, x):
        if x is not None:
            if type(x) is torch.Tensor:
                x = x.detach().numpy()
        return x

    def cond_to_torch(self, x):
        if x is not None:
            if type(x) is not torch.Tensor:
                x = torch.from_numpy(np.array(x))
                x.requires_grad = False
            else:
                x.requires_grad = False
        return x

    def cond_to_cuda(self, x):
        if self.cuda:
            if x is not None:
                if type(x) is torch.Tensor and torch.cuda.is_available():
                    x = x.cuda()
                    x.requires_grad = False
        return x

    def cond_to_cpu(self, x):
        if x is not None:
            if type(x) is torch.Tensor:
                if x.is_cuda:
                    x = x.cpu()
        return x

    def model_to_numpy(self):
        def recursive_numpy(x):
            if type(x) is torch.Tensor:
                x = self.cond_to_numpy(x)
            elif type(x) is list:
                if len(x) > 0:
                    if type(x[0]) is torch.Tensor:
                        for j, i in enumerate(x):
                            x[j] = self.cond_to_numpy(i)
            else:
                x = x
            return x

        self.x_basis = recursive_numpy(self.x_basis)
        self.x_train = recursive_numpy(self.x_train)
        self.y_train = recursive_numpy(self.y_train)
        self.f_star = recursive_numpy(self.f_star)
        self.f_star_sm = recursive_numpy(self.f_star_sm)
        self.cov_f = recursive_numpy(self.cov_f)
        self.cov_f_sm = recursive_numpy(self.cov_f_sm)
        self.y_var = recursive_numpy(self.y_var)
        self.var = recursive_numpy(self.var)
        self.A = recursive_numpy(self.A)
        self.Gamma = recursive_numpy(self.Gamma)
        self.C = recursive_numpy(self.C)
        self.Sigma = recursive_numpy(self.Sigma)
        self.likelihood = recursive_numpy(self.likelihood)
        self.K = recursive_numpy(self.K)
        if getattr(self, 'transition_policy', None) is None:
            self.internal_params.to_numpy()
        if getattr(self, 'observation_policy', None) is None:
            self.observation_params.to_numpy()

    def model_to_torch(self):
        def recursive_torch(x):
            if type(x) is np.ndarray:
                x = self.cond_to_torch(x)
            elif type(x) is list:
                if len(x) > 0:
                    if type(x[0]) is np.ndarray or type(x[0]) is np.float64:
                        for j, i in enumerate(x):
                            x[j] = self.cond_to_torch(i)
            else:
                x = x
            return x

        self.x_basis = recursive_torch(self.x_basis)
        self.x_train = recursive_torch(self.x_train)
        self.y_train = recursive_torch(self.y_train)
        self.f_star = recursive_torch(self.f_star)
        self.f_star_sm = recursive_torch(self.f_star_sm)
        self.cov_f = recursive_torch(self.cov_f)
        self.cov_f_sm = recursive_torch(self.cov_f_sm)
        self.y_var = recursive_torch(self.y_var)
        self.var = recursive_torch(self.var)
        self.A = recursive_torch(self.A)
        self.Gamma = recursive_torch(self.Gamma)
        self.C = recursive_torch(self.C)
        self.Sigma = recursive_torch(self.Sigma)
        self.likelihood = recursive_torch(self.likelihood)
        self.K = recursive_torch(self.K)
        if getattr(self, 'transition_policy', None) is None:
            self.internal_params.to_torch()
        if getattr(self, 'observation_policy', None) is None:
            self.observation_params.to_torch()

    def model_to_cuda(self):
        if torch.cuda.is_available() and self.cuda:
            def recursive_cuda(x):
                if type(x) is torch.Tensor:
                    x = x.cuda()
                    x.requires_grad = False
                elif type(x) is list:
                    if len(x) > 0:
                        if type(x[0]) is torch.Tensor or type(x[0]) is torch.float64:
                            for j, i in enumerate(x):
                                x[j] = i.cuda()
                                x[j].requires_grad = False
                else:
                    x = x
                return x

            self.x_basis = recursive_cuda(self.x_basis)
            self.x_train = recursive_cuda(self.x_train)
            self.y_train = recursive_cuda(self.y_train)
            self.f_star = recursive_cuda(self.f_star)
            self.f_star_sm = recursive_cuda(self.f_star_sm)
            self.cov_f = recursive_cuda(self.cov_f)
            self.cov_f_sm = recursive_cuda(self.cov_f_sm)
            self.y_var = recursive_cuda(self.y_var)
            self.var = recursive_cuda(self.var)
            self.A = recursive_cuda(self.A)
            self.Gamma = recursive_cuda(self.Gamma)
            self.C = recursive_cuda(self.C)
            self.Sigma = recursive_cuda(self.Sigma)
            self.likelihood = recursive_cuda(self.likelihood)
            self.K = recursive_cuda(self.K)
            if getattr(self, 'transition_policy', None) is None:
                self.internal_params.to_cuda()
            if getattr(self, 'observation_policy', None) is None:
                self.observation_params.to_cuda()
            self.cuda = True
            self.gp.cuda = True

    def model_to_cpu(self):
        def recursive_cpu(x):
            if type(x) is torch.Tensor:
                x = x.cpu()
            elif type(x) is list:
                if len(x) > 0:
                    if type(x[0]) is torch.Tensor or type(x[0]) is torch.float64:
                        for j, i in enumerate(x):
                            x[j] = i.cpu()
            else:
                x = x
            return x

        self.x_basis = recursive_cpu(self.x_basis)
        self.x_train = recursive_cpu(self.x_train)
        self.y_train = recursive_cpu(self.y_train)
        self.f_star = recursive_cpu(self.f_star)
        self.f_star_sm = recursive_cpu(self.f_star_sm)
        self.cov_f = recursive_cpu(self.cov_f)
        self.cov_f_sm = recursive_cpu(self.cov_f_sm)
        self.y_var = recursive_cpu(self.y_var)
        self.var = recursive_cpu(self.var)
        self.A = recursive_cpu(self.A)
        self.Gamma = recursive_cpu(self.Gamma)
        self.C = recursive_cpu(self.C)
        self.Sigma = recursive_cpu(self.Sigma)
        self.likelihood = recursive_cpu(self.likelihood)
        self.K = recursive_cpu(self.K)
        if getattr(self, 'transition_policy', None) is None:
            self.internal_params.to_cpu()
        if getattr(self, 'observation_policy', None) is None:
            self.observation_params.to_cpu()
        self.cuda = False
        self.gp.cuda = False
