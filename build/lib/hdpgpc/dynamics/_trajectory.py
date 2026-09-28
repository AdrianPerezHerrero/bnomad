"""Original trajectory-level LDS updates on the supplied emission state.

The state argument retains the legacy name self to leave method bodies intact.
It is the existing emission object, not a copied state container. These private
functions use its histories, matrix/default fields, parameter distributions,
GP projection helpers and conversion/stabilization methods. They import no
emission or high-level model class and retain all existing mutation/update gates.
"""

import numpy as np
import torch
from ._distributions import matrix_normal_inv_wishart, inv_wishart


def reinit_LDS(self, save_last=False, save_last_diag=False, return_likelihood=False):
    """ Method to reinitiate LDS parameters. Can save some of them.
    """
    if save_last:
        ind_ = -1
        if save_last_diag:
            ini_A, ini_Gamma, ini_C, ini_Sigma = self.A_def, torch.diag(torch.diag(self.Gamma[ind_])) * 3.0, self.C_def, torch.diag(torch.diag(self.Sigma[ind_])) * 3.0
        else:
            ini_A, ini_Gamma, ini_C, ini_Sigma = self.A[ind_], self.Gamma[ind_], self.C[ind_], self.Sigma[ind_]
    else:
        ini_A, ini_Gamma, ini_C, ini_Sigma = self.A_def, self.Gamma_def, self.C_def, self.Sigma_def
        if return_likelihood:
            A_, Gam_, C_, Sig_ = self.A[-1], self.Gamma[-1], self.C[-1], self.Sigma[-1]
    self.A = [ini_A]
    self.Gamma = [ini_Gamma]
    self.C = [ini_C]
    self.Sigma = [ini_Sigma]
    self.internal_params = matrix_normal_inv_wishart(ini_A, torch.eye(ini_A.shape[0], device=self.device), self.free_deg_MNIV, ini_Gamma)
    self.observation_params = matrix_normal_inv_wishart(ini_C, torch.eye(ini_C.shape[0], device=self.device), self.free_deg_MNIV, ini_Sigma)
    if return_likelihood:
        return self.internal_params.log_likelihood_MNIW(A_, Gam_), self.observation_params.log_likelihood_MNIW(C_, Sig_)


def return_LDS_param_likelihood(self, first=False):
    """ Method to compute the likelihood of LDS parameters over the prior.
    """
    elb_LDS = 0
    #for i in range(len(self.A)):
    ini_A, ini_Gamma, ini_C, ini_Sigma, n0 = self.A_def, self.Gamma_def, self.C_def, self.Sigma_def, self.internal_params.n0
    if first:
        #ini_noise = self.cond_to_cuda(self.cond_to_torch(self.gp.kernel.get_params()["k2__noise_level"]))
        ini_noise = torch.mean(torch.diag(self.Sigma[-1])) * 2e-0
        ini_noise_ = torch.mean(torch.diag(self.Gamma[-1])) * 2e-0
        A_, Gam_, C_, Sig_ = (self.A[-1],
                              self.Gamma[-1] + torch.mul(ini_noise_, torch.eye(self.Gamma[-1].shape[0],
                                                                              device=ini_noise.device)),
                              self.C[-1],
                              self.Sigma[-1] + torch.mul(ini_noise, torch.eye(self.Sigma[-1].shape[0],
                                                                              device=ini_noise.device)))
        # A_, Gam_, C_, Sig_ = (self.A[-1], self.Gamma[-1] + self.cov_f[-1], self.C[-1],
        #                        self.Sigma[-1] + self.cov_f[-1])
    else:
        A_, Gam_, C_, Sig_, n0 = self.A[-1], self.Gamma[-1], self.C[-1], self.Sigma[-1], self.internal_params.n0
    if len(torch.nonzero(ini_Gamma))< 1:
        log_lik_A_Gam = 0.0
    else:
        int_params = matrix_normal_inv_wishart(ini_A, torch.eye(ini_A.shape[0], device=self.device), self.free_deg_MNIV, ini_Gamma)
        log_lik_A_Gam = int_params.log_likelihood_MNIW(A_, Gam_, n0)
    obs_params = matrix_normal_inv_wishart(ini_C, torch.eye(ini_C.shape[0], device=self.device), self.free_deg_MNIV, ini_Sigma)
    elb_LDS = elb_LDS + log_lik_A_Gam + obs_params.log_likelihood_MNIW(C_, Sig_, n0)
    return elb_LDS / ini_A.shape[0] * 100


def backwards(self, h=1.0):
    """Method to compute backward recursion weighted by the responsibility.
    """
    if len(torch.nonzero(self.Gamma[-1])) < 1:
        model_type = 'static'
    else:
        model_type = 'dynamic'
    if h == 1.0:
        mean = list(self.f_star[1:])
        covs = list(self.cov_f[1:])
        if model_type == 'dynamic':
            aux_f_star, aux_cov_f = self.gp.backward(self.A[1:], self.Gamma[1:], mean, covs)
        else:
            aux_f_star, aux_cov_f = self.gp.backward(self.A[0], self.Gamma[0], mean, covs)
        for i in range(len(mean)):
            self.f_star_sm[i + 1] = aux_f_star[i]
            self.cov_f_sm[i + 1] = aux_cov_f[i]


def backwards_pair(self, h, snr=None):
    """ Fast method to compute the last two backward iterations.
    """
    if len(self.indexes) > 1:
        if h == 1.0:
            if snr is None:
                mean = list(self.f_star[-2:])
                covs = list(self.cov_f[-2:])
                aux_f_star, aux_cov_f = self.gp.backward_notrange(self.A[-1], self.Gamma[-1], mean, covs)
                for i in range(len(mean)):
                    self.f_star_sm[-(i + 1)] = aux_f_star[-(i + 1)]
                    self.cov_f_sm[-(i + 1)] = aux_cov_f[-(i + 1)]
            else:
                if snr > 0.5:
                    mean = self.f_star_sm[-2:]
                    covs = self.cov_f_sm[-2:]
                    aux_f_star, aux_cov_f = self.gp.backward_notrange(self.A[-1], self.Gamma[-1], mean, covs)
                    for i in range(len(mean)):
                        self.f_star_sm[-(i + 1)] = aux_f_star[-(i + 1)]
                        self.cov_f_sm[-(i + 1)] = aux_cov_f[-(i + 1)]


def new_params(self, batch=None, reestimate=True, model_type='dynamic', verbose=True, check_var=False):
    """ Maximum Likelihood computation of the new LDS params.
    """
    if batch is None or batch >= self.N:
        batch = self.N
    if reestimate:
        converged = False
        # Declare parameters to start iterations.
        # As we include all y_train, we should select only indexed samples
        N = self.N
        means = self.f_star_sm[1:]
        covs = self.cov_f_sm[1:]
        y_samples = self.y_train
        A_prior = self.A[-1]
        Gamma_prior = self.Gamma[-1]
        C_prior = self.C[-1]
        Sigma_prior = self.Sigma[-1]
        if self.annealing:
            Gamma_prior = Gamma_prior - self.Gamma[0] / (2 * N)
            Sigma_prior = Sigma_prior - self.Sigma[0] / (2 * N)
        # Compute previous likelihood
        try:
            lik_pre = self.gp.log_likelihood(N - batch, batch - 1, means[N - batch:], covs[N - batch:], A_prior,
                                             Gamma_prior, y_samples[N - batch:], C_prior, Sigma_prior)
        except RuntimeWarning:
            print("Starting parameters are divergent, using initials.")
            converged = True
        lik_best = lik_pre
        trials = 0
        if N < 101:
            try_max = 6
        else:
            try_max = 4
        A_best = A_prior
        Gamma_best = Gamma_prior
        C_best = C_prior
        Sigma_best = Sigma_prior
        while not converged and trials < try_max:
            try:
                A_new, Gamma_new, C_new, Sigma_new = self.gp.new_params_LDS(A_prior, Gamma_prior, C_prior, Sigma_prior,
                                                                            y_samples[N - batch:], means[N - batch:],
                                                                            covs[N - batch:], model_type)
                means, covs = self.gp.backward(A_new, Gamma_new, means, covs)
                lik_post = self.gp.log_likelihood(N - batch, batch - 1, means[N - batch:], covs[N - batch:], A_new,
                                                  Gamma_new, y_samples[N - batch:], C_new, Sigma_new)
            except RuntimeWarning:
                if verbose:
                    print("Finded parameters are divergent, using initials.")
                    break
            if not torch.isnan(lik_post) and torch.isclose(lik_best, lik_post, 0.01) and lik_best <= lik_post and not torch.isinf(lik_post):
                converged = True
                if verbose:
                    print('Last iteration: ', torch.abs(lik_pre - lik_post))
                A_best = A_new
                Gamma_best = Gamma_new
                C_best = C_new
                Sigma_best = Sigma_new
            else:
                if torch.isnan(lik_post):
                    if verbose:
                        print('Singular matrix detected, using previous.')
                        trials = try_max
                elif lik_best > lik_post:
                    if verbose:
                        print('Divergence detected, using previous.')
                        trials = try_max
                elif torch.isinf(lik_post):
                    if verbose:
                        print('Divergence detected, using previous.')
                        trials = try_max
                else:
                    if verbose:
                        print('Iterating: Step-', lik_post - lik_pre, ' Diference with best-', lik_post - lik_best)
                    if lik_best <= lik_post:
                        # Here we are obtaining a better model so we save it
                        lik_best = lik_post
                        A_best = A_new
                        Gamma_best = Gamma_new
                        C_best = C_new
                        Sigma_best = Sigma_new
                    # We keep iterating until we reach some critical point.
                    lik_pre = lik_post
                    A_prior = A_new
                    Gamma_prior = Gamma_new
                    C_prior = C_new
                    Sigma_prior = Sigma_new
                trials = trials + 1
        if converged:
            if verbose:
                print('Converged estimation of new LDS parameters.')
            if self.annealing:
                Gamma_best = Gamma_best + self.Gamma[0] / (2 * N)
                Sigma_best = Sigma_best + self.Sigma[0] / (2 * N)
            self.A.append(A_best)
            self.Gamma.append(Gamma_best)
            self.C.append(C_best)
            if check_var:
                Sigma_best = self.check_bound_sigma(Sigma_best)
            self.Sigma.append(Sigma_best)
            self.var.append(torch.atleast_2d(torch.diag(Gamma_best)).T)
            self.y_var.append(torch.atleast_2d(torch.diag(Sigma_best)).T)
        else:
            self.A.append(self.A[-1])
            self.Gamma.append(self.Gamma[-1])
            self.C.append(self.C[-1])
            self.Sigma.append(self.Sigma[-1])
            self.var.append(torch.atleast_2d(torch.diag(self.Gamma[-1])).T)
            self.y_var.append(torch.atleast_2d(torch.diag(self.Sigma[-1])).T)
    else:
        self.A.append(self.A[-1])
        self.Gamma.append(self.Gamma[-1])
        self.C.append(self.C[-1])
        self.Sigma.append(self.Sigma[-1])
        self.var.append(torch.atleast_2d(torch.diag(self.Gamma[-1])).T)
        self.y_var.append(torch.atleast_2d(torch.diag(self.Sigma[-1])).T)


def new_params_weighted(self, h, batch=None, reestimate=True, model_type='dynamic', min_samples=1, max_samples=6,
                        div_samples=15, verbose=True, check_var=False):
    """ Method to compute iteratively LDS params conditioned on the responsibility.
    """
    if not np.isclose(h, 0, rtol=1e-1, atol=1e-1):
        num_included = self.N
        if num_included > 500:
            div_samples = 10
        if num_included > min_samples and num_included < max_samples or (
                num_included % div_samples == 0 and not num_included == 0):
            self.backwards()
            self.new_params(batch, reestimate, model_type, verbose=verbose, check_var=check_var)
        else:
            self.new_params(0, reestimate=False, verbose=verbose)


def bayesian_new_params(self, h, model_type='dynamic', full_data=False, q=None, force=False, snr=1.0):
    """ Method to compute the variational Bayesian step for LDS parameters. Can deal with dynamic or static models.
        Can use full data batch n-step estimation or 1-step estimation.
    """
    if len(torch.nonzero(self.Gamma[-1]))< 1:
        model_type = 'static'
    if h == 1.0:
        if snr > 0.5:
            if (full_data and 1 < self.N) or 1 < self.N < self.estimation_limit or force:
                try:
                    # if not full_data:
                    #     self.backwards()
                    if model_type == 'dynamic':
                        if not full_data:
                            A, Gamma = self.A[-1], self.Gamma[-1]
                            # samples_A = self.f_star_sm[-1]
                            # samples_A_ = self.f_star_sm[-2]
                            samples_A = self.f_star_sm[-1]
                            samples_A_ = self.f_star_sm[-2]
                            #samples_A_ = torch.matmul(A, self.f_star_sm[-2])
                            cov = self.cov_f_sm[-1]
                            cov_ = self.cov_f_sm[-2]
                            cov_f_ = self.cov_f[-1]
                            cov_f__ = self.cov_f[-2]
                            P = A @ cov_ @ A.T + Gamma
                            L_P = self._chol_spd(P)
                            J = torch.cholesky_solve(A @ cov_.T, L_P).T
                            cov_cross = J @ cov
                            cov_cross = 0.5 * (cov_cross + cov_cross.T)
                            if True:
                                cov = torch.zeros(cov.shape, device=cov.device)
                                cov_ = torch.zeros(cov.shape, device=cov.device)
                                cov_cross = torch.zeros(cov.shape, device=cov.device)
                        else:
                            n_f = min(self.estimation_limit,len(self.f_star_sm)-2) if (
                                    self.estimation_limit != np.PINF) else len(self.f_star_sm)-2
                            samples_A = torch.stack(self.f_star_sm[2:n_f+2])[:, :, 0].T
                            samples_A_ = torch.stack(self.f_star_sm[1:n_f+1])[:, :, 0].T
                            cov = torch.sum(torch.stack(self.cov_f_sm[2:n_f+2]), axis=0)
                            cov_ = torch.sum(torch.stack(self.cov_f_sm[1:n_f+1]), axis=0)
                            cov_cross = torch.zeros_like(cov)
                            A, Gamma = self.A[-1], self.Gamma[-1]
                            for t in range(n_f + 1):
                                P = A @ self.cov_f_sm[t] @ A.T + Gamma
                                L_P = self._chol_spd(P)
                                J = torch.cholesky_solve(A @ self.cov_f_sm[t].T, L_P).T
                                cov_cross = cov_cross + J @ self.cov_f_sm[t + 1]
                            cov_cross = 0.5 * (cov_cross + cov_cross.T)
                        if not full_data:
                            N_k = 1
                        elif self.estimation_limit != np.PINF:
                            N_k = self.estimation_limit
                        else:
                            N_k = samples_A.shape[1]
                        new_int_dist = self.internal_params.posterior(N_k, samples_A, samples_A_, cov, cov_, cov_cross,
                                                                      annealing=self.annealing)
                    elif model_type == 'static':
                        N_k = 1
                        new_int_dist = self.internal_params
                    else:
                        print("Only programmed static and dynamic models.")
                        new_int_dist = self.internal_params
                    if not full_data:
                        samples_C = self.y_train[-1]
                        #samples_C_, _ = self.observe_last(self.x_train[-1])
                        if torch.equal(self.x_basis, self.x_train[-1]):
                            samples_C_ = self.f_star_sm[-1]
                            C, Sigma = self.C[-1], self.Sigma[-1]
                            cov = torch.zeros(Sigma.shape, device=Sigma.device)
                            cov_ = torch.zeros(cov.shape, device=cov.device)
                            cov_cross = torch.zeros(cov.shape, device=cov.device)
                        else:
                            samples_C_, cov = self.resample_latent_mean(self.x_train[-1])
                            cov = torch.zeros(cov.shape, device=cov.device)
                            cov_ = torch.zeros(cov.shape, device=cov.device)
                            cov_cross = torch.zeros(cov.shape, device=cov.device)
                        if model_type == 'static':
                            samples_C_, _ = self.resample_latent_mean(self.x_train[-1])
                        samples_C_ = self.cond_to_cuda(samples_C_)
                    else:
                        samples_C  = torch.stack(self.y_train[:n_f])[:, :, 0].T
                        samples_C_ = torch.stack(self.f_star_sm[1:n_f+1])[:, :, 0].T
                        cov_ = torch.sum(torch.stack(self.cov_f_sm[1:n_f+1]), axis=0)
                        cov = torch.zeros(cov_.shape, device=cov_.device)
                        C, Sigma = self.C[-1], self.Sigma[-1]
                        cov_cross = torch.zeros_like(cov_)
                        for t in range(n_f + 1):
                            P = C @ self.cov_f_sm[t] @ C.T + Sigma
                            L_P = self._chol_spd(P)
                            J = torch.cholesky_solve(C @ self.cov_f_sm[t].T, L_P).T
                            cov_cross = cov_cross + J @ Sigma
                        cov = cov + Sigma
                        if False:
                            cov = torch.zeros(cov.shape, device=cov.device)
                            cov_ = torch.zeros(cov.shape, device=cov.device)
                            cov_cross = torch.zeros(cov.shape, device=cov.device)
                    if torch.equal(self.x_basis, self.x_train[-1]):
                        project_mat = None
                    else:
                        project_mat = self.reduce_noise_matrix(self.x_basis, self.x_train[-1])
                    new_obs_dist = self.observation_params.posterior(N_k, samples_C, samples_C_, cov, cov_, cov_cross,
                                                                     sse_matrix=project_mat, annealing=self.annealing)
                except torch.linalg.LinAlgError:
                    print("Alg error matrix ill conditioned.")
                    new_int_dist = self.internal_params
                    new_obs_dist = self.observation_params
            else:
                new_int_dist = self.internal_params
                new_obs_dist = self.observation_params
            self.internal_params = new_int_dist
            self.observation_params = new_obs_dist
            if 1 < self.N:
                Gamma_ = new_int_dist.get_scale(final=full_data)
                Sigma_ = new_obs_dist.get_scale(final=full_data)
            else:
                Gamma_ = self.Gamma[-1]
                Sigma_ = self.Sigma[-1]
            if self.annealing:
                if model_type == 'static':
                    factor_S = self.Sigma[0] / (self.N ** 2)
                    factor_G = self.Gamma[0]
                else:
                    factor_G = self.Gamma[0] / (self.N ** 2)
                    factor_S = self.Sigma[0] / (self.N ** 2)
                Gamma_ = Gamma_ + factor_G
                Sigma_ = Sigma_ + factor_S
            if self.N < self.estimation_limit or full_data:
                self.A.append(new_int_dist.get_mean())
                self.Gamma.append(Gamma_)
                if model_type == 'static':
                    self.C.append(new_obs_dist.get_mean())
                else:
                    self.C.append(new_obs_dist.get_mean())
                self.Sigma.append(Sigma_)
                self.var.append(torch.atleast_2d(torch.sqrt(torch.diag(Gamma_))).T)
                self.y_var.append(torch.atleast_2d(torch.sqrt(torch.diag(Sigma_))).T)
        else:
            new_int_dist = self.internal_params
            new_obs_dist = self.observation_params
            Gamma_ = new_int_dist.get_scale(final=full_data)
            Sigma_ = new_obs_dist.get_scale(final=full_data)
            self.A.append(new_int_dist.get_mean())
            self.Gamma.append(Gamma_)
            if model_type == 'static':
                self.C.append(new_obs_dist.get_C())
            else:
                self.C.append(new_obs_dist.get_mean())
            self.Sigma.append(Sigma_)
            self.var.append(torch.atleast_2d(torch.sqrt(torch.diag(Gamma_))).T)
            self.y_var.append(torch.atleast_2d(torch.sqrt(torch.diag(Sigma_))).T)
