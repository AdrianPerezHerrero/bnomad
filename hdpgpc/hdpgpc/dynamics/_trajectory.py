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
    if torch.count_nonzero(ini_Gamma) == 0:
        self.observation_params = inv_wishart(self.free_deg_MNIV, ini_Sigma, ini_C)
    else:
        self.observation_params = matrix_normal_inv_wishart(ini_C, torch.eye(ini_C.shape[0], device=self.device), self.free_deg_MNIV, ini_Sigma)
    if return_likelihood:
        return self.internal_params.log_likelihood_MNIW(A_, Gam_), self.observation_params.log_likelihood_MNIW(C_, Sig_)


def return_LDS_param_likelihood(self, first=False):
    """Return the LDS parameter contribution, minus the posterior-to-prior KLs."""
    if not self.bayesian:
        return self.Sigma_def.new_zeros(())

    contribution = self.Sigma_def.new_zeros(())
    if torch.count_nonzero(self.Gamma_def) > 0:
        prior_A = matrix_normal_inv_wishart(
            self.A_def, torch.eye(self.A_def.shape[1], device=self.A_def.device,
                                  dtype=self.A_def.dtype),
            self.free_deg_MNIV, self.Gamma_def)
        contribution = contribution - self.internal_params.kl_divergence(prior_A)

    if isinstance(self.observation_params, inv_wishart):
        prior_C = inv_wishart(self.free_deg_MNIV, self.Sigma_def, self.C_def)
    else:
        prior_C = matrix_normal_inv_wishart(
            self.C_def, torch.eye(self.C_def.shape[1], device=self.C_def.device,
                                  dtype=self.C_def.dtype),
            self.free_deg_MNIV, self.Sigma_def)
    return contribution - self.observation_params.kl_divergence(prior_C)


def backwards(self, h=1.0):
    """Method to compute backward recursion weighted by the responsibility.
    """
    if len(torch.nonzero(self.Gamma[-1])) < 1:
        model_type = 'static'
    else:
        model_type = 'dynamic'
    if h > 0.0:
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
        if h > 0.0:
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


def _lds_posterior(self, distribution, n_k, y1, y2, cov, cov_, cov_cross,
                   *, sse_matrix=None, annealing=False):
    """Select the configured update without changing sufficient-statistic routing."""
    kwargs = {'sse_matrix': sse_matrix, 'annealing': annealing}
    policy = getattr(self, 'lds_update', 'conjugate_sequential')
    if policy == 'natural_gradient':
        return distribution.posterior_natural_gradient(
            n_k, y1, y2, cov, cov_, cov_cross,
            step_size=getattr(self, 'lds_step_size', 0.2), **kwargs,
        )
    if policy == 'legacy' and isinstance(distribution, matrix_normal_inv_wishart):
        return distribution.posterior_legacy(n_k, y1, y2, cov, cov_, cov_cross, **kwargs)
    if policy not in ('conjugate_sequential', 'legacy'):
        raise ValueError(f'Unknown LDS update policy: {policy}')
    return distribution.posterior(n_k, y1, y2, cov, cov_, cov_cross, **kwargs)


def _lds_scale(self, distribution, *, final):
    scale = distribution.get_scale(final=final)
    # The historical MNIW getter inflated the online scale but returned the
    # stored scale for a full-data update. Keep that behavior with legacy.
    if (getattr(self, 'lds_update', 'conjugate_sequential') == 'legacy'
            and isinstance(distribution, matrix_normal_inv_wishart) and not final):
        return scale * distribution.n0 / (distribution.n0 - 2)
    return scale


def hold_lds_params(self):
    """Keep one parameter slot per filtered beat without a sequential update."""
    for name in ('A', 'Gamma', 'C', 'Sigma', 'var', 'y_var'):
        history = getattr(self, name)
        history.append(history[-1])


def bayesian_batch_params(self, resp, grid_samples=None):
    """One responsibility-weighted MNIW update from the smoothed cluster path.

    The first included state has an observation but no LDS transition. A
    transition receives the responsibility of its destination observation.
    The distributions on entry are the original cluster priors, even when
    filtering was warm-started from a previous offline iteration.
    """
    if not self.indexes:
        return
    if not self.bayesian:
        raise ValueError('conjugate_batch requires Bayesian LDS parameters')
    if grid_samples is not None:
        raise NotImplementedError('conjugate_batch requires a common observation grid')

    count = len(self.indexes)
    limit = count if np.isinf(self.estimation_limit) else min(count, int(self.estimation_limit))
    if limit == 0:
        return
    prior_obs = self.observation_params
    prior_int = self.internal_params
    device, dtype = self.f_star_sm[0].device, self.f_star_sm[0].dtype
    dim = self.f_star_sm[0].numel()
    obs_cov = torch.zeros((dim, dim), device=device, dtype=dtype)
    obs_y, obs_f = [], []
    transition_y, transition_f = [], []
    transition_cov_y = torch.zeros_like(obs_cov)
    transition_cov_f = torch.zeros_like(obs_cov)
    transition_cov_cross = torch.zeros_like(obs_cov)
    observation_mass = 0.0
    transition_mass = 0.0

    for position, index in enumerate(self.indexes[:limit]):
        weight = float(resp[index].item())
        if weight <= 0:
            continue
        root = weight ** 0.5
        mean = self.f_star_sm[position + 1].reshape(-1, 1)
        covariance = self.cov_f_sm[position + 1]
        observation = self.y_train[position].reshape(-1, 1).to(device=device, dtype=dtype)
        if not torch.equal(self.x_train[position], self.x_basis):
            raise NotImplementedError('conjugate_batch requires a common observation grid')
        if observation.shape[0] != prior_obs.get_mean().shape[0]:
            raise ValueError('conjugate_batch observation dimension differs from the LDS grid')
        obs_y.append(root * observation)
        obs_f.append(root * mean)
        obs_cov = obs_cov + weight * covariance
        observation_mass += weight

        if position == 0 or torch.count_nonzero(self.Gamma[0]) == 0:
            continue
        previous_mean = self.f_star_sm[position].reshape(-1, 1)
        previous_cov = self.cov_f_sm[position]
        filtered_previous_cov = self.cov_f[position]
        A_q, Gamma_q = self.A[position], self.Gamma[position]
        predicted_cov = A_q @ filtered_previous_cov @ A_q.T + Gamma_q
        gain = torch.cholesky_solve(A_q @ filtered_previous_cov.T,
                                    self._chol_spd(predicted_cov)).T
        transition_y.append(root * mean)
        transition_f.append(root * previous_mean)
        transition_cov_y = transition_cov_y + weight * covariance
        transition_cov_f = transition_cov_f + weight * previous_cov
        transition_cov_cross = transition_cov_cross + weight * covariance @ gain.T
        transition_mass += weight

    if not obs_y:
        return
    zeros = torch.zeros_like(obs_cov)
    observations = torch.cat(obs_y, dim=1)
    latent_observations = torch.cat(obs_f, dim=1)
    if isinstance(prior_obs, inv_wishart):
        C_fixed = prior_obs.get_C()
        residual = observations - C_fixed @ latent_observations
        scatter = residual @ residual.T + C_fixed @ obs_cov @ C_fixed.T
        new_n0 = prior_obs.n0 + observation_mass
        new_scale = ((prior_obs.n0 - 2) * prior_obs.scale + scatter) / (new_n0 - 2)
        new_obs = inv_wishart(new_n0, 0.5 * (new_scale + new_scale.T), C_fixed)
    else:
        new_obs = prior_obs.posterior(
            observation_mass, observations, latent_observations,
            zeros, obs_cov, zeros)
    new_int = prior_int
    if transition_y:
        new_int = prior_int.posterior(
            transition_mass, torch.cat(transition_y, dim=1),
            torch.cat(transition_f, dim=1), transition_cov_y,
            transition_cov_f, transition_cov_cross)

    self.observation_params = new_obs
    self.internal_params = new_int
    A_new = new_int.get_mean()
    Gamma_new = _lds_scale(self, new_int, final=True) if transition_y else self.Gamma[-1]
    C_new = new_obs.get_mean()
    Sigma_new = _lds_scale(self, new_obs, final=True)
    if self.annealing:
        mass = max(observation_mass, 1.0)
        if transition_y:
            Gamma_new = Gamma_new + self.Gamma_def / (mass ** 2)
        Sigma_new = Sigma_new + self.Sigma_def / (mass ** 2)
    self.A[-1], self.Gamma[-1] = A_new, Gamma_new
    self.C[-1], self.Sigma[-1] = C_new, Sigma_new
    self.var[-1] = torch.atleast_2d(torch.sqrt(torch.diag(Gamma_new).clamp_min(0))).T
    self.y_var[-1] = torch.atleast_2d(torch.sqrt(torch.diag(Sigma_new).clamp_min(0))).T


def bayesian_new_params(self, h, model_type='dynamic', full_data=False, q=None, force=False,
                        snr=1.0, first_observation=False):
    """ Method to compute the variational Bayesian step for LDS parameters. Can deal with dynamic or static models.
        Can use full data batch n-step estimation or 1-step estimation.
        Online first_observation updates C/Sigma from one beat without an A/Gamma transition.
    """
    if len(torch.nonzero(self.Gamma[-1]))< 1:
        model_type = 'static'
    first_observation_update = (first_observation and model_type == 'dynamic'
                                and self.N == 1 and not full_data)
    if h > 0.0:
        if snr > 0.5:
            if ((full_data and 1 < self.N) or 1 < self.N < self.estimation_limit
                    or force or first_observation_update):
                failure_stage = 'internal predictive covariance'
                try:
                    # if not full_data:
                    #     self.backwards()
                    if model_type == 'dynamic' and not first_observation_update:
                        if not full_data:
                            A, Gamma = self.A[-1], self.Gamma[-1]
                            samples_A = self.f_star_sm[-1]
                            samples_A_ = self.f_star_sm[-2]
                            cov = self.cov_f_sm[-1]
                            cov_ = self.cov_f_sm[-2]
                            if self.lds_update == 'legacy':
                                # Preserve the historical mean-only update.
                                cov = torch.zeros_like(cov)
                                cov_ = torch.zeros_like(cov_)
                                cov_cross = torch.zeros_like(cov)
                            else:
                                # E[f_t f_t^T], E[f_{t-1} f_{t-1}^T], and
                                # E[f_t f_{t-1}^T] use the current smoothed pair.
                                # The RTS gain uses the filtered previous
                                # covariance, before the current beat was seen.
                                filtered_prev = self.cov_f[-2]
                                predicted = A @ filtered_prev @ A.T + Gamma
                                gain = torch.cholesky_solve(
                                    A @ filtered_prev.T, self._chol_spd(predicted)).T
                                cov_cross = cov @ gain.T
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
                            N_k = h
                        elif self.estimation_limit != np.PINF:
                            N_k = self.estimation_limit
                        else:
                            N_k = samples_A.shape[1]
                        if h < 1.0 and not full_data:
                            root_h = np.sqrt(h)
                            samples_A, samples_A_ = samples_A * root_h, samples_A_ * root_h
                            cov, cov_, cov_cross = cov * h, cov_ * h, cov_cross * h
                        failure_stage = 'internal MNIW posterior'
                        new_int_dist = _lds_posterior(
                            self, self.internal_params, N_k, samples_A, samples_A_,
                            cov, cov_, cov_cross, annealing=self.annealing,
                        )
                    elif model_type == 'dynamic':
                        # One beat has an observation but no state transition.
                        N_k = h
                        new_int_dist = self.internal_params
                    elif model_type == 'static':
                        N_k = h
                        new_int_dist = self.internal_params
                    else:
                        print("Only programmed static and dynamic models.")
                        new_int_dist = self.internal_params
                    if not full_data:
                        samples_C = self.y_train[-1]
                        if torch.equal(self.x_basis, self.x_train[-1]):
                            samples_C_ = self.f_star_sm[-1]
                            cov_ = self.cov_f_sm[-1]
                        else:
                            samples_C_, cov_ = self.resample_latent_mean(self.x_train[-1])
                        if model_type == 'static':
                            samples_C_, cov_ = self.resample_latent_mean(self.x_train[-1])
                        samples_C_ = self.cond_to_cuda(samples_C_)
                        if self.lds_update == 'legacy':
                            cov_ = torch.zeros_like(cov_)
                        # The observation is known, so only its latent input
                        # contributes posterior covariance to the MNIW update.
                        cov = torch.zeros_like(cov_)
                        cov_cross = torch.zeros_like(cov_)
                    else:
                        samples_C  = torch.stack(self.y_train[:n_f])[:, :, 0].T
                        samples_C_ = torch.stack(self.f_star_sm[1:n_f+1])[:, :, 0].T
                        cov_ = torch.sum(torch.stack(self.cov_f_sm[1:n_f+1]), axis=0)
                        cov = torch.zeros(cov_.shape, device=cov_.device)
                        C, Sigma = self.C[-1], self.Sigma[-1]
                        cov_cross = torch.zeros_like(cov_)
                        failure_stage = 'observation predictive covariance'
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
                    failure_stage = 'observation projection'
                    if torch.equal(self.x_basis, self.x_train[-1]):
                        project_mat = None
                    else:
                        project_mat = self.reduce_noise_matrix(self.x_basis, self.x_train[-1])
                    if h < 1.0 and not full_data:
                        root_h = np.sqrt(h)
                        samples_C, samples_C_ = samples_C * root_h, samples_C_ * root_h
                        cov, cov_, cov_cross = cov * h, cov_ * h, cov_cross * h
                    failure_stage = 'observation MNIW posterior'
                    new_obs_dist = _lds_posterior(
                        self, self.observation_params, N_k, samples_C, samples_C_,
                        cov, cov_, cov_cross, sse_matrix=project_mat,
                        annealing=self.annealing,
                    )
                except torch.linalg.LinAlgError as exc:
                    failures = getattr(self, '_lds_update_failures', 0) + 1
                    self._lds_update_failures = failures
                    if failures == 1 or failures & (failures - 1) == 0:
                        print(f"LDS update skipped at {failure_stage}: "
                              f"responsibility={h:.3g}, included={self.N}, "
                              f"failures={failures}; {exc}", flush=True)
                    new_int_dist = self.internal_params
                    new_obs_dist = self.observation_params
            else:
                new_int_dist = self.internal_params
                new_obs_dist = self.observation_params
            self.internal_params = new_int_dist
            self.observation_params = new_obs_dist
            if 1 < self.N or first_observation_update:
                Gamma_ = (self.Gamma[-1] if first_observation_update else
                          _lds_scale(self, new_int_dist, final=full_data))
                Sigma_ = _lds_scale(self, new_obs_dist, final=full_data)
            else:
                Gamma_ = self.Gamma[-1]
                Sigma_ = self.Sigma[-1]
            if self.annealing:
                if model_type == 'static':
                    factor_S = self.Sigma[0] / (self.N ** 2)
                    factor_G = self.Gamma[0]
                else:
                    factor_G = (torch.zeros_like(self.Gamma[0]) if first_observation_update
                                else self.Gamma[0] / (self.N ** 2))
                    factor_S = self.Sigma[0] / (self.N ** 2)
                Gamma_ = Gamma_ + factor_G
                Sigma_ = Sigma_ + factor_S
            if self.N < self.estimation_limit or full_data or first_observation_update:
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
            Gamma_ = _lds_scale(self, new_int_dist, final=full_data)
            Sigma_ = _lds_scale(self, new_obs_dist, final=full_data)
            self.A.append(new_int_dist.get_mean())
            self.Gamma.append(Gamma_)
            if model_type == 'static':
                self.C.append(new_obs_dist.get_C())
            else:
                self.C.append(new_obs_dist.get_mean())
            self.Sigma.append(Sigma_)
            self.var.append(torch.atleast_2d(torch.sqrt(torch.diag(Gamma_))).T)
            self.y_var.append(torch.atleast_2d(torch.sqrt(torch.diag(Sigma_))).T)
