"""Native offline fitting, classification and label-conditioned fitting schedules.
Keep the global/local alternation and mutation order unchanged."""

import numpy as np
import torch
from ..alignment import _components as _alignment


def _log(self, *args, **kwargs):
    """Print offline diagnostics only when the model's verbose mode is enabled."""
    if self.verbose:
        print(*args, **kwargs)


def fit(self, x_trains, y_trains, it_limit=None, warp=False):
    """
    Include the batch of samples received and restimate all the parameters of the
    full model, including the state-space and the GPS.
    Parameters
    ----------
    x_trains : time indexes of each sample (nxs_samples
    y_trains : array sample of shape (nxs_samples)

    Returns
    -------
    self : returns an instance of self.
    """
    # # Redefine HDP hyperparams for batch inclusion
    # self.gamma = 0.1
    # self.transAlpha = 0.1
    # self.startAlpha = 0.1
    # self.kappa = 0.0
    self.warp = _alignment.configured(self).enabled(warp)
    _log(self, "------ HDP Hyperparameters ------", flush=True)
    _log(self, "gamma: " + str(self.gamma))
    _log(self, "transAlpha: " + str(self.transAlpha))
    _log(self, "startAlpha: " + str(self.startAlpha))
    _log(self, "kappa: " + str(self.kappa))
    _log(self, "---------------------------------", flush=True)
    if self.reduce_outputs:
        y_trains = self.reduce_num_outputs(y_trains)
    n_samples = np.array(y_trains).shape[0]
    n_outputs = np.array(y_trains).shape[2]
    t = self.T
    self.T = self.T + n_samples
    # Compute SNR to ponderate q on each lead.
    self.compute_snr_ini(y_trains)
    M = self.M
    y_trains = self.cond_cuda(self.cond_to_torch(y_trains))
    x_trains = self.cond_cuda(self.cond_to_torch(x_trains))
    self.x_train = self.cond_cuda(self.cond_to_torch(x_trains))
    iteration = 0
    reparam = True
    # Start loop of EM
    elbo = -np.inf
    resp = self.cond_cuda(torch.zeros((n_samples, M)))
    respPair = self.cond_cuda(torch.zeros((n_samples, M, M)))
    respPair[:, 0, 0] = respPair[:, 0, 0] + 1.0
    q = self.cond_cuda(torch.zeros((n_samples, M, n_outputs)))
    q_lat = self.cond_cuda(torch.zeros((n_samples, M, n_outputs)))
    snr = self.snr_norm
    resp[:, 0] = resp[:, 0] + 1.0
    y_trains_w = torch.clone(y_trains)
    warp_computed = False
    previous_elbo = None
    # Initial estimation of full group and conditioning the covariance matrices.
    if self.reestimate_initial_params:
        self.redefine_default(x_trains, y_trains, resp)
    startStateCount = None
    transStateCount = None
    reallocate = False
    while True:
        resp, respPair, q, q_lat, snr, end = self.refill(resp, respPair, startStateCount, transStateCount, q, q_lat,
                                                         snr)
        M = self.M
        if resp.shape[1] == 1:
            startStateCount = self.cond_cuda(resp[0])
            transStateCount = self.cond_cuda(torch.sum(respPair, axis=0))
            self.reinit_global_params(M, transStateCount, startStateCount)
            nIters = 2
            for giter in range(nIters):
                self.transTheta, self.startTheta = self._calcThetaFull(self.cond_cuda(transStateCount),
                                                                       self.cond_cuda(startStateCount), M + 1)
                self.rho, self.omega = self.find_optimum_rhoOmega()
        if end:
            break
        self._iteration_birth_candidates = []
        resp, respPair, q, q_lat, snr, y_trains_w, reallocate = self.variational_local_terms_batch(M, x_trains,
                                                                                                   y_trains,
                                                                                                   y_trains_w,
                                                                                                   self.transTheta,
                                                                                                   self.startTheta,
                                                                                                   resp, respPair,
                                                                                                   q, q_lat, snr,
                                                                                                   reallocate)
        if resp.shape[1] > M:
            self.M = M + 1
            M = self.M
        resp__ = torch.clone(resp)
        respPair__ = torch.clone(respPair)
        # Update HDP variational params.
        if self.hmm_switch:
            startStateCount = self.cond_cuda(resp__[0])
            transStateCount = self.cond_cuda(torch.sum(respPair__, axis=0))
        else:
            transStateCount = self.cond_cuda(torch.ones((M + 1, M + 1)))  # Cause we are in independent observations
            startStateCount = self.cond_cuda(torch.ones(M + 1))

        self.reinit_global_params(M, transStateCount, startStateCount)
        nIters = 2
        for giter in range(nIters):
            self.transTheta, self.startTheta = self._calcThetaFull(self.cond_cuda(transStateCount),
                                                                   self.cond_cuda(startStateCount), M + 1)
            self.rho, self.omega = self.find_optimum_rhoOmega()

        # Update transition matrix
        transPi = self.hdp.stored_log_transitions(M, self.transTheta)
        self.trans_A = transPi
        resp_group = torch.sum(resp, dim=0).detach().cpu().tolist()
        if not self.verbose:
            print(f"Iteration {iteration + 1}: responsibility sums={resp_group}", flush=True)
        # Compute ELBO and start steps of EM
        # TODO: adapt this option to full cuda implementation.
        if self.T > 1:
            elbo_ = self.calcELBO_NonlinearTerms(resp=self.cond_to_numpy(self.cond_cpu(resp)),
                                                 respPair=self.cond_to_numpy(self.cond_cpu(respPair)))
            _log(self, '\n-------End Lower Bound Iteration ' + str(iteration) + '-------')
            if iteration == 14:
                _log(self, "STOP")
            q_obs, elbo_lin = self.compute_q_elbo(resp, respPair, self.weight_mean(q), self.weight_mean(q_lat),
                                                  self.gpmodels, self.M, snr='saved', post=False)
            elbo_ = elbo_ + elbo_lin + q_obs
            if self.verbose:
                delta = "n/a" if previous_elbo is None else f"{(elbo_ - previous_elbo).item():+.6g}"
                proposed = getattr(self, '_iteration_birth_candidates', [])
                print(f"Iteration {iteration + 1}: clusters={self.M}, "
                      f"responsibility sums={resp_group}, "
                      f"proposed birth samples (1-based)={[i + 1 for i in proposed]}, "
                      f"ELBO={elbo_.item():.6g}, delta_ELBO={delta}", flush=True)
            previous_elbo = elbo_
            del self._iteration_birth_candidates
            iteration = iteration + 1
            if it_limit is not None and iteration >= it_limit:
                self.train_elbo.append(elbo_)
                self.resp_assigned.append(torch.where(resp == 1.0)[1])
                break
            if self.M == self.max_models:
                self.train_elbo.append(elbo_)
                self.resp_assigned.append(torch.where(resp == 1.0)[1])
                break
            resp_group = torch.sum(resp, dim=0)
            self.train_elbo.append(elbo_)
            self.resp_assigned.append(torch.argmax(resp, axis=1))
            self.q.append(q)
            self.q_last, self.q_lat_last, self.snr_last = q, q_lat, snr
            self.startStateCount_last, self.transStateCount_last = startStateCount, transStateCount
            self.resp_last, self.respPair_last = resp, respPair
            self.elbo_last = elbo_
            if (torch.where(resp_group == 0.0)[0].shape[0] > 1.0 or
                    (len(self.resp_assigned) > 1 and (
                            self.resp_assigned[-2].shape[0] == self.resp_assigned[-1].shape[0])
                     and torch.all(self.resp_assigned[-2] == self.resp_assigned[-1]))):
                break
            else:
                elbo = elbo_
        else:
            break


def variational_local_terms_batch(self, M, x_trains, y_trains, y_trains_w, transTheta, startTheta, resp, respPair,
                                  q, q_lat, snr, reallocate):
    """
    Compute variational terms of the SLDS for a batch of examples. Iterates over q_first and q_all until convergence is reached.
    Parameters
    ----------
    y : vector double
        observation vector (or warped versions).
    liks : vector, optional
        likelihoods of the warpings.
    """
    # Parameters of model
    t = self.T - 1
    # Compute the initial values
    if transTheta is None:
        transTheta = self.transTheta
    if startTheta is None:
        startTheta = self.startTheta
    transPi = self.hdp.batch_log_transitions(M, transTheta)
    self.trans_A = transPi
    # Calculate LOG of start state prob vector
    startPi = self.hdp.batch_log_start(M, startTheta)
    i = 0
    reparam = True
    resp_per_group = torch.sum(resp, dim=0)
    if resp_per_group.shape[0] == 1.0 or resp_per_group[-2] >= 1.0 or not self.gpmodels[0][0].fitted:
        resp, respPair, q, q_lat, snr, y_trains_w, reallocate = self.estimate_q_first(M, x_trains=x_trains, y_trains=y_trains,
                                     y_trains_w_=y_trains_w, resp=resp, respPair=respPair, q_=q, q_lat_=q_lat, snr_=snr,
                                     startPi=startPi, transPi=transPi, reallocate_=reallocate, reparam=reparam)
        if resp.shape[1] > self.M:
            q_bas, elbo_bas = self.compute_q_elbo(resp, respPair, self.weight_mean(q), self.weight_mean(q_lat),
                                                  self.gpmodels, self.M, snr='saved', post=True)
        else:
            q_bas, elbo_bas = self.compute_q_elbo(resp, respPair, self.weight_mean(q), self.weight_mean(q_lat),
                                                  self.gpmodels, self.M, snr='saved', post=False)
        i = i + 1
        _log(self, "First resp: " + str(torch.sum(resp, dim=0).int().numpy()))
    else:
        q_bas, elbo_bas = self.compute_q_elbo(resp, respPair, self.weight_mean(q), self.weight_mean(q_lat),
                                              self.gpmodels, self.M, snr='saved', post=False)
        _log(self, "Not first estimated q.")
    q_def__, elbo_def__ = torch.clone(q_bas), torch.clone(elbo_bas)
    if not reallocate:
        while True:
            M = resp.shape[1]
            resp, respPair, q, q_lat, snr, y_trains_w, gpmodels = self.estimate_q_all( M, x_trains=x_trains, y_trains=y_trains, y_trains_w_=y_trains_w,
                                           resp=resp, respPair=respPair, q_=q, q_lat_=q_lat, snr_=snr, startPi=startPi, transPi=transPi, q_def=q_def__, elbo_def=elbo_def__,
                                           reparam=reparam)
            self.gpmodels = gpmodels
            if resp.shape[1] > self.M:
                q_post, elbo_post = self.compute_q_elbo(resp, respPair, self.weight_mean(q),
                                                        self.weight_mean(q_lat), self.gpmodels, self.M, snr='saved',
                                                        post=True)
            else:
                q_post, elbo_post = self.compute_q_elbo(resp, respPair, self.weight_mean(q),
                                                        self.weight_mean(q_lat), self.gpmodels, self.M, snr='saved',
                                                        post=False)
            _log(self, "ELBO_reduction: " + str(((q_post + elbo_post) - (q_bas + elbo_bas)).item()))
            if (torch.isclose(q_bas + elbo_bas, q_post + elbo_post, rtol=1e-5) and i > 0) or i == 10:  # or reparam:
                if not reparam:
                    reparam = True
                else:
                    break
            q_bas = q_post
            elbo_bas = elbo_post
            i = i + 1
    # q_obs = torch.sum(torch.sum(q_)).item()/len(y_trains)
    # self.update_initial_sigma()
    return resp, respPair, q, q_lat, snr, y_trains_w, reallocate


def estimate_q_first(self, M, x_trains, y_trains, y_trains_w_, resp, respPair, q_, q_lat_, snr_, startPi, transPi, reallocate_=False, reparam=False):
    """ Smart and complex computation of new group assignation and compare with the last iteration then decides if generate group or not.
        First, it computes if a reallocation of the samples is better, if not, proposes a new birth group and computes the ELBO to measure the goodness.
    """
    empty_estimation = False
    y_trains_w, x_w, liks = self.warp_batch_by_resp_amtgp_cached(
        x_trains=x_trains,
        y_trains=y_trains,
        resp_temp=resp,
        f_ind_old=self.f_ind_old,
    )
    if torch.mean(q_) == 0.0:
        snr_ = torch.zeros(y_trains.shape[0], M, self.n_outputs)
        for ld in range(self.n_outputs):
            if not self.share_gp:
                gp = self.create_gp_default()
            else:
                if ld == 0:
                    gp = self.create_gp_default()
                else:
                    gp = self.gpmodel_deepcopy(self.gpmodels[ld - 1][0])
                    if gp.fitted:
                        gp.reinit_LDS(save_last=False)
                        gp.reinit_GP(save_last=False)
            q_[:, 0, ld], q_lat_[:, 0, ld]= gp.full_pass_weighted(x_trains, y_trains_w[:, :, [ld], 0], resp[:, 0],
                                                 snr=self.snr_norm[:, ld])
            q_[:, 0, ld] = q_[:, 0, ld] + liks[:, 0, ld]
            snr_[:, 0, ld] = self.compute_snr(y_trains_w[:, :, ld, 0], gp)
            self.gpmodels[ld][0] = gp
    reallocate = False

    q_simple = torch.clone(q_)

    indexes_ = []
    for m in range(M):
        indexes_.append(torch.tensor(self.gpmodels[0][m].indexes).long())
        if indexes_[m].shape[0] == 0:
            indexes_[m] = torch.where(resp[:, m] == self.cond_cuda(torch.tensor(1.0)))[0].long()
    f_ind_old = torch.clone(self.f_ind_old)

    # Compute q with best index of the old model.
    snr_temp = torch.zeros(y_trains.shape[0], M, self.n_outputs)
    for ld in range(self.n_outputs):
        if self.verbose:
            _log(self, "\n-----------Lead " + str(ld + 1) + "-----------")
        for m in range(M):
            gp = self.gpmodel_deepcopy(self.gpmodels[ld][m])
            if gp.fitted:
                gp.reinit_LDS(save_last=False)
                gp.reinit_GP(save_last=False)
            if len(indexes_[m]) > 0:
                gp.include_weighted_sample(0, x_trains[f_ind_old[m]], x_trains[f_ind_old[m]], y_trains[f_ind_old[m],:,[ld]], h=1.0)
            q_simple[:, m, ld] = gp.compute_sq_err_all(x_trains, y_trains_w[:,:, [ld],m])
            q_simple[:, m, ld] = q_simple[:, m, ld] + liks[:, m, ld]
            snr_temp[:, m, ld] = self.compute_snr(y_trains[:,:,ld], gp)

    if M > 1:
        q_aux = torch.clone(q_simple)
        snr_aux = torch.clone(snr_)
        if torch.sum(resp, dim=0)[-1] == 0:
            q_aux[:, -1, :] = torch.zeros(q_aux[:, -1, :].shape) + torch.min(q_aux) * 2.0
            snr_aux[:, -1, :] = torch.zeros(snr_aux[:, -1, :].shape) + torch.min(snr_aux) * 2.0

        q_norm, _ = self.LogLik(self.weight_mean(q_aux, snr_aux))
        alpha, margprob = self.forward(startPi, transPi, q_norm)
        beta = self.backward(transPi, q_norm, margprob)
        resplog_temp, _ = self.LogLik(torch.log(alpha * beta), axis=1)
        respPairlog_temp, _ = self.LogLik(self.coupled_state_coef(alpha, beta, transPi, q_norm, margprob), axis=1)
        resp_temp = self._safe_exp(resplog_temp)
        respPair_temp = self._safe_exp(respPairlog_temp)
        # resp_temp, respPair_temp = self.refill_resp(resp_temp, respPair_temp)

        resp_per_group_temp = torch.sum(resp_temp, axis=0)
        reorder = torch.argsort(resp_per_group_temp, descending=True)
        resp_temp = resp_temp[:, reorder]

        # First, try to reallocate beats, if this not works then generate new group.
        q = torch.clone(q_)
        q_lat = torch.clone(q_lat_)
        gpmodels_temp = [[] for _ in range(self.n_outputs)]
        for ld in range(self.n_outputs):
            if self.verbose:
                _log(self, "\n-----------Lead " + str(ld + 1) + "-----------")
            for m in range(M):
                if self.verbose:
                    _log(self, "\n   -----------Model " + str(m + 1) + "-----------")
                if not torch.equal(resp[:, reorder[m]].long(), resp_temp[:, m].long()):
                    gp = self.gpmodel_deepcopy(self.gpmodels[ld][reorder[m]])
                    if gp.fitted:
                        gp.reinit_LDS(save_last=False)
                        gp.reinit_GP(save_last=False)
                    q[:, m, ld], q_lat[:, m, ld] = gp.full_pass_weighted(x_trains, y_trains_w[:,:,[ld],reorder[m]],
                                                        resp_temp[:, m], q=q[:,reorder[m], ld],
                                                        q_lat=q_lat[:, reorder[m], ld],
                                                        snr=self.snr_norm[:,ld])
                    q[:, m, ld] = q[:, m, ld] + liks[:, reorder[m], ld]
                    snr_aux[:, m, ld] = self.compute_snr(y_trains_w[:, :, ld, reorder[m]], gp)
                else:
                    gp = self.gpmodels[ld][reorder[m]]
                    q[:, m, ld] = torch.clone(q_[:, reorder[m], ld])
                    snr_aux[:, m, ld] = torch.clone(snr_[:, reorder[m], ld])
                gpmodels_temp[ld].append(gp)

        q_bas_, elbo_bas_ = self.compute_q_elbo(resp_temp, respPair_temp, self.weight_mean(q, snr_aux),
                                                self.weight_mean(q_lat, snr_aux), gpmodels_temp, M,
                                                snr=snr_aux, post=False)
        q_def__, elbo_def__ = self.compute_q_elbo(resp, respPair, self.weight_mean(q_, snr_),
                                                  self.weight_mean(q_lat_, snr_),
                                                  self.gpmodels, M, snr=snr_, post=False)
        i__ = 0
        while True:
            resp_temp, respPair_temp, q, q_lat, snr_aux, y_trains_w, gpmodels_temp = self.estimate_q_all(M,
                                                                                                         x_trains=x_trains,
                                                                                                         y_trains=y_trains,
                                                                                                         y_trains_w_=y_trains_w,
                                                                                                         resp=resp_temp,
                                                                                                         respPair=respPair_temp,
                                                                                                         q_=q,
                                                                                                         q_lat_=q_lat,
                                                                                                         snr_=snr_aux,
                                                                                                         startPi=startPi,
                                                                                                         transPi=transPi,
                                                                                                         q_def=q_def__,
                                                                                                         elbo_def=elbo_def__,
                                                                                                         gpmodels=gpmodels_temp,
                                                                                                         reparam=reparam,
                                                                                                         post=False)
            q_post, elbo_post = self.compute_q_elbo(resp_temp, respPair_temp, self.weight_mean(q, snr_aux),
                                                    self.weight_mean(q_lat, snr_aux), gpmodels_temp, M, snr=snr_aux,
                                                    post=False)
            _log(self, "ELBO_reduction: " + str(((q_post + elbo_post) - (q_bas_ + elbo_bas_)).item()))
            if (torch.isclose(q_bas_ + elbo_bas_, q_post + elbo_post,
                              rtol=1e-5) and i__ > 0) or i__ == 20:  # or reparam:
                break
            q_bas_ = q_post
            elbo_bas_ = elbo_post
            i__ = i__ + 1

        new_indexes = torch.where(torch.sum(np.abs(resp - resp_temp), dim=1) > 1.0)[0]
        _log(self, ">>> Prev -------")
        q_bas, elbo_bas = self.compute_q_elbo(resp, respPair, self.weight_mean(q_, snr_),
                                              self.weight_mean(q_lat_, snr_),
                                              self.gpmodels, M, snr=snr_, post=False)
        _log(self, ">>> Post -------")
        q_bas_post, elbo_post = self.compute_q_elbo(resp_temp, respPair_temp, self.weight_mean(q, snr_aux),
                                                    self.weight_mean(q_lat, snr_aux),
                                                    gpmodels_temp, M, snr=snr_aux, post=False)
        update_snr = True
        if (torch.where(torch.sum(resp_temp, dim=0) < 1.0)[0].shape[0] == 0):
            if q_bas < q_bas_post:
                if not q_bas + elbo_bas < q_bas_post + elbo_post:
                    _log(self, "Possibly better q_obs but worse elbo.")
            if q_bas + elbo_bas < q_bas_post + elbo_post and q_bas != q_bas_post:
                _log(self, "Reallocating beats into existing groups.")
                reallocate = True
                self.gpmodels = gpmodels_temp
                self.x_w = x_w[:, :, :, reorder]
                self.liks = liks[:, reorder, :]
                y_trains_w = y_trains_w[:, :, :, reorder]
                self.y_w = y_trains_w
                self.y_train = self.select_assigned_warp(y_trains_w, resp_temp)
                f_ind_old_new = torch.full_like(f_ind_old, -1)
                used_representers = set()

                for k in range(M):
                    # Samples assigned to cluster k
                    indexes_k = torch.where(resp_temp[:, k] == 1.0)[0]

                    # Rank candidates by weight, best first
                    order = torch.argsort(self.weight_mean(q_simple, snr_aux)[indexes_k, k],
                                          descending=True)
                    sorted_candidates = indexes_k[order]

                    candidate = None
                    for idx in sorted_candidates:
                        idx_val = int(idx.item())
                        if idx_val not in used_representers:
                            candidate = idx_val
                            break

                    # Fallback if all candidates were already used
                    if candidate is None:
                        candidate = int(sorted_candidates[0].item())

                    f_ind_old_new[k] = candidate
                    used_representers.add(candidate)
                self.f_ind_old = torch.clone(f_ind_old_new)

                if update_snr:
                    self.snr_norm = self.normalize_snr(snr_aux)
                else:
                    snr_aux = snr_
                return resp_temp, respPair_temp, q, q_lat, snr_aux, y_trains_w, reallocate
            else:
                _log(self, "Not reallocating, trying to generate new group.")
        else:
            _log(self, ">>> Possible emergency reallocation. Prev ----")
            q_bas, elbo_bas = self.compute_q_elbo(resp, respPair, self.weight_mean(q_, snr_),
                                                  self.weight_mean(q_lat_, snr_), self.gpmodels, self.M,
                                                  snr=snr_, post=False)
            if q_bas + elbo_bas < q_bas_ + elbo_bas_:
                _log(self, "Emergency reallocation and removing last group.")
                reallocate = True
                for ld in range(self.n_outputs):
                    gpmodels_temp[ld] = gpmodels_temp[ld][:-1]
                self.gpmodels = gpmodels_temp
                if update_snr:
                    self.snr_norm = self.normalize_snr(snr_aux)
                resp_temp, respPair_temp, q, q_lat, snr_aux = self.remove_last_group(resp_temp,
                                                                                     respPair_temp, q,
                                                                                     q_lat, snr_aux)
                resp_per_group_temp = torch.sum(resp_temp[torch.where(resp_temp == 1.0)[0]], axis=0)
                reorder = torch.argsort(resp_per_group_temp, descending=True)
                self.f_ind_old = self.f_ind_old[reorder]
                return resp_temp, respPair_temp, q, q_lat, snr_aux, y_trains_w, reallocate
            else:
                _log(self, "Bad estimation")
                empty_estimation = False
    # f_ind_new_potential = torch.argsort(self.weight_mean(q_simple)[torch.where(resp == 1.0)])
    q_sim_s = self.weight_mean(q_simple)[torch.where(resp == 1.0)]
    q_sim_s = (q_sim_s - torch.max(q_sim_s)) / (torch.max(q_sim_s) - torch.min(q_sim_s))
    q_s = self.weight_mean(q_)[torch.where(resp == 1.0)]
    q_s = (q_s - torch.max(q_s)) / (torch.max(q_s) - torch.min(q_s))
    q_lat_s = self.weight_mean(q_lat_)[torch.where(resp == 1.0)]
    q_lat_s = (q_lat_s - torch.max(q_lat_s)) / (torch.max(q_lat_s) - torch.min(q_lat_s))
    f_ind_new_potential = torch.argsort(q_sim_s)
    q_rank = q_sim_s
    potential_weight = torch.zeros(f_ind_new_potential.shape[0])
    potential_ind = {}
    potential_q = torch.zeros(f_ind_new_potential.shape[0])
    for j, ind in enumerate(f_ind_new_potential):
        potential_ind[ind.item()] = torch.where(torch.isclose(q_rank, q_rank[ind], rtol=0.01))[0]
        potential_weight[ind] = torch.where(torch.isclose(q_rank, q_rank[ind], rtol=0.01))[0].shape[0]
        potential_q[ind] = torch.sum(q_rank[potential_ind[ind.item()]])
    n_steps = self.n_explore_steps
    f_ind_new_potential_def = torch.zeros(n_steps).long()
    last_indexes = torch.tensor([-1])
    j_ = 0
    for j, f_ind_new in enumerate(f_ind_new_potential):
        if j_ == np.max([n_steps // 2.0, 1]):
            # if j_ == n_steps:
            break
        m_chosen = -1
        for m in range(M - 1):
            if f_ind_new in indexes_[m]:
                m_chosen = m
                break
        if m_chosen == -1:
            m_chosen = torch.argmax(resp[f_ind_new])
        f_ind_old_chosen = f_ind_old[m_chosen]
        if f_ind_new != f_ind_old_chosen:
            for l_ in last_indexes:
                if not l_ in potential_ind[f_ind_new.item()]:
                    last_indexes = potential_ind[f_ind_new.item()]
                    f_ind_new_potential_def[j_] = f_ind_new
                    j_ = j_ + 1
                    break

    # Potential compared with accumulated q -------------------------
    q_aux = torch.clone(q_simple)
    f_ind_new_q = torch.argsort(q_s + q_lat_s)
    last_indexes = torch.tensor([-1])
    j_ = int(np.max([n_steps // 2.0, 1]))
    for j, f_ind_new in enumerate(f_ind_new_q):
        if j_ == n_steps:
            break
        m_chosen = -1
        for m in range(M - 1):
            if f_ind_new in indexes_[m]:
                m_chosen = m
                break
        if m_chosen == -1:
            m_chosen = torch.argmax(resp[f_ind_new])
        f_ind_old_chosen = f_ind_old[m_chosen]
        if f_ind_new != f_ind_old_chosen:
            for l_ in last_indexes:
                if not l_ in potential_ind[f_ind_new.item()]:
                    last_indexes = potential_ind[f_ind_new.item()]
                    f_ind_new_potential_def[j_] = f_ind_new
                    j_ = j_ + 1
                    break
    # -----------------------------------------
    # ord_ = torch.argsort(potential_q[f_ind_new_potential_def[:n_steps]])#, descending=True)
    # f_ind_new_potential_def[:n_steps] = f_ind_new_potential_def[ord_]
    # Adding 5 possible potential indexes not by q.
    # f_ind_new_potential_def = torch.concatenate([f_ind_new_potential_def,f_ind_new_q_def])
    # n_steps = n_steps + 5
    step = 0
    last_indexes = torch.tensor([-1])
    q = torch.clone(q_aux)
    q_lat = torch.clone(q_lat_)
    snr_aux = torch.clone(snr_)
    resp_, respPair_, q_def, q_lat_def, snr_aux_def = self.new_group(resp, respPair, torch.clone(q),
                                                                     torch.clone(q_lat), torch.clone(snr_aux))
    _, _, q__def, q_lat__def, snr__def = self.new_group(resp, respPair, torch.clone(q_),
                                                        torch.clone(q_lat_), torch.clone(snr_))
    M = M + 1
    f_ind_old = torch.zeros(M, device=resp.device).long()
    f_ind_old[:self.f_ind_old.shape[0]] = torch.clone(self.f_ind_old)
    for j, f_ind_new in enumerate(f_ind_new_potential_def):
        if step == n_steps:
            break
        m_chosen = -1
        for m in range(M - 1):
            if f_ind_new in indexes_[m]:
                m_chosen = m
                break
        if m_chosen == -1:
            m_chosen = torch.argmax(resp[f_ind_new])
        f_ind_old_chosen = f_ind_old[m_chosen]
        ld_belong = torch.argmax(self.snr_norm[f_ind_new])
        if f_ind_new != f_ind_old_chosen:
            some_new_index = False
            for l_ in last_indexes:
                if not l_ in potential_ind[f_ind_new.item()]:
                    some_new_index = True
                    break
            if some_new_index:
                if hasattr(self, '_iteration_birth_candidates'):
                    self._iteration_birth_candidates.append(int(f_ind_new.item()))
                if not empty_estimation:
                    f_ind_old_temp = f_ind_old.clone()
                    f_ind_old_temp[-1] = f_ind_new
                    y_trains_w, x_w, liks = self.warp_batch_by_resp_amtgp_cached(
                        x_trains=x_trains,
                        y_trains=y_trains,
                        resp_temp=resp_,
                        f_ind_old=f_ind_old_temp)
                    q_simple_ = torch.clone(q_def)
                    q, q_lat, snr_aux = torch.clone(q_def), torch.clone(q_lat_def), torch.clone(snr_aux_def)
                    q__, q_lat__, snr__ = torch.clone(q__def), torch.clone(q_lat__def), torch.clone(snr__def)
                    last_indexes = potential_ind[f_ind_new.item()]
                    _log(self, "Step " + str(step + 1) + "/" + str(n_steps) + "- Trying to divide: " + str(
                        m_chosen) + " with beat " + str(f_ind_new.item()))
                    step = step + 1
                    for ld in range(self.n_outputs):
                        gp = self.gpmodel_deepcopy(self.gpmodels[ld][m_chosen])
                        if gp.fitted:
                            gp.reinit_LDS(save_last=False)
                            gp.reinit_GP(save_last=False)
                        gp.include_weighted_sample(0, x_trains[f_ind_new], x_trains[f_ind_new],
                                                   y_trains[f_ind_new, :, [ld]], h=1.0)
                        q_simple_[:, -1, ld] = gp.compute_sq_err_all(x_trains, y_trains_w[:, :, [ld], -1])
                        q_simple_[:, -1, ld] = q_simple_[:, -1, ld] + liks[:, -1, ld]
                        snr_aux[:, -1, ld] = self.compute_snr(y_trains_w[:, :, ld, -1], gp)
                    # Compute resp
                    q_mean = self.weight_mean(q_simple_, snr_aux)
                    q_norm, _ = self.LogLik(q_mean)
                    alpha, margprob = self.forward(startPi, transPi, q_norm)
                    beta = self.backward(transPi, q_norm, margprob)
                    resplog_temp, _ = self.LogLik(torch.log(alpha * beta), axis=1)
                    respPairlog_temp, _ = self.LogLik(
                        self.coupled_state_coef(alpha, beta, transPi, q_norm, margprob), axis=1)
                    resp_temp = self._safe_exp(resplog_temp)
                    respPair_temp = self._safe_exp(respPairlog_temp)
                    # resp_temp, respPair_temp = self.refill_resp(resp_temp, respPair_temp)
                else:
                    q, q_lat, snr_aux = torch.clone(q__def), torch.clone(q_lat__def), torch.clone(snr__def)
                    q__, q_lat__, snr__ = torch.clone(q__def), torch.clone(q_lat__def), torch.clone(snr__def)
                    q[:, -1, :] = torch.zeros(q[:, -1, :].shape) + torch.min(q) * 2.0
                    q__[:, -1, :] = torch.zeros(q__[:, -1, :].shape) + torch.min(q__) * 2.0
                    snr_aux[:, -1, :] = torch.zeros(snr_aux[:, -1, :].shape) + torch.min(snr_aux) * 2.0
                    q__[f_ind_new, -1, :] = 0.0
                    q_mean = self.weight_mean(q__, snr_aux)
                    q_norm, _ = self.LogLik(q_mean)
                    alpha, margprob = self.forward(startPi, transPi, q_norm)
                    beta = self.backward(transPi, q_norm, margprob)
                    resplog_temp, _ = self.LogLik(torch.log(alpha * beta), axis=1)
                    respPairlog_temp, _ = self.LogLik(
                        self.coupled_state_coef(alpha, beta, transPi, q_norm, margprob), axis=1)
                    resp_temp = self._safe_exp(resplog_temp)
                    respPair_temp = self._safe_exp(respPairlog_temp)

                # Reallocating resp to preserve order.
                resp_per_group_temp = torch.sum(resp_temp, axis=0)
                reorder = torch.argsort(resp_per_group_temp, descending=True)
                # reorder = torch.arange(resp_per_group_temp.shape[0])
                resp_temp = resp_temp[:, reorder]

                # Compute chosen model conditioned on new resp
                # Update all models conditioned on new resp if it has changed
                gpmodels_temp = [[] for _ in range(self.n_outputs)]
                for ld in range(self.n_outputs):
                    for m in range(M):
                        if reorder[m] == M - 1:
                            if self.share_gp:
                                gp = self.gpmodel_deepcopy(self.gpmodels[ld][m_chosen])
                            else:
                                # If not share_gp then new GP is used for a new model, more expensive but official model.
                                gp = self.create_gp_default()
                            if gp.fitted:
                                gp.reinit_LDS(save_last=False)
                                gp.reinit_GP(save_last=False)
                            q[:, m, ld], q_lat[:, m, ld] = gp.full_pass_weighted(x_trains,
                                                                                 y_trains_w[:, :, [ld], reorder[m]],
                                                                                 resp_temp[:, m],
                                                                                 q=q__[:, reorder[m], ld],
                                                                                 q_lat=q_lat__[:, reorder[m], ld],
                                                                                 snr=self.snr_norm[:, ld])
                            q[:, m, ld] = q[:, m, ld] + liks[:, reorder[m], ld]
                            snr_aux[:, m, ld] = self.compute_snr(y_trains_w[:, :, ld, reorder[m]], gp)
                        else:
                            if not torch.equal(resp[:, reorder[m]].long(), resp_temp[:, m].long()):
                                gp = self.gpmodel_deepcopy(self.gpmodels[ld][reorder[m]])
                                if gp.fitted:
                                    gp.reinit_LDS(save_last=False)
                                    gp.reinit_GP(save_last=False)
                                q[:, m, ld], q_lat[:, m, ld] = gp.full_pass_weighted(x_trains,
                                                                                     y_trains_w[:, :, [ld], reorder[m]],
                                                                                     resp_temp[:, m],
                                                                                     q=q__[:, reorder[m], ld],
                                                                                     q_lat=q_lat__[:, reorder[m],
                                                                                           ld],
                                                                                     snr=self.snr_norm[:, ld])
                                q[:, m, ld] = q[:, m, ld] + liks[:, reorder[m], ld]
                                snr_aux[:, m, ld] = self.compute_snr(y_trains_w[:, :, ld, reorder[m]], gp)
                            else:
                                gp = self.gpmodels[ld][reorder[m]]
                                q[:, m, ld] = torch.clone(q__[:, reorder[m], ld])
                                q_lat[:, m, ld] = torch.clone(q_lat__[:, reorder[m], ld])
                                snr_aux[:, m, ld] = torch.clone(snr__[:, reorder[m], ld])
                        gpmodels_temp[ld].append(gp)

                update_snr = True

                q_bas_, elbo_bas_ = self.compute_q_elbo(resp_temp, respPair_temp, self.weight_mean(q, snr_aux),
                                                        self.weight_mean(q_lat, snr_aux), gpmodels_temp, M,
                                                        snr=snr_aux, post=True)
                if torch.argmax(
                        torch.sum(resp_temp, dim=0)).item() == resp_temp.shape[1] - 1:
                    _log(self, "Bad estimation")
                    continue
                if (torch.where(torch.sum(resp_temp, dim=0) < 1.0)[0].shape[0] > 0):
                    _log(self, ">>> Possible emergency reallocation. Prev ----")
                    q_bas, elbo_bas = self.compute_q_elbo(resp, respPair, self.weight_mean(q_, snr_),
                                                          self.weight_mean(q_lat_, snr_), self.gpmodels, self.M,
                                                          snr=snr_, post=False)
                    if q_bas + elbo_bas < q_bas_ + elbo_bas_:
                        _log(self, "Emergency reallocation and removing last group.")
                        reallocate = True
                        for ld in range(self.n_outputs):
                            gpmodels_temp[ld] = gpmodels_temp[ld][:-1]
                            self.wp_sys[ld] = self.wp_sys[ld][:-1]
                        resp_temp, respPair_temp, q, q_lat, snr_aux = self.remove_last_group(resp_temp,
                                                                                             respPair_temp, q,
                                                                                             q_lat, snr_aux)
                        self.gpmodels = gpmodels_temp
                        self.f_ind_old = f_ind_old[reorder]
                        y_trains_w = y_trains_w[:,:,:,reorder]
                        self.y_w = y_trains_w
                        self.y_train = self.select_assigned_warp(y_trains_w, resp_temp)
                        if update_snr:
                            self.snr_norm = self.normalize_snr(snr_aux)
                        return resp_temp, respPair_temp, q, q_lat, snr_aux, y_trains_w, reallocate
                    else:
                        _log(self, "Bad estimation")
                        continue
                q_def__, elbo_def__ = self.compute_q_elbo(resp, respPair, self.weight_mean(q_, snr_),
                                                          self.weight_mean(q_lat_, snr_), self.gpmodels, self.M,
                                                          snr=snr_, post=False)
                i__ = 0
                post_ = True
                while True:
                    resp_temp, respPair_temp, q, q_lat, snr_aux, y_trains_w, gpmodels_temp = self.estimate_q_all(M,
                                                                                                             x_trains=x_trains,
                                                                                                             y_trains=y_trains,
                                                                                                             y_trains_w_=y_trains_w,
                                                                                                             resp=resp_temp,
                                                                                                             respPair=respPair_temp,
                                                                                                             q_=q,
                                                                                                             q_lat_=q_lat,
                                                                                                             snr_=snr_aux,
                                                                                                             startPi=startPi,
                                                                                                             transPi=transPi,
                                                                                                             q_def=q_def__,
                                                                                                             elbo_def=elbo_def__,
                                                                                                             gpmodels=gpmodels_temp,
                                                                                                             reparam=reparam,
                                                                                                             f_ind_old=f_ind_old_temp)
                    q_post, elbo_post = self.compute_q_elbo(resp_temp, respPair_temp, self.weight_mean(q, snr_aux),
                                                            self.weight_mean(q_lat, snr_aux), gpmodels_temp, M,
                                                            snr=snr_aux, post=post_)
                    _log(self, "ELBO_reduction: " + str(((q_post + elbo_post) - (q_bas_ + elbo_bas_)).item()))
                    if (torch.isclose(q_bas_ + elbo_bas_, q_post + elbo_post,
                                      rtol=1e-5) and i__ > 0) or i__ == 10:  # or reparam:
                        break
                    q_bas_ = q_post
                    elbo_bas_ = elbo_post
                    i__ = i__ + 1

                # new_indexes = torch.where(torch.sum(np.abs(resp - resp_temp), dim=1) > 1.0)[0]
                _log(self, "- Trying to divide: " + str(
                    m_chosen) + " with beat " + str(f_ind_new.item()))
                _log(self, ">>> Prev -------")
                q_bas, elbo_bas = self.compute_q_elbo(resp, respPair, self.weight_mean(q_, snr_),
                                                      self.weight_mean(q_lat_, snr_), self.gpmodels, self.M,
                                                      snr=snr_, post=False)
                _log(self, ">>> Post -------")
                q_bas_post, elbo_post = self.compute_q_elbo(resp_temp, respPair_temp, self.weight_mean(q, snr_aux),
                                                            self.weight_mean(q_lat, snr_aux), gpmodels_temp, M,
                                                            snr=snr_aux, post=post_)

                update_snr = True

                if torch.all(torch.sum(resp_temp, dim=0) >= 1.0) and not torch.argmax(
                        torch.sum(resp_temp, dim=0)).item() == resp_temp.shape[1] - 1:
                    if q_bas < q_bas_post:
                        if not q_bas + elbo_bas < q_bas_post + elbo_post:
                            _log(self, "Possibly better q_obs but worse elbo.")
                    if q_bas + elbo_bas < q_bas_post + elbo_post:
                        _log(self, "Chosen to divide: " + str(m_chosen) + " with beat " + str(f_ind_new.item()))
                        self.gpmodels = gpmodels_temp
                        for ld in range(self.n_outputs):
                            self.wp_sys[ld].append(self.create_wp_sys_default())
                        self.x_w = x_w[:,:,:,reorder]
                        self.liks = liks[:,reorder,:]
                        y_trains_w = y_trains_w[:,:,:,reorder]
                        self.y_w = y_trains_w
                        self.y_train = self.select_assigned_warp(y_trains_w, resp_temp)
                        f_ind_old_new = torch.full_like(f_ind_old, -1)
                        used_representers = set()

                        for k in range(M):
                            # Samples assigned to cluster k
                            indexes_k = torch.where(resp_temp[:, k] == 1.0)[0]

                            # Rank candidates by weight, best first
                            order = torch.argsort(self.weight_mean(q_simple_, snr_aux)[indexes_k, k],
                                                  descending=True)
                            sorted_candidates = indexes_k[order]

                            candidate = None
                            for idx in sorted_candidates:
                                idx_val = int(idx.item())
                                if idx_val not in used_representers:
                                    candidate = idx_val
                                    break

                            # Fallback if all candidates were already used
                            if candidate is None:
                                candidate = int(sorted_candidates[0].item())

                            f_ind_old_new[k] = candidate
                            used_representers.add(candidate)
                        self.f_ind_old = torch.clone(f_ind_old_new)
                        if update_snr:
                            self.snr_norm = self.normalize_snr(snr_aux)
                        else:
                            snr_aux = snr_
                        return resp_temp, respPair_temp, q, q_lat, snr_aux, y_trains_w, reallocate
                else:
                    _log(self, "Bad estimation")
    reallocate = True
    return resp, respPair, q_, q_lat_, snr_, y_trains_w_, reallocate


def estimate_q_all(self, M, x_trains, y_trains, y_trains_w_, resp, respPair, q_, q_lat_, snr_, startPi, transPi,
                   q_def, elbo_def, gpmodels=None, reparam=False, post=True, f_ind_old=None):
    """ Internal method to converge the ELBO using the most recent assignation of the examples.
    """
    if gpmodels is None:
        gpmodels = self.gpmodels
    if f_ind_old is None:
        f_ind_old = self.f_ind_old
    q = torch.zeros((len(x_trains), M, self.n_outputs), device=self.device) + torch.min(q_) * 2.0
    q_lat = torch.zeros((len(x_trains), M, self.n_outputs), device=self.device)  # + torch.min(q_lat_) * 2.0
    snr_aux = torch.clone(snr_)

    q_norm, _ = self.LogLik(self.weight_mean(q_, snr_aux))
    alpha, margprob = self.forward(startPi, transPi, q_norm)
    beta = self.backward(transPi, q_norm, margprob)
    logresp, _ = self.LogLik(torch.log(alpha * beta), axis=1)
    logrespPair, _ = self.LogLik(self.coupled_state_coef(alpha, beta, transPi, q_norm, margprob), axis=1)
    resp_temp = self._safe_exp(logresp)
    respPair_temp = self._safe_exp(logrespPair)

    resp_per_group_temp = torch.sum(resp_temp[torch.where(resp_temp == 1.0)[0]], axis=0)
    reorder = torch.argsort(resp_per_group_temp, descending=True)
    resp_temp = torch.clone(resp_temp[:, reorder])

    # after resp_temp = resp_temp[:, reorder] (or clone thereof)
    y_trains_w, x_w, liks = self.warp_batch_by_resp_amtgp_cached(
        x_trains=x_trains,
        y_trains=y_trains,
        resp_temp=resp_temp,  # the argument of estimate_q_all
        f_ind_old=f_ind_old
    )

    q__ = None
    indexes_ = [[] for _ in range(self.n_outputs)]
    gpmodels_temp = [[] for _ in range(self.n_outputs)]
    for ld in range(self.n_outputs):
        if self.verbose:
            _log(self, "\n-----------Lead " + str(ld + 1) + "-----------")
        for m in range(M):
            if self.verbose:
                _log(self, "\n   -----------Model " + str(m + 1) + "-----------")
            indexes_[ld].append(torch.where(resp_temp[:, m] == self.cond_cuda(torch.tensor(1.0)))[0].int())
            if len(gpmodels[ld]) > reorder[m]:
                gp = gpmodels[ld][reorder[m]]
                gp_indexes = torch.tensor(gp.indexes, device=indexes_[ld][m].device).int()
                if not torch.equal(indexes_[ld][m], gp_indexes):
                    gp = self.gpmodel_deepcopy(gpmodels[ld][reorder[m]])
                    if gp.fitted:
                        if reparam:
                            gp.reinit_LDS(save_last=False)
                            gp.reinit_GP(save_last=False)
                        else:
                            gp.reinit_LDS(save_last=True)
                            gp.reinit_GP(save_last=False)
                    else:
                        # No added indexes so createtemp
                        gp = self.create_gp_default(i=reorder[m])
                    q[:, m, ld], q_lat[:, m, ld] = gp.full_pass_weighted(x_trains, y_trains_w[:, :, [ld], reorder[m]],
                                                                         resp_temp[:, m],
                                                                         q=q_[:, reorder[m], ld],
                                                                         q_lat=q_lat_[:, reorder[m], ld],
                                                                         snr=self.snr_norm[:, ld])
                    q[:, m, ld] = q[:, m, ld] + liks[:, reorder[m], ld]
                    snr_aux[:, m, ld] = self.compute_snr(y_trains_w[:, :, ld, reorder[m]], gp)
                else:
                    gp = gpmodels[ld][reorder[m]]
                    q[:, m, ld] = q_[:, reorder[m], ld]
                    q_lat[:, m, ld] = q_lat_[:, reorder[m], ld]
                    snr_aux[:, m, ld] = torch.clone(snr_[:, m, ld])
            else:
                # New GP to add
                gp = self.create_gp_default(i=reorder[m])
                if len(indexes_[ld][m]) > 0.0:
                    q[:, m, ld], q_lat[:, m, ld] = gp.full_pass_weighted(x_trains, y_trains_w[:, :, [ld], reorder[m]],
                                                                         resp_temp[:, m],
                                                                         q=q_[:, reorder[m], ld],
                                                                         q_lat=q_lat[:, reorder[m], ld],
                                                                         snr=self.snr_norm[:, ld])
                    q[:, m, ld] = q[:, m, ld] + liks[:, reorder[m], ld]
                    snr_aux[:, m, ld] = self.compute_snr(y_trains_w[:, :, ld, reorder[m]], gp)
                else:
                    q[:, m, ld] = q_[:, m, ld]
                    q_lat[:, m, ld] = q_lat_[:, m, ld]
                    snr_aux[:, m, ld] = torch.zeros(snr_.shape[0])
            gpmodels_temp[ld].append(gp)

    _log(self, ">>> Q_all_loop -------")
    q_bas, elbo_bas = self.compute_q_elbo(resp, respPair, self.weight_mean(q_, snr_),
                                          self.weight_mean(q_lat_, snr_), gpmodels, self.M, snr=snr_, post=post)
    q_bas_post, elbo_post = self.compute_q_elbo(resp_temp, respPair_temp, self.weight_mean(q, snr_aux),
                                                self.weight_mean(q_lat, snr_aux), gpmodels_temp, M, snr=snr_aux,
                                                post=post)
    update_snr = True
    if torch.any(torch.sum(resp_temp, dim=1) > 2.0):
        _log(self, "STOP")
    if torch.all(torch.sum(resp_temp, dim=0) >= 1.0):
        if q_bas + elbo_bas < q_bas_post + elbo_post:
            # self.gpmodels = gpmodels_temp
            self.x_w = x_w
            self.liks = liks
            y_trains_w = y_trains_w[:,:,:,reorder]
            if reorder.shape[0] == self.f_ind_old.shape[0]:
                self.f_ind_old = self.f_ind_old[reorder]
            if update_snr:
                self.snr_norm = self.normalize_snr(snr_aux)
            else:
                snr_aux = snr_
            return resp_temp, respPair_temp, q, q_lat, snr_aux, y_trains_w, gpmodels_temp
        else:
            return resp, respPair, q_, q_lat_, snr_, y_trains_w_, gpmodels
    else:
        _log(self, f">>> Possible emergency reallocation. Prev ----:\n Q_em: {q_def}, Elbo: {elbo_def}")
        if (q_def + elbo_def < q_bas_post + elbo_post) and (q_bas + elbo_bas < q_bas_post + elbo_post):
            _log(self, "Emergency reallocation and removing last group.")
            reallocate = True
            for ld in range(self.n_outputs):
                gpmodels_temp[ld] = gpmodels_temp[ld][:-1]
            self.gpmodels = gpmodels_temp
            if update_snr:
                self.snr_norm = self.normalize_snr(snr_aux)
            resp_temp, respPair_temp, q, q_lat, snr_aux = self.remove_last_group(resp_temp,
                                                                                 respPair_temp, q,
                                                                                 q_lat, snr_aux)
            resp_per_group_temp = torch.sum(resp_temp[torch.where(resp_temp == 1.0)[0]], axis=0)
            reorder = torch.argsort(resp_per_group_temp, descending=True)
            self.f_ind_old = self.f_ind_old[reorder]
            return resp_temp, respPair_temp, q, q_lat, snr_aux, y_trains_w, gpmodels_temp
        else:
            _log(self, "Bad estimation")
            return resp, respPair, q_, q_lat_, snr_, y_trains_w, gpmodels


def classify_batch(self, x_trains, y_trains, learning=False, it_limit=None, warp=False):
    warp = _alignment.configured(self).enabled(warp)
    if not learning:
        x_trains = self.cond_cuda(self.cond_to_torch(x_trains))
        y_trains = self.cond_cuda(self.cond_to_torch(y_trains))
        q = torch.zeros(y_trains.shape[0], self.M, self.n_outputs)
        snr = torch.zeros(y_trains.shape[0], self.M, self.n_outputs)
        for ld in range(self.n_outputs):
            for m in range(self.M):
                gp = self.gpmodels[ld][m]
                for i in range(y_trains.shape[0]):
                    q[i, m, ld] = gp.log_sq_error(x_trains[i], y_trains[i, :, [ld]], i=-1)
                snr[:, m, ld] = self.compute_snr(y_trains[:, :, ld], gp)
        transTheta = self.transTheta
        startTheta = self.startTheta
        transPi = self.hdp.classification_log_transitions(self.M, transTheta)
        # Calculate LOG of start state prob vector
        startPi = self.hdp.classification_log_start(self.M, startTheta)

        q_norm, _ = self.LogLik(self.weight_mean(q, snr))
        alpha, margprob = self.forward(startPi, transPi, q_norm)
        beta = self.backward(transPi, q_norm, margprob)
        resplog_temp, _ = self.LogLik(torch.log(alpha * beta), axis=1)
        respPairlog_temp, _ = self.LogLik(self.coupled_state_coef(alpha, beta, transPi, q_norm, margprob), axis=1)
        resp = self._safe_exp(resplog_temp)
        return torch.where(resp == 1.0)[1]
    if learning:
        x_trains = self.cond_cuda(self.cond_to_torch(x_trains))
        y_trains = self.cond_cuda(self.cond_to_torch(y_trains))
        q_new = torch.zeros(y_trains.shape[0], self.M, self.n_outputs)
        snr_new = torch.zeros(y_trains.shape[0], self.M, self.n_outputs)
        for ld in range(self.n_outputs):
            for m in range(self.M):
                gp = self.gpmodels[ld][m]
                for i in range(y_trains.shape[0]):
                    q_new[i, m, ld] = gp.log_sq_error(x_trains[i], y_trains[i, :, [ld]], i=-1)
                snr_new[:, m, ld] = self.compute_snr(y_trains[:, :, ld], gp)
        transTheta = self.transTheta
        startTheta = self.startTheta
        transPi = self.hdp.batch_log_transitions(self.M, transTheta)
        # Calculate LOG of start state prob vector
        startPi = self.hdp.batch_log_start(self.M, startTheta)

        q_norm, _ = self.LogLik(self.weight_mean(q_new, snr_new))
        alpha, margprob = self.forward(startPi, transPi, q_norm)
        beta = self.backward(transPi, q_norm, margprob)
        resplog_temp, _ = self.LogLik(torch.log(alpha * beta), axis=1)
        respPairlog_temp, _ = self.LogLik(self.coupled_state_coef(alpha, beta, transPi, q_norm, margprob), axis=1)
        resp_new = self._safe_exp(resplog_temp)
        respPair_new = self._safe_exp(respPairlog_temp)
        x_trains = torch.concatenate([self.x_train, x_trains])
        y_trains = torch.concatenate([self.y_train, y_trains])
        y_trains_w = y_trains
        self.T = y_trains.shape[0]
        resp, respPair = torch.concatenate([self.resp_last, resp_new]), torch.concatenate(
            [self.respPair_last, respPair_new])
        self.snr_norm = torch.concatenate([self.snr_norm, self.normalize_snr(snr_new)])
        resp_per_group_temp = torch.sum(resp, dim=0)
        reorder = torch.argsort(resp_per_group_temp, descending=True)
        resp = resp[:, reorder]

        q = torch.zeros(y_trains.shape[0], self.M, self.n_outputs)
        q_lat = torch.zeros(y_trains.shape[0], self.M, self.n_outputs)
        snr = torch.zeros(y_trains.shape[0], self.M, self.n_outputs)

        gpmodels_temp = [[] for _ in range(self.n_outputs)]
        for ld in range(self.n_outputs):
            if self.verbose:
                _log(self, "\n-----------Lead " + str(ld + 1) + "-----------")
            for m in range(self.M):
                if self.verbose:
                    _log(self, "\n   -----------Model " + str(m + 1) + "-----------")
                gp = self.gpmodel_deepcopy(self.gpmodels[ld][reorder[m]])
                if gp.fitted:
                    gp.reinit_LDS(save_last=False)
                    gp.reinit_GP(save_last=False)
                q[:, m, ld], q_lat[:, m, ld] = gp.full_pass_weighted(x_trains, y_trains[:, :, [ld]],
                                                                     resp[:, m])
                snr[:, m, ld] = self.compute_snr(y_trains[:, :, ld], gp)
                gpmodels_temp[ld].append(gp)
        self.gpmodels = gpmodels_temp
        q_norm, _ = self.LogLik(self.weight_mean(q, snr))
        alpha, margprob = self.forward(startPi, transPi, q_norm)
        beta = self.backward(transPi, q_norm, margprob)
        resplog_temp, _ = self.LogLik(torch.log(alpha * beta), axis=1)
        respPairlog_temp, _ = self.LogLik(self.coupled_state_coef(alpha, beta, transPi, q_norm, margprob), axis=1)
        resp = self._safe_exp(resplog_temp)
        respPair = self._safe_exp(respPairlog_temp)

        startStateCount = self.cond_cuda(resp[0])
        transStateCount = self.cond_cuda(torch.sum(respPair, axis=0))
        reallocate = False
        elbo = self.elbo_last
        iteration = 0
        while True:
            resp, respPair, q, q_lat, snr, end = self.refill(resp, respPair, startStateCount, transStateCount, q,
                                                             q_lat, snr)
            M = self.M
            if end:
                break
            resp, respPair, q, q_lat, snr, y_trains_w, reallocate = self.variational_local_terms_batch(M, x_trains,
                                                                                                       y_trains,
                                                                                                       y_trains_w,
                                                                                                       self.transTheta,
                                                                                                       self.startTheta,
                                                                                                       resp,
                                                                                                       respPair, q,
                                                                                                       q_lat, snr,
                                                                                                       reallocate)
            if resp.shape[1] > M:
                self.M = M + 1
                M = self.M
            resp__ = torch.clone(resp)
            respPair__ = torch.clone(respPair)
            # Update HDP variational params.
            if self.hmm_switch:
                startStateCount = self.cond_cuda(resp__[0])
                transStateCount = self.cond_cuda(torch.sum(respPair__, axis=0))
            else:
                transStateCount = self.cond_cuda(
                    torch.ones((M + 1, M + 1)))  # Cause we are in independent observations
                startStateCount = self.cond_cuda(torch.ones(M + 1))

            self.reinit_global_params(M, transStateCount, startStateCount)
            nIters = 2
            for giter in range(nIters):
                self.transTheta, self.startTheta = self._calcThetaFull(self.cond_cuda(transStateCount),
                                                                       self.cond_cuda(startStateCount), M + 1)
                self.rho, self.omega = self.find_optimum_rhoOmega()

            # Update transition matrix
            transPi = self.hdp.stored_log_transitions(M, self.transTheta)
            self.trans_A = transPi
            # Compute ELBO and start steps of EM
            if self.T > 1:
                elbo_ = self.calcELBO_NonlinearTerms(resp=self.cond_to_numpy(self.cond_cpu(resp)),
                                                     respPair=self.cond_to_numpy(self.cond_cpu(respPair)))
                _log(self, '\n-------End Lower Bound Iteration ' + str(iteration) + '-------')
                q_obs, elbo_lin = self.compute_q_elbo(resp, respPair, self.weight_mean(q), self.weight_mean(q_lat),
                                                      self.gpmodels, self.M, snr='saved', post=False)
                elbo_ = elbo_ + elbo_lin + q_obs
                _log(self, "ELBO + Nonlinear: " + str(elbo_))
                iteration = iteration + 1
                _log(self, '\n-------Start lower Bound Iteration ' + str(iteration) + '-------')
                if it_limit is not None and iteration >= it_limit:
                    self.train_elbo.append(elbo_)
                    self.resp_assigned.append(torch.where(resp == 1.0)[1])
                    break
                resp_group = torch.sum(resp, dim=0)
                self.train_elbo.append(elbo_)
                self.resp_assigned.append(torch.where(resp == 1.0)[1])
                self.q_last, self.q_lat_last, self.snr_last = q, q_lat, snr
                self.startStateCount_last, self.transStateCount_last = startStateCount, transStateCount
                self.resp_last, self.respPair_last = resp, respPair
                if (torch.where(resp_group == 0.0)[0].shape[0] > 1.0 or
                        (len(self.resp_assigned) > 1 and (
                                self.resp_assigned[-2].shape[0] == self.resp_assigned[-1].shape[0])
                         and torch.all(self.resp_assigned[-2] == self.resp_assigned[-1]))):
                    if warp_computed or not warp:
                        if not warp:
                            self.y_train = y_trains
                        break
                    else:
                        elbo = elbo_
                        q, q_lat, warp_computed = self.compute_warp_actual_state(x_trains, y_trains, q=q,
                                                                                 q_lat=q_lat)
                else:
                    elbo = elbo_
                    self.y_train = y_trains
            else:
                break


def fit_labels(self, x_trains, y_trains, labels, M, warp=False):
    warp = _alignment.configured(self).enabled(warp)
    assert y_trains.shape[2] == self.n_outputs
    y_trains = self.cond_cuda(self.cond_to_torch(y_trains))
    x_trains = self.cond_cuda(self.cond_to_torch(x_trains))
    if M != self.M:
        gp = self.gpmodels[0][0]
        self.gpmodels = [[] for _ in range(self.n_outputs)]
        for ld in range(self.n_outputs):
            for m in range(M):
                self.gpmodels[ld].append(self.gpmodel_deepcopy(gp))
                self.wp_sys[ld].append(self.create_wp_sys_default())
    self.M = M
    self.T = y_trains.shape[0]
    self.y_train = y_trains
    self.x_train = x_trains
    self.y = y_trains
    self.model_type = [self.model_type[0]] * M
    resp = torch.zeros(y_trains.shape[0], M)
    resp[torch.arange(y_trains.shape[0]), labels] = 1.0
    respPair = self.cond_cuda(torch.zeros((y_trains.shape[0], M, M)))
    respPair[np.arange(y_trains.shape[0] - 1), labels[:-1], labels[1:]] = 1.0
    q = torch.zeros(y_trains.shape[0], M, self.n_outputs)
    q_lat = torch.zeros(y_trains.shape[0], M, self.n_outputs)
    snr = torch.zeros(y_trains.shape[0], M, self.n_outputs)
    for ld in range(self.n_outputs):
        for m in range(M):
            gp = self.gpmodel_deepcopy(self.gpmodels[0][0])
            if gp.fitted:
                gp.reinit_LDS(save_last=False)
                gp.reinit_GP(save_last=False)
            q[:, m, ld], q_lat[:, m, ld] = gp.full_pass_weighted(x_trains, y_trains[:, :, [ld]],
                                                                 resp[:, m])
            snr[:, m, ld] = self.compute_snr(y_trains[:, :, ld], gp)
            self.gpmodels[ld][m] = gp
    if warp:
        q, q_lat, warp_computed, y_trains = self.compute_warp_actual_state(x_trains, y_trains, q=q,
                                                                           q_lat=q_lat)
        for ld in range(self.n_outputs):
            for m in range(M):
                gp = self.gpmodel_deepcopy(self.gpmodels[0][0])
                if gp.fitted:
                    gp.reinit_LDS(save_last=False)
                    gp.reinit_GP(save_last=False)
                q[:, m, ld], q_lat[:, m, ld] = gp.full_pass_weighted(x_trains, y_trains[:, :, [ld]],
                                                                     resp[:, m])
                snr[:, m, ld] = self.compute_snr(y_trains[:, :, ld], gp)
                self.gpmodels[ld][m] = gp
    self.q.append(q)
    resp__ = torch.clone(resp)
    respPair__ = torch.clone(respPair)
    # Update HDP variational params.
    startStateCount = self.cond_cuda(resp__[0])
    transStateCount = self.cond_cuda(torch.sum(respPair__, axis=0))
    resp, respPair, q, q_lat, snr, end = self.refill(resp, respPair, startStateCount, transStateCount, q, q_lat,
                                                     snr)

    self.reinit_global_params(M, transStateCount, startStateCount)
    nIters = 2
    for giter in range(nIters):
        self.transTheta, self.startTheta = self._calcThetaFull(self.cond_cuda(transStateCount),
                                                               self.cond_cuda(startStateCount), M + 1)
        self.rho, self.omega = self.find_optimum_rhoOmega()
    transTheta = self.transTheta
    startTheta = self.startTheta
    transPi = self.hdp.batch_log_transitions(M, transTheta)
    self.trans_A = transPi
    self.resp_assigned.append(torch.where(resp == 1.0)[1])
    self.q_last, self.q_lat_last, self.snr_last = q, q_lat, snr
    self.startStateCount_last, self.transStateCount_last = startStateCount, transStateCount
    self.resp_last, self.respPair_last = resp, respPair
    self.snr_norm = self.normalize_snr(snr)
    self.f_ind_old = torch.zeros(M, device=self.device).long()
    for m in range(self.M):
        ind = self.gpmodels[0][m].indexes
        self.f_ind_old[m] = ind[torch.argmax(self.weight_mean(q, snr)[ind, m]).long()]
    elbo_ = self.calcELBO_NonlinearTerms(resp=self.cond_to_numpy(self.cond_cpu(resp)),
                                         respPair=self.cond_to_numpy(self.cond_cpu(respPair)))
    q_obs, elbo_lin = self.compute_q_elbo(resp, respPair, self.weight_mean(q), self.weight_mean(q_lat),
                                          self.gpmodels, self.M, snr='saved', post=False)
    elbo_ = elbo_ + elbo_lin + q_obs
    _log(self, '\n-------ELBO:' + str(elbo_) + '-------')
    self.elbo_last = elbo_

