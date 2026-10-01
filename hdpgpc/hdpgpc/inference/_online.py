"""Native standard and fast online schedules.
The explicitly supplied mutable state owns histories. Keep the separate update
orders, proposal loops, and acceptance checks unchanged."""

import numpy as np
import torch
from ..alignment import _components as _alignment


def _report_assignment(self, sample_index, model, birth):
    if not self.verbose:
        return
    if birth:
        print(f"Sample {sample_index + 1}: created cluster {model + 1}.", flush=True)
    print(f"Sample {sample_index + 1}: assigned to cluster {model + 1}.", flush=True)


def update(self, x_train, y, with_warp=True, force_model=None, minibatch=0, classify=False):
    """
    Include the sample received and restimate all the parameters of the
    full model, including the state-space and the GPS.
    Parameters
    ----------
    y : array sample of shape (s_samples)

    Returns
    -------
    self : returns an instance of self.
    """

    # Adding one sample
    one_sample = True
    t = self.T
    if not classify:
        self.T = self.T + 1
        self.snr_norm = torch.ones(self.T, self.n_outputs)
    M = self.M
    y = self.cond_cuda(self.cond_to_torch(y))
    x_train = self.cond_cuda(self.cond_to_torch(x_train))
    if minibatch == 0 and self.batch is not None:
        minibatch = self.batch
    if minibatch >= t:
        minibatch = 0
    # Compute warp once
    y_mod = []
    liks = np.array([0.0] * (M + 1))
    for m in range(M):
        y_ = self.y.copy()[-1 * minibatch:]
        y_mod.append(y_)
    if not classify:
        self.y.append(y)
        self.x_train.append(x_train)
    y_mod, liks = _alignment.configured(self).prepare_online(
        self, x_train, y, t, M, y_mod, liks, with_warp, force_model, fast=False)
    elbo = 0.0
    q_all = 0.0
    q_aux = torch.zeros(self.T, self.M + 1, self.n_outputs) - np.inf
    q_lat = torch.zeros(self.T, self.M + 1, self.n_outputs)
    if t > 0:
        if not classify:
            q_aux[:-1, :self.q[-1].shape[1], :] = torch.clone(self.q[-1])
        else:
            q_aux = torch.clone(self.q[-1])
    for ld in range(self.n_outputs):
        for m, gp in enumerate(self.gpmodels[ld]):
            q_lat[:, m, ld] = gp.compute_q_lat_all(torch.from_numpy(np.array(self.x_train)), h_ini=1.0)
            q_aux[-1, m, ld] = gp.log_sq_error(torch.from_numpy(np.array(self.x_train[-1])), y_mod[m][-1], i=-1) + liks[m]
            #q_aux[[-1], m, ld] = self.estimate_new(t, gp, self.x_train[-1], y_mod[m][-1], h=1.0)
    if t > 0:
        resp, resp_log, respPair, respPair_log = self.variational_local_terms(q_aux, self.transTheta, self.startTheta)
        q_all, elbo = self.compute_q_elbo(resp[:-1,:-1], respPair[:-1,:-1,:-1], self.weight_mean(q_aux)[:-1,:-1],
                                                          self.weight_mean(q_lat)[:-1,:-1],
                                                          self.gpmodels, self.M, snr='saved', post=False,
                                                          one_sample=True, verb=self.verbose)
    if t > 0:
        resp, resp_log, respPair, respPair_log = self.variational_local_terms(q_aux, self.transTheta,
                                                                              self.startTheta)
        q_all, elbo = self.compute_q_elbo(resp[:-1, :-1], respPair[:-1, :-1, :-1],
                                          self.weight_mean(q_aux)[:-1, :-1],
                                          self.weight_mean(q_lat)[:-1, :-1],
                                          self.gpmodels, self.M, snr='saved', post=False,
                                          one_sample=True, verb=self.verbose)
    if t > 0 and not classify:
        # Define order
        q_ord = torch.argsort(self.weight_mean(q_aux)[-1, :-1], descending=True)
        m = q_ord[-1].item()
        # m = q_ord[0].item()
        q_prev = torch.clone(q_aux)
        q_lat_prev = torch.clone(q_lat)
        # Compute first birth cost
        for ld in range(self.n_outputs):
            prov_gp = self.gpmodel_deepcopy(self.gpmodels[ld][m])
            prov_gp.reinit_GP(save_last=False)
            prov_gp.reinit_LDS(save_last=False)
            q_prev[[-1], -1, ld] = self.estimate_new(t, prov_gp, self.x_train[-1], y_mod[-1][-1], h=1.0) + liks[-1]
            prov_gp.include_weighted_sample(t, self.x_train[-1], self.x_train[-1], y_mod[-1][-1], 1.0)
            self.gpmodels[ld].append(prov_gp)
            # self.M = M + 1
            q_lat_prev[:, -1, ld] = prov_gp.compute_q_lat_all(torch.from_numpy(np.array(self.x_train)), h_ini=1.0)
        resp_prev, resp_prev_log, respPair_prev, respPair_prev_log = self.variational_local_terms(q_prev,
                                                                                                  self.transTheta,
                                                                                                  self.startTheta,
                                                                                                  liks)
        q_prev_post, elbo_prev_post = self.compute_q_elbo(resp_prev, respPair_prev, self.weight_mean(q_prev),
                                                          self.weight_mean(q_lat_prev),
                                                          self.gpmodels, self.M, snr='saved', one_sample=True,
                                                          post=True, verb=self.verbose)
        elbo_prev_post = elbo_prev_post - elbo  # / np.log(self.T + 1)
        q_prev_post = q_prev_post - q_all
        for ld in range(self.n_outputs):
            self.gpmodels[ld].pop()
            self.M = M
        if torch.argmax(q_prev[-1]) == self.M:
            q_post = torch.clone(q_aux)
            q_lat_post = torch.clone(q_lat)
            for m in q_ord:
                saved_gps = [self.gpmodels[ld][m] for ld in range(self.n_outputs)]
                for ld in range(self.n_outputs):
                    post_gp = self.gpmodel_deepcopy(self.gpmodels[ld][m])
                    q_post[[-1], m, ld] = self.estimate_new(t, post_gp, self.x_train[-1], y_mod[m][-1], h=1.0) + liks[m]
                    post_gp.include_weighted_sample(t, self.x_train[-1], self.x_train[-1], y_mod[m][-1], 1.0)
                    self.gpmodels[ld][m] = post_gp
                    post_gp.backwards_pair(1.0)
                    post_gp.bayesian_new_params(1.0)
                    q_lat_post[:, m, ld] = post_gp.compute_q_lat_all(torch.from_numpy(np.array(self.x_train)),
                                                                     h_ini=1.0)
                    # q_post[[-1], m, ld] = self.estimate_new(t, post_gp, self.x_train[-1], y_mod[m][-1], h=1.0)
                resp_post, resp_post_log, respPair_post, respPair_post_log = self.variational_local_terms(q_post,
                                                                                                          self.transTheta,
                                                                                                          self.startTheta,
                                                                                                          liks)
                q_bas_post, elbo_bas_post = self.compute_q_elbo(resp_post[:, :-1], respPair_post[:, :-1, :-1],
                                                                self.weight_mean(q_post)[:, :-1],
                                                                self.weight_mean(q_lat_post)[:, :-1],
                                                                self.gpmodels, self.M, snr='saved', post=False,
                                                                one_sample=True, verb=self.verbose)
                elbo_bas_post = elbo_bas_post - elbo  # / np.log(self.T + 1)
                q_bas_post = q_bas_post - q_all

                if q_bas_post + elbo_bas_post > q_prev_post + elbo_prev_post:
                    resp, resplog, respPair, respPairlog = self.variational_local_terms(q_post, self.transTheta,
                                                                                        self.startTheta, liks)
                    q_chos = q_post
                    q_lat_chos = q_lat_post
                    for ld in range(self.n_outputs):
                        self.gpmodels[ld][m] = saved_gps[ld]
                    break
                else:
                    for ld in range(self.n_outputs):
                        self.gpmodels[ld][m] = saved_gps[ld]
                    q_chos = q_prev
                    q_lat_chos = q_lat_prev
                    resp, resplog, respPair, respPairlog = resp_prev, resp_prev_log, respPair_prev, respPair_prev_log
        else:
            q_chos = q_aux
            q_lat_chos = q_lat
            resp, resplog, respPair, respPairlog = self.variational_local_terms(q_chos, self.transTheta,
                                                                                self.startTheta, liks)
    else:
        q_chos = q_aux
        q_lat_chos = q_lat
        resp, resplog, respPair, respPairlog = self.variational_local_terms(q_aux, self.transTheta, self.startTheta,
                                                                            liks)

    if len(resp.shape) == 1:
        resp_mod = self.cond_to_numpy(self.cond_cpu(resp))
        resp_modlog = self.cond_to_numpy(self.cond_cpu(resplog))
    else:
        resp_mod = self.cond_to_numpy(self.cond_cpu(resp[-1]))
        resp_modlog = self.cond_to_numpy(self.cond_cpu(resplog[-1]))

    if classify:
        return q_chos[:-1], resp_mod[:-1], liks[:-1]

    # Normalize when responsability is equally shared
    if sum(np.isclose(resp_mod, max(resp_mod), rtol=1e-2)) > 1:
        h_argmax = np.nanargmax(resp_mod)
        resp_mod[:] = 0.0
        resp_mod[h_argmax] = 1.0

    model = np.argmax(resp_mod)
    if self.max_models is not None and model >= self.max_models:
        force_model = np.argmax(resp_modlog[:-1])
        model = force_model

    # When converged apply all transformations to real models.
    if not force_model is None:
        resp_mod[:] = 0.0
        resp_mod[force_model] = 1.0
        model = np.argmax(resp_mod)
        # resp = self.normalize_log(resp_mod)

    # Birth of new model
    birth = False
    if model == self.M:
        birth = True
    if birth:
        self.M = self.M + 1
        M = self.M
        y_mod.append(self.y.copy())
        for ld in range(self.n_outputs):
            self.gpmodels[ld].append(self.create_gp_default())
            self.wp_sys[ld].append(self.create_wp_sys_default())
        self.x_basis.append(self.x_basis_ini)
        resp, respPair, q_chos, q_lat_chos, reorder = self.reorder(resp, respPair, q_chos, q_lat_chos)
        startStateCount = resp[0]
        transStateCount = torch.sum(respPair, axis=0)
        if M > 2:
            self.reinit_global_params(M - 1, transStateCount, startStateCount)
        if M >= 2:
            nIters = 4
            for giter in range(nIters):
                self.transTheta, self.startTheta = self._calcThetaFull(self.cond_cuda(transStateCount),
                                                                       self.cond_cuda(startStateCount), M)
                self.rho, self.omega = self.find_optimum_rhoOmega()

        # Update transition matrix
        transPi = self.hdp.stored_log_transitions(M, self.transTheta)
        self.trans_A = transPi
    else:
        if force_model is None:
            resp, respPair, q_chos, q_lat_chos, reorder = self.reorder(resp, respPair, q_chos, q_lat_chos)
        startStateCount = resp[0, :M]
        transStateCount = torch.sum(respPair[:, :M, :M], axis=0)
        if M > 2:
            self.reinit_global_params(M - 1, transStateCount, startStateCount)
        if M >= 2:
            nIters = 4
            for giter in range(nIters):
                self.transTheta, self.startTheta = self._calcThetaFull(self.cond_cuda(transStateCount),
                                                                       self.cond_cuda(startStateCount), M)
                self.rho, self.omega = self.find_optimum_rhoOmega()

        # Update transition matrix
        transPi = self.hdp.stored_log_transitions(M, self.transTheta)
        self.trans_A = transPi

    if len(resp.shape) == 1:
        resp_mod = self.cond_to_numpy(self.cond_cpu(resp))
        resp_modlog = self.cond_to_numpy(self.cond_cpu(resplog))
    else:
        resp_mod = self.cond_to_numpy(self.cond_cpu(resp[-1]))
        resp_modlog = self.cond_to_numpy(self.cond_cpu(resplog[-1]))

    # Normalize when responsability is equally shared
    if sum(np.isclose(resp_mod, max(resp_mod), rtol=1e-2)) > 1:
        h_argmax = np.nanargmax(resp_mod)
        resp_mod[:] = 0.0
        resp_mod[h_argmax] = 1.0

    model = np.argmax(resp_mod)
    if self.max_models is not None and model >= self.max_models:
        force_model = np.argmax(resp_modlog[:-1])
        model = force_model

    # When converged apply all transformations to real models.
    if not force_model is None:
        resp_mod[:] = 0.0
        resp_mod[force_model] = 1.0
        model = force_model
        q_chos[-1, model] = torch.max(q_chos[-1])
        q_lat_chos[-1, model] = torch.max(q_lat_chos[-1])
        resp[-1, :] = 0.0
        resp[-1, model] = 1.0
        respPair[-1, model, :] = 0.0
        respPair[-1, :, model] = 0.0
        respPair[-1, model, model] = 1.0

    # Assign and update GP model
    self.actual_state = model
    if minibatch == 0:
        minibatch = None
    for ld in range(self.n_outputs):
        for m in range(M):
            new_x_basis = self.gpmodels[ld][m].include_weighted_sample(t, self.x_train[-1], self.x_train[-1], y_mod[reorder[m]][-1], resp_mod[m])
            self.x_basis[m] = self.cond_cuda(self.cond_to_torch(new_x_basis.detach()))
            if resp_mod[m] > 0.9:
                self.y_train = torch.concatenate([self.y_train, torch.atleast_2d(y_mod[reorder[m]][-1]).T[:, :, np.newaxis]])
            if self.bayesian_params:
                self.gpmodels[ld][m].bayesian_new_params(resp_mod[m], model_type=self.model_type[m])
            else:
                self.gpmodels[ld][m].new_params_weighted(resp_mod[m], batch=minibatch, min_samples=self.min_samples,
                                                         max_samples=self.max_samples,
                                                         div_samples=self.div_samples, verbose=False,
                                                         model_type=self.model_type[m], check_var=self.check_var)
    self.compute_q_elbo(resp[:, :M], respPair[:, :M, :M], self.weight_mean(q_chos)[:, :M],
                        self.weight_mean(q_lat_chos)[:, :M],
                        self.gpmodels, self.M, snr='saved', post=False, one_sample=True)
    # q_ = torch.zeros((self.T, self.M, self.n_outputs))
    # for ld in range(self.n_outputs):
    #     q_[:,:,ld] = self.compute_q(y=y_mod, gpmodels=self.gpmodels[ld], ld=ld)
    self.resp_assigned.append(torch.argmax(resp, axis=1))
    self.q.append(q_chos)
    for ld in range(self.n_outputs):
        _alignment.configured(self).update_selected(self, ld, model, x_train, with_warp)
    _report_assignment(self, t, model, birth)


def _init_resp_cache_cold_start(self, init_state: int = 0):
    T = self.T
    M = self.M
    device = self.device

    resp = self.cond_cuda(torch.zeros((T, M), device=device, dtype=torch.float64))
    resp[0, init_state] = 1.0

    respPair = self.cond_cuda(torch.zeros((T, M, M), device=device, dtype=torch.float64))
    respPair[0, init_state, init_state] = 1.0  # safe default for T=1

    self.resp_last = resp
    self.respPair_last = respPair


def _ensure_fast_cache_online(self):
    """
    Ensure q_last, q_lat_last, resp_last, respPair_last exist for the ONLINE scheme.
    - q_last: last stored q (T,M,n_outputs)
    - q_lat_last: cached latent error (T,M,n_outputs). If missing, computed ONCE.
    - resp_last/respPair_last: cached hard responsibilities (T,M) and pairs (T,M,M).
    """
    device = self.device

    # q_last
    if not hasattr(self, "q_last") or self.q_last is None:
        if len(self.q) > 0:
            q0 = self.q[-1]
            self.q_last = q0[:, :self.M, :].detach()
        else:
            self.q_last = (torch.zeros((self.T, self.M, self.n_outputs), device=device) - np.inf)

    # resp_last / respPair_last (rebuild from stored hard labels if needed)
    if not hasattr(self, "resp_last") or self.resp_last is None:
        if len(self.resp_assigned) > 0:
            labels = self.resp_assigned[-1].detach().to(device)
            T = labels.shape[0]
            resp = torch.zeros((T, self.M), device=device, dtype=torch.float64)
            resp[torch.arange(T, device=device), labels] = 1.0
            respPair = torch.zeros((T, self.M, self.M), device=device, dtype=torch.float64)
            if T > 1:
                respPair[torch.arange(1, T, device=device), labels[:-1], labels[1:]] = 1.0
            self.resp_last = resp
            self.respPair_last = respPair
        else:
            self.resp_last = torch.zeros((self.T, self.M), device=device, dtype=torch.float64)
            self.respPair_last = torch.zeros((self.T, self.M, self.M), device=device, dtype=torch.float64)
            if self.T > 0:
                self.resp_last[0, 0] = 1.0
                self.respPair_last[0, 0, 0] = 1.0

    # q_lat_last (if missing: one-time fill; afterwards we only patch tail indices)
    if not hasattr(self, "q_lat_last") or self.q_lat_last is None:
        x_tr = self.cond_cuda(self.cond_to_torch(torch.from_numpy(np.array(self.x_train))))
        q_lat = torch.zeros((self.T, self.M, self.n_outputs), device=device, dtype=torch.float64)
        for ld in range(self.n_outputs):
            for m, gp in enumerate(self.gpmodels[ld]):
                q_lat[:, m, ld] = gp.compute_q_lat_all(x_tr, h_ini=1.0)
        self.q_lat_last = q_lat


def _update_q_lat_tail(self, gp, q_lat_col, update_idxs, h_ini=1.0):
    """
    Patch q_lat ONLY at a few time indices (tail), using gp.log_lat_error.
    This is the fast replacement for gp.compute_q_lat_all over the whole dataset.
    """
    for t_idx in update_idxs:
        if t_idx in gp.indexes:
            j = gp.indexes.index(t_idx)
            try:
                q_lat_col[t_idx] = gp.log_lat_error(j, h_ini)
            except Exception:
                pass
    return q_lat_col


def _append_hard_step(self, resp_prev, respPair_prev, new_state, K):
    """
    Build (T+1,K) resp and (T+1,K,K) respPair by reusing previous tensors and
    appending the last hard state + last transition.
    """
    T_prev = resp_prev.shape[0]
    device = resp_prev.device
    dtype = resp_prev.dtype

    resp = torch.zeros((T_prev + 1, K), device=device, dtype=dtype)
    resp[:T_prev, :resp_prev.shape[1]] = resp_prev
    resp[T_prev, new_state] = 1.0

    respPair = torch.zeros((T_prev + 1, K, K), device=device, dtype=dtype)
    if respPair_prev is not None and respPair_prev.numel() > 0:
        respPair[:T_prev, :respPair_prev.shape[1], :respPair_prev.shape[2]] = respPair_prev

    if T_prev == 0:
        respPair[T_prev, new_state, new_state] = 1.0
    else:
        prev_state = int(torch.argmax(resp_prev[-1]).item())
        respPair[T_prev, prev_state, new_state] = 1.0

    return resp, respPair


def update_fast(self, x_train, y, with_warp=True, force_model=None, minibatch=0, classify=False):
    """
    Fast online include that STILL compares:
      - birth new cluster  vs  absorb into existing clusters
    using estimate_new + ELBO (like update), but without recomputing
    q / q_lat / forward-backward over all past samples.

    Key approximation (as you requested):
      - past resp/respPair are reused; only last step is appended (hard).
      - q_lat is patched only at tail indices (t and maybe t-1).
    """

    one_sample = True
    t = self.T
    if not classify:
        self.T = self.T + 1
        self.snr_norm = torch.ones(self.T, self.n_outputs, device=self.device, dtype=torch.float64)

    M = self.M
    y = self.cond_cuda(self.cond_to_torch(y))
    x_train = self.cond_cuda(self.cond_to_torch(x_train))

    if minibatch == 0 and self.batch is not None:
        minibatch = self.batch
    if minibatch >= t:
        minibatch = 0

    # Caches from previous iteration
    if t > 0:
        self._ensure_fast_cache_online()
    else:
        self._init_resp_cache_cold_start()

    # --- Warp (same as update) ---
    y_mod = []
    liks = np.array([0.0] * (M + 1))
    for m in range(M):
        y_ = self.y.copy()[-1 * minibatch:]
        y_mod.append(y_)

    if not classify:
        self.y.append(y)
        self.x_train.append(x_train)

    y_mod, liks = _alignment.configured(self).prepare_online(
        self, x_train, y, t, M, y_mod, liks, with_warp, force_model, fast=True)

    # --- Build q_aux/q_lat using cached history; only compute last-row SSE ---
    q_aux = torch.zeros(self.T, M + 1, self.n_outputs, device=self.device) - np.inf
    q_lat = torch.zeros(self.T, M + 1, self.n_outputs, device=self.device, dtype=torch.float64)

    if t > 0:
        q_aux[:-1, :self.q_last.shape[1], :] = self.q_last
        q_lat[:-1, :self.q_lat_last.shape[1], :] = self.q_lat_last

    for ld in range(self.n_outputs):
        for m, gp in enumerate(self.gpmodels[ld]):
            q_aux[-1, m, ld] = gp.log_sq_error(
                self.cond_cuda(self.cond_to_torch(torch.from_numpy(np.array(self.x_train[-1])))),
                y_mod[m][-1],
                i=-1
            )

    # --- SPECIAL CASE: first sample (t == 0) ---
    if t == 0:
        init_state = 0 if (force_model is None) else int(force_model)

        # Hard resp for first time step
        resp = torch.zeros((self.T, M + 1), device=self.device, dtype=torch.float64)
        resp[0, init_state] = 1.0

        respPair = torch.zeros((self.T, M + 1, M + 1), device=self.device, dtype=torch.float64)
        respPair[0, init_state, init_state] = 1.0  # safe self-transition

        q_chos, q_lat_chos = q_aux, q_lat

        # Initialize caches for next iterations (only real models, exclude birth column)
        self.resp_last = resp[:, :M].detach()
        self.respPair_last = respPair[:, :M, :M].detach()
        self.q_last = q_chos[:, :M, :].detach()
        self.q_lat_last = q_lat_chos[:, :M, :].detach()

        # keep bookkeeping consistent with update
        self.resp_assigned.append(torch.argmax(resp[:, :M], dim=1))
        self.q.append(q_chos.detach())
    else:
        # --- Baseline (previous step) score (used for deltas) ---
        base_q, base_elbo = self.compute_q_elbo(
            self.resp_last, self.respPair_last,
            self.weight_mean(self.q_last, self.snr_norm[:-1, np.newaxis]),
            self.weight_mean(self.q_lat_last, self.snr_norm[:-1, np.newaxis]),
            self.gpmodels, self.M,
            snr='saved', post=False, one_sample=True, verb=False
        )
        base_total = (base_q + base_elbo).detach()

        # Default choice: no birth, just keep q_aux/q_lat and pick best SSE model
        q_chos, q_lat_chos = q_aux, q_lat
        m_best_sse = int(torch.argmax(self.weight_mean(q_aux)[-1, :-1]).item())
        resp_h, respPair_h = self._append_hard_step(self.resp_last, self.respPair_last, new_state=m_best_sse, K=M)
        resp = torch.zeros((self.T, M + 1), device=self.device, dtype=torch.float64)
        resp[:, :M] = resp_h
        respPair = torch.zeros((self.T, M + 1, M + 1), device=self.device, dtype=torch.float64)
        respPair[:, :M, :M] = respPair_h

    # No comparison if first sample or classify
    if t > 0 and (not classify) and (force_model is None):
        # Order candidates by current SSE (same as update)
        q_ord = torch.argsort(self.weight_mean(q_aux)[-1, :-1], descending=True)
        m_template = int(
            q_ord[-1].item())  # template for prov_gp (same as update) :contentReference[oaicite:5]{index=5}

        # ========== Birth candidate ==========
        q_prev = torch.clone(q_aux)
        q_lat_prev = torch.clone(q_lat)
        prov_gps = []

        for ld in range(self.n_outputs):
            prov_gp = self.gpmodel_deepcopy(self.gpmodels[ld][m_template])
            prov_gp.reinit_GP(save_last=False)
            prov_gp.reinit_LDS(save_last=False)

            # Keep original behavior: birth uses y_mod[-1][-1] :contentReference[oaicite:6]{index=6}
            y_birth = y_mod[-1][-1]
            q_prev[-1, -1, ld] = self.estimate_new(t, prov_gp, self.x_train[-1], y_birth, h=1.0)

            prov_gp.include_weighted_sample(t, self.x_train[-1], self.x_train[-1], y_birth, 1.0)
            # q_lat for a new model is non-zero only at t (indexes contains only t) :contentReference[oaicite:7]{index=7}
            if self.model_type_def == 'dynamic':
                q_lat_prev[:, -1, ld] = self._update_q_lat_tail(prov_gp, q_lat_prev[:, -1, ld], [t], h_ini=0.5) * 5.0

            prov_gps.append(prov_gp)

        # Gate (same intent as original): only compare absorb if birth is best by emission :contentReference[oaicite:8]{index=8}
        if int(torch.argmax(self.weight_mean(q_prev)[-1]).item()) == M:
            # Forced hard resp for birth
            resp_birth, respPair_birth = self._append_hard_step(self.resp_last, self.respPair_last, new_state=M,
                                                                K=M + 1)
            gpmodels_birth = [[] for _ in range(self.n_outputs)]
            for ld in range(self.n_outputs):
                gpmodels_birth[ld] = list(self.gpmodels[ld]) + [prov_gps[ld]]

            q_b, elbo_b = self.compute_q_elbo(
                resp_birth, respPair_birth,
                self.weight_mean(q_prev), self.weight_mean(q_lat_prev),
                gpmodels_birth, M + 1,
                snr='saved', post=True, one_sample=True, verb=False
            )
            best_total = (q_b + elbo_b).detach() - base_total
            best_kind = ("birth", None)
            best_pack = (q_prev, q_lat_prev, resp_birth, respPair_birth)

            # ========== Absorb candidates (ordered) ==========
            for m_cand in q_ord:
                m_cand = int(m_cand.item())
                q_post = torch.clone(q_aux)
                q_lat_post = torch.clone(q_lat)

                gpmodels_post = [[] for _ in range(self.n_outputs)]
                for ld in range(self.n_outputs):
                    gpmodels_post[ld] = list(self.gpmodels[ld])
                    post_gp = self.gpmodel_deepcopy(self.gpmodels[ld][m_cand])

                    # Use estimate_new so the sample is assumed INCLUDED (this is what you asked to restore) :contentReference[oaicite:9]{index=9}
                    q_post[-1, m_cand, ld] = self.estimate_new(t, post_gp, self.x_train[-1], y_mod[m_cand][-1],
                                                               h=1.0)

                    post_gp.include_weighted_sample(t, self.x_train[-1], self.x_train[-1], y_mod[m_cand][-1], 1.0)
                    post_gp.backwards_pair(1.0)  # tail-only smoothing :contentReference[oaicite:10]{index=10}
                    if self.bayesian_params:
                        post_gp.bayesian_new_params(1.0, model_type=self.model_type[m_cand])
                    else:
                        post_gp.new_params_weighted(
                            1.0, batch=None, min_samples=self.min_samples, max_samples=self.max_samples,
                            div_samples=self.div_samples, verbose=False, model_type=self.model_type[m_cand],
                            check_var=self.check_var
                        )

                    # Patch q_lat only at tail indices (t and maybe t-1) instead of compute_q_lat_all :contentReference[oaicite:11]{index=11}
                    if self.model_type[m_cand] == 'dynamic':
                        upd = [t]
                        if (t - 1) >= 0:
                            upd.append(t - 1)
                        q_lat_post[:, m_cand, ld] = self._update_q_lat_tail(post_gp, q_lat_post[:, m_cand, ld], upd,
                                                                            h_ini=1.0)

                    gpmodels_post[ld][m_cand] = post_gp

                # Forced hard resp for absorb (no birth column in scoring, like update does with [:,:-1]) :contentReference[oaicite:12]{index=12}
                resp_abs, respPair_abs = self._append_hard_step(self.resp_last, self.respPair_last,
                                                                new_state=m_cand, K=M)

                q_a, elbo_a = self.compute_q_elbo(
                    resp_abs, respPair_abs,
                    self.weight_mean(q_post)[:, :M],
                    self.weight_mean(q_lat_post)[:, :M],
                    gpmodels_post, M,
                    snr='saved', post=False, one_sample=True, verb=False
                )
                absorb_total = (q_a + elbo_a).detach() - base_total

                if absorb_total > best_total:
                    best_total = absorb_total
                    best_kind = ("absorb", m_cand)
                    # Expand absorb resp to M+1 columns for downstream compatibility
                    resp_full = torch.zeros((self.T, M + 1), device=self.device, dtype=torch.float64)
                    resp_full[:, :M] = resp_abs
                    respPair_full = torch.zeros((self.T, M + 1, M + 1), device=self.device, dtype=torch.float64)
                    respPair_full[:, :M, :M] = respPair_abs
                    best_pack = (q_post, q_lat_post, resp_full, respPair_full)
                    break  # same behavior as original: accept first improving candidate

            q_chos, q_lat_chos, resp, respPair = best_pack

    # ---- Now continue like update: choose model, apply birth, update GPs, etc. ----
    resp_mod = self.cond_to_numpy(self.cond_cpu(resp[-1]))
    resp_modlog = np.log(np.maximum(resp_mod, 1e-12))

    if sum(np.isclose(resp_mod, max(resp_mod), rtol=1e-2)) > 1:
        h_argmax = np.nanargmax(resp_mod)
        resp_mod[:] = 0.0
        resp_mod[h_argmax] = 1.0

    model = int(np.argmax(resp_mod))
    if self.max_models is not None and model >= self.max_models:
        force_model = int(np.argmax(resp_modlog[:-1]))
        model = force_model

    if force_model is not None:
        resp_mod[:] = 0.0
        resp_mod[force_model] = 1.0
        model = int(np.argmax(resp_mod))

    birth = (model == self.M)
    if birth:
        self.M = self.M + 1
        M = self.M
        y_mod.append(self.y.copy())
        for ld in range(self.n_outputs):
            self.gpmodels[ld].append(self.create_gp_default())
            self.wp_sys[ld].append(self.create_wp_sys_default())
        self.x_basis.append(self.x_basis_ini)
    if force_model is None:
        resp, respPair, q_chos, q_lat_chos, reorder = self.reorder(resp, respPair, q_chos, q_lat_chos)

    # Update HDP global params
    M_eff = self.M
    startStateCount = resp[0, :M_eff]
    transStateCount = torch.sum(respPair[:, :M_eff, :M_eff], axis=0)
    if M_eff > 2:
        self.reinit_global_params(M_eff - 1, transStateCount, startStateCount)
    if M_eff >= 2:
        for _ in range(4):
            self.transTheta, self.startTheta = self._calcThetaFull(self.cond_cuda(transStateCount),
                                                                   self.cond_cuda(startStateCount), M_eff)
            self.rho, self.omega = self.find_optimum_rhoOmega()

    # Update actual GP models with last hard assignment
    self.actual_state = model

    if minibatch == 0:
        minibatch = None

    for ld in range(self.n_outputs):
        for m in range(self.M):
            h = float(resp_mod[m]) if m < len(resp_mod) else 0.0
            new_x_basis = self.gpmodels[ld][m].include_weighted_sample(
                t, self.x_train[-1], self.x_train[-1],
                y_mod[reorder[m]][-1] if m < len(y_mod) else y,
                h
            )
            self.x_basis[m] = self.cond_cuda(self.cond_to_torch(new_x_basis.detach()))
            if h == 1.0:
                self.y_train = torch.concatenate(
                    [self.y_train, torch.atleast_2d(y_mod[reorder[m]][-1]).T[:, :, np.newaxis]])

            if self.bayesian_params:
                self.gpmodels[ld][m].bayesian_new_params(h, model_type=self.model_type[m])
            else:
                self.gpmodels[ld][m].new_params_weighted(h, batch=minibatch, min_samples=self.min_samples,
                                                         max_samples=self.max_samples,
                                                         div_samples=self.div_samples,
                                                         verbose=False, model_type=self.model_type[m],
                                                         check_var=self.check_var)

    # Store history + refresh caches for next call
    self.resp_assigned.append(torch.argmax(resp[:, :self.M], axis=1))
    self.q.append(q_chos.detach())

    self.q_last = q_chos[:, :self.M, :].detach()
    self.q_lat_last = q_lat_chos[:, :self.M, :].detach()
    self.resp_last = resp[:, :self.M].detach()
    self.respPair_last = respPair[:, :self.M, :self.M].detach()

    for ld in range(self.n_outputs):
        _alignment.configured(self).update_selected(self, ld, model, x_train, with_warp)
    _report_assignment(self, t, model, birth)

