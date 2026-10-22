"""Original model/warper integration, with caller-owned histories and caches.

The explicit state argument retains the legacy name self. These functions import
no model class, retain old candidate strategies and shapes, and do not own the
inference schedule. Enabled/disabled online branches are copied without arithmetic
changes; stateless policies select them at the model boundary.
"""

import numpy as np
import torch


def reset_warp_cache(self):
    self._warp_cache_state = None
    self._warp_cache_ref = None
    self._y_trains_w_cache = None


def compute_warp_actual_state(self, x_trains, y_trains, q=None, q_lat=None, snr=None):
    """ Compute warp for every sample with an assignation to a gpmodel in the actual state of the model.
    """
    y_trains = self.cond_cuda(self.cond_to_torch(y_trains))
    x_trains = self.cond_cuda(self.cond_to_torch(x_trains))
    y_trains_w = torch.clone(y_trains)
    self.x_w = torch.zeros(y_trains.shape)
    self.liks_w = torch.zeros((y_trains.shape[0], self.n_outputs))
    for ld in range(self.n_outputs):
        for j in range(y_trains.shape[0]):
            y = y_trains[j, :, [ld]]
            m_ = 0
            for m, gp in enumerate(self.gpmodels[ld]):
                if j in gp.indexes:
                    m_ = m
            y_w, x_w, lik = self.compute_warp_y(x_trains[j], y, force_model=m_,
                                                gpmodel=self.gpmodels[ld][m_])
            y_trains_w[j, :, [ld]] = y_w[m_]
            self.x_w[j, :, [ld]] = x_w[m_]
            self.liks_w[j, ld] = lik[m_]
        if q is not None:
            for m, gp in enumerate(self.gpmodels[ld]):
                q[:, m, ld] = gp.compute_sq_err_all(x_trains, y_trains_w[:, :, ld])
                q_lat[:, m, ld] = gp.compute_q_lat_all(x_trains)
    warp_computed = True
    self.y_train = y_trains_w
    return q, q_lat, warp_computed, y_trains_w


def compute_warp_actual_state_amtgp(self, x_trains, y_trains, q=None, q_lat=None, snr=None):
    """ Compute warp for every sample with an assignation to a gpmodel in the actual state of the model.
    """
    y_trains = self.cond_cuda(self.cond_to_torch(y_trains))
    x_trains = self.cond_cuda(self.cond_to_torch(x_trains))
    y_trains_w = torch.clone(y_trains)
    self.x_w = torch.zeros(y_trains.shape)
    self.liks_w = torch.zeros((y_trains.shape[0], self.n_outputs))
    # inside compute_warp_actual_state(...)

    for ld in range(self.n_outputs):
        for m, gp in enumerate(self.gpmodels[ld]):
            idx = torch.tensor(gp.indexes, device=self.device).long()
            if idx.numel() == 0:
                continue

            # requirement: common grid for the batch
            x0 = x_trains[idx[0]]  # (T,)
            # (optional safety) if grids differ, split by unique grids or fallback to per-sample

            mean, cov = gp.observe_last(x0)
            noise = torch.diag(cov)  # as you already do in compute_warp_y :contentReference[oaicite:8]{index=8}

            yB = y_trains[idx, :, [ld]]  # (B,T,1)

            xwB, ywB, likB, _ = self.wp_sys[ld][m].compute_warp_batch(
                x0, yB, mean,
                theta=gp.gp.kernel.get_params()["k1__k2__length_scale"],
                noise=noise, train_iter=250
            )

            # Add the same baseline normalization you already use:
            # liks[m] = lik_ + wp_sys[ld][-1].warp_gp.log_sq_error(...) :contentReference[oaicite:9]{index=9}
            baseB = self.wp_sys[ld][-1].warp_gp.log_sq_error_batch(x0, xwB[:, :, 0])
            likTot = likB + baseB

            y_trains_w[idx, :, [ld]] = ywB[:,:,0]  # assimilate warped observations
            self.x_w[idx, :, [ld]] = xwB[:,:,0]  # store warps
            self.liks_w[idx, ld] = likTot  # store warp likelihoods
        if q is not None:
            for m, gp in enumerate(self.gpmodels[ld]):
                q[:, m, ld] = gp.compute_sq_err_all(x_trains, y_trains_w[:, :, ld])
                q_lat[:, m, ld] = gp.compute_q_lat_all(x_trains)
    warp_computed = True
    self.y_train = y_trains_w
    return q, q_lat, warp_computed, y_trains_w


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
    M = self.M
    l = len(x_train)
    x_w = [self.cond_cuda(self.cond_to_torch(np.atleast_2d(np.zeros(l)).T))] * M
    y_w = [self.cond_cuda(self.cond_to_torch(y))] * M
    base = self.wp_sys[ld][-1].warp_gp.log_sq_error(x_train, self.cond_cuda(self.cond_to_torch(x_w[-1]))).item()
    liks = torch.from_numpy(np.array([base] * M))
    if strategie=='greedy' or strategie=='greedy_bound':
        q_C = np.zeros(len(self.gpmodels[0]))
        for g, gpmodel in enumerate(self.gpmodels[0]):
            q_C[g] = self.estimate_new(-1, gpmodel, x_train=self.x_train[-1], y=y)
    def trans_noise(noise):
        if torch.mean(noise) > 1.0:
            noise = noise
        else:
            noise = noise
        if self.model_type[m] == 'static':
            noise = noise * 0.5
        else:
            noise = noise * 1.0
        return noise

    if not force_model is None:
        m = force_model
        if gpmodel is None:
            model = self.gpmodels[ld][m]
        else:
            model = gpmodel
        wp_sys_ = self.wp_sys[ld][m]
        if len(model.indexes) != 0:
            if i is None:
                mean, cov = model.observe_last(x_train)
            else:
                mean, cov = model.observe(x_train, t=i)
            noise = torch.diag(cov)
            noise = trans_noise(noise)
            x_, y_, lik_, loss_ = wp_sys_.compute_warp(self.cond_detach(x_train), self.cond_detach(y),
                                                       self.cond_detach(mean),
                                                       model.gp.kernel.get_params()["k1__k2__length_scale"],
                                                       noise=noise,
                                                       visualize=False, verbose=False,
                                                       train_iter=250)
            y_w[m] = y_
            x_w[m] = x_
            liks[m] = lik_ + self.wp_sys[ld][-1].warp_gp.log_sq_error(x_train,
                                                                      self.cond_cuda(self.cond_to_torch(x_))).item()
        else:
            liks[m] = liks[m] + torch.max(liks[:-1])
        liks[-1] = liks[-1] + torch.max(liks[:-1])
    elif strategie == 'standard':
        for m in range(len(self.gpmodels[ld])):
            model = self.gpmodels[ld][m]
            wp_sys_ = self.wp_sys[ld][m]
            if len(model.indexes) != 0:
                if self.verbose:
                    print("Warp respect model: ", m + 1)
                mean, cov = model.observe_last(x_train)
                noise = torch.diag(cov)
                noise = trans_noise(noise)
                x_, y_, lik_, loss_ = wp_sys_.compute_warp(self.cond_detach(x_train), self.cond_detach(y),
                                                       self.cond_detach(mean),
                                                       model.gp.kernel.get_params()["k1__k2__length_scale"],
                                                       noise=noise, visualize=False, verbose=self.verbose,
                                                       train_iter=250)
                y_w[m] = y_
                x_w[m] = x_
                liks[m] = lik_ + self.wp_sys[ld][-1].warp_gp.log_sq_error(x_train,
                                                                          self.cond_cuda(
                                                                              self.cond_to_torch(x_))).item()
            else:
                liks[m] = liks[m] + torch.max(liks[:-1])
        liks[-1] = liks[-1] + torch.max(liks[:-1])
    elif strategie == 'greedy_bound':
        order_C = np.argsort(-q_C)
        for i in range(len(order_C)):
            m = order_C[i]
            model = self.gpmodels[ld][m]
            wp_sys_ = self.wp_sys[ld][m]
            if len(model.indexes) != 0:
                if self.verbose:
                    print("Warp respect model: ", m + 1)
                mean, cov = model.observe_last(x_train)
                noise = torch.diag(cov)
                noise = trans_noise(noise)
                x_, y_, lik_, loss_ = wp_sys_.compute_warp(self.cond_detach(x_train), self.cond_detach(y),
                                                           self.cond_detach(mean),
                                                           model.gp.kernel.get_params()["k1__k2__length_scale"],
                                                           noise=noise, visualize=False, verbose=self.verbose,
                                                           train_iter=250)
                y_w[m] = y_
                x_w[m] = x_
                # Normalize over log_sq from 0 knowledge model
                liks[m] = lik_ + self.wp_sys[ld][-1].warp_gp.log_sq_error(x_train,
                                                                          self.cond_cuda(
                                                                              self.cond_to_torch(x_))).item()
            else:
                liks[m] = liks[m] + torch.max(liks[order_C[:i+1]])
                if self.verbose:
                    print("Warp respect zero-knowledge model:", m + 1)
            if i >= 3:
                break
        liks[-1] = liks[-1] + torch.max(liks[:-1])
    elif strategie == 'greedy':
        order_C = np.argsort(-q_C)
        for i in range(len(order_C)):
            m = order_C[i]
            model = self.gpmodels[ld][m]
            wp_sys_ = self.wp_sys[ld][m]
            if len(model.indexes) != 0:
                if self.verbose:
                    print("Warp respect model: ", m + 1)
                mean, cov = model.observe_last(x_train)
                noise = torch.diag(cov)
                noise = trans_noise(noise)
                x_, y_, lik_, loss_ = wp_sys_.compute_warp(self.cond_detach(x_train), self.cond_detach(y),
                                                       self.cond_detach(mean),
                                                       model.gp.kernel.get_params()["k1__k2__length_scale"],
                                                       noise=noise,
                                                       visualize=False, verbose=self.verbose,
                                                       train_iter=250)

                y_w[m] = y_
                x_w[m] = x_
                # Normalize over log_sq from 0 knowledge model
                basis_lik = self.wp_sys[ld][-1].warp_gp.log_sq_error(x_train,
                                                                     self.cond_cuda(self.cond_to_torch(x_))).item()
                liks[m] = lik_ + basis_lik
                liks_ = np.array(liks[m]) * 0.5
                if i < len(order_C) - 1 and i < 8:
                    first = order_C[0]
                    if ((q_C[m] + liks_ - q_C[order_C[i + 1]]) / (q_C[m] - q_C[order_C[i + 1]]) > 0.3 / (
                            np.log(max(model.N, 1) + 1))) or i == 5:
                        for j_ in order_C[i + 1:]:
                            liks[j_] = liks[j_] + torch.max(liks[order_C[:i+1]])
                        liks[-1] = liks[-1] + torch.max(liks[:-1])
                        break
                else:
                    for j_ in order_C[i + 1:]:
                        liks[j_] = liks[j_] + torch.max(liks[order_C[:i+1]])
                    liks[-1] = liks[-1] + torch.max(liks[:-1])
                    break
            else:
                liks[m] = liks[m] + torch.max(liks[:-1])
                if self.verbose:
                    print("Warp respect zero-knowledge model:", m + 1)

    else:
        print('Only two strategies implemented: standard and greedy')
    return y_w, x_w, liks


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
    import torch

    x_trains = self.cond_cuda(self.cond_to_torch(x_trains))
    y_trains = self.cond_cuda(self.cond_to_torch(y_trains))
    resp_temp = self.cond_cuda(self.cond_to_torch(resp_temp))

    if gpmodels is None:
        gpmodels = self.gpmodels
    if f_ind_old is None:
        f_ind_old = self.f_ind_old

    N, T, D_out = y_trains.shape
    M = resp_temp.shape[1]
    assert D_out == self.n_outputs

    # NEW: keep one warped version per cluster
    y_trains_w = y_trains.unsqueeze(-1).repeat(1, 1, 1, M)  # (N,T,D,M)
    x_w = torch.zeros((N, T, self.n_outputs, M), device=self.device, dtype=torch.float64)
    liks_full = torch.zeros((N, M, self.n_outputs), device=self.device, dtype=torch.float64)

    def _base_score_batch(ld, x0, xwB_2d):
        base_gp = self.wp_sys[ld][-1].warp_gp
        if hasattr(base_gp, "log_sq_error_batch"):
            return base_gp.log_sq_error_batch(x0, xwB_2d)
        out = torch.zeros((xwB_2d.shape[0],), device=xwB_2d.device, dtype=torch.float64)
        for i in range(xwB_2d.shape[0]):
            out[i] = base_gp.log_sq_error(
                x0,
                self.cond_cuda(self.cond_to_torch(xwB_2d[i][:, None]))
            ).reshape(())
        return out

    for ld in range(self.n_outputs):
        for m in range(M):
            # Important: columns of resp_temp are already the active cluster order
            state_raw = m

            state_idx_gp = min(state_raw, len(gpmodels[ld]) - 1)
            state_idx_wp = min(state_raw, len(self.wp_sys[ld]) - 1)
            state_idx_f = min(state_raw, int(f_ind_old.shape[0]) - 1)

            ref = int(f_ind_old[state_idx_f].item())

            x0 = x_trains[ref]  # (T,)
            y_model = y_trains[ref, :, [ld]]  # (T,1)

            gp = gpmodels[ld][state_idx_gp]
            theta = gp.gp.kernel.get_params()["k1__k2__length_scale"]

            noise_scalar = torch.mean(torch.diag(gp.Sigma[-1])).detach()
            noise_vec = noise_scalar * torch.ones(T, device=x0.device, dtype=x0.dtype)

            warper = self.wp_sys[ld][state_idx_wp]

            for s in range(0, N, batch_size):
                idx = torch.arange(s, min(s + batch_size, N), device=self.device)
                yB = y_trains[idx, :, [ld]]  # (B,T,1)

                xwB, ywB, likB, _ = warper.compute_warp_batch(
                    x0, yB, y_model,
                    theta=theta,
                    noise=noise_vec,
                    train_iter=train_iter,
                    verbose=False,
                )

                baseB = _base_score_batch(ld, x0, xwB[:, :, 0])
                likTot = likB + baseB

                # FULL competition term
                liks_full[idx, m, ld] = likTot

                # NEW: store warped batch for THIS cluster m
                y_trains_w[idx, :, ld, m] = ywB[:, :, 0]
                x_w[idx, :, ld, m] = xwB[:, :, 0]

    return y_trains_w, x_w, liks_full


def warp_batch_by_resp_amtgp_cached(
        self,
        x_trains,
        y_trains,
        resp_temp,  # (N,M)
        f_ind_old=None,
        train_iter: int = 50,
        batch_size: int = 128,
):

    x_trains = self.cond_cuda(self.cond_to_torch(x_trains))
    y_trains = self.cond_cuda(self.cond_to_torch(y_trains))
    resp_temp = self.cond_cuda(self.cond_to_torch(resp_temp))

    if f_ind_old is None:
        f_ind_old = self.f_ind_old

    N, T, D_out = y_trains.shape
    M = resp_temp.shape[1]
    assert D_out == self.n_outputs

    if not hasattr(self, "_warp_cache_full"):
        self._warp_cache_full = {}
    #     self._warp_cache_full_shape = None
    #
    # if self._warp_cache_full_shape != (N, T, self.n_outputs, M):
    #     self._warp_cache_full.clear()
    #     self._warp_cache_full_shape = (N, T, self.n_outputs, M)

    if not self.warp:
        y_trains_w = y_trains.unsqueeze(-1).expand(-1, -1, -1, M)
        x_w = x_trains.unsqueeze(-1).expand(-1, -1, -1, M)
        liks_full = torch.zeros((N, M, self.n_outputs),
                                device=self.device, dtype=torch.float64)
        return y_trains_w, x_w, liks_full

    y_trains_w = torch.empty((N, T, D_out, M), device=self.device, dtype=torch.float64)
    x_w = torch.empty((N, T, self.n_outputs, M), device=self.device, dtype=torch.float64)
    liks_full = torch.zeros((N, M, self.n_outputs), device=self.device, dtype=torch.float64)

    def _base_score_batch(ld, x0, xwB_2d):
        base_gp = self.wp_sys[ld][-1].warp_gp
        if hasattr(base_gp, "log_sq_error_batch"):
            return base_gp.log_sq_error_batch(x0, xwB_2d)
        out = torch.zeros((xwB_2d.shape[0],), device=xwB_2d.device, dtype=torch.float64)
        for i in range(xwB_2d.shape[0]):
            out[i] = base_gp.log_sq_error(
                x0,
                self.cond_cuda(self.cond_to_torch(xwB_2d[i][:, None]))
            ).reshape(())
        return out

    for ld in range(self.n_outputs):
        for m in range(M):
            state_raw = m

            state_idx_wp = min(state_raw, len(self.wp_sys[ld]) - 1)
            state_idx_f = state_raw

            ref = int(f_ind_old[state_idx_f].item())


            key = (ld, ref)

            if key in self._warp_cache_full:
                xw_all_2d, yw_all_2d, lik_all = self._warp_cache_full[key]
            else:
                x0 = x_trains[ref]
                y_model = y_trains[ref, :, [ld]]

                noise_scalar = np.sqrt(self.ini_sigma_def)
                noise_vec = noise_scalar * torch.ones(T, device=x0.device, dtype=x0.dtype)

                warper = self.wp_sys[ld][state_idx_wp]

                xw_all_2d = torch.zeros((N, T), device=self.device, dtype=torch.float64)
                yw_all_2d = torch.zeros((N, T), device=self.device, dtype=torch.float64)
                lik_all = torch.zeros((N,), device=self.device, dtype=torch.float64)

                #batch_size = N
                for s in range(0, N, batch_size):
                    idx = torch.arange(s, min(s + batch_size, N), device=self.device)
                    yB = y_trains[idx, :, [ld]]

                    xwB, ywB, likB, _ = warper.compute_warp_batch(
                        x0, yB, y_model,
                        theta=self.kernel_def.get_params()["k1__k2__length_scale"],
                        noise=noise_vec,
                        train_iter=train_iter,
                        verbose=False,
                    )

                    baseB = _base_score_batch(ld, x0, xwB[:, :, 0])
                    likTot = likB + baseB

                    xw_all_2d[idx, :] = xwB[:, :, 0]
                    yw_all_2d[idx, :] = ywB[:, :, 0]
                    lik_all[idx] = likTot

                self._warp_cache_full[key] = (xw_all_2d, yw_all_2d, lik_all)

            liks_full[:, m, ld] = lik_all
            y_trains_w[:, :, ld, m] = yw_all_2d
            x_w[:, :, ld, m] = xw_all_2d

    return y_trains_w, x_w, liks_full


def select_assigned_warp(self, y_trains_w, resp):
    # y_trains_w: (N,T,D,M)
    if y_trains_w.ndim == 3:
        return y_trains_w
    z = torch.argmax(resp, dim=1)  # (N,)
    idx = z[:, None, None, None].expand(-1, y_trains_w.shape[1], y_trains_w.shape[2], 1)
    return torch.gather(y_trains_w, 3, idx).squeeze(3)  # (N,T,D)

def prepare_online_standard(self, x_train, y, t, M, y_mod, liks, warp, force_model=None):
    if warp:
        if t > 0:
            for ld in range(self.n_outputs):
                y_w, x_w, liks = self.compute_warp_y(x_train, y[:, [ld]], self.method_compute_warp,
                                                     force_model=force_model, ld=ld)
                for m in range(M):
                    # TODO: here incorporate n_outputs dimension to y to allow the iteration over the rest of the model.
                    y_mod[m].append(self.cond_cuda(y_w[m]))
                self.y_w.append(y_w)
                self.x_w.append(x_w)
                self.liks.append(liks)
        else:
            for m in range(M):
                y_mod[m].append(self.cond_cuda(y))
            self.y_w.append([self.cond_cuda(y)] * (M))
            self.x_w.append([self.cond_cuda(torch.atleast_2d(torch.zeros(y.shape[0])).T)] * (M))
            self.liks.append(liks)
    else:
        for m in range(M):
            y_mod[m].append(self.cond_cuda(y))
    return y_mod, liks


def update_selected(self, ld, model, x_train, warp):
    if len(self.gpmodels[ld][model].indexes) > 1 and self.warp_updating[model] and warp:
        self.wp_sys[ld][model].update_warp(x_train, self.x_w[-1][model])


def prepare_online_fast(self, x_train, y, t, M, y_mod, liks, warp, force_model=None):
    if warp:
        if t > 0:
            for ld in range(self.n_outputs):
                y_w, x_w, liks = self.compute_warp_y(
                    x_train, y[:, [ld]], self.method_compute_warp, force_model=force_model, ld=ld
                )
                for m in range(M):
                    y_mod[m].append(self.cond_cuda(y_w[m]))
                self.y_w.append(y_w)
                self.x_w.append(x_w)
                self.liks.append(liks)
        else:
            for m in range(M):
                y_mod[m].append(self.cond_cuda(y))
            self.y_w.append([self.cond_cuda(y)] * M)
            self.x_w.append(
                [self.cond_cuda(torch.atleast_2d(torch.zeros(y.shape[0], device=self.device)).T)] * M)
            self.liks.append(liks)
    else:
        for m in range(M):
            y_mod[m].append(self.cond_cuda(y))
    return y_mod, liks


def identity_cached_batch(
        self,
        x_trains,
        y_trains,
        resp_temp,  # (N,M)
        f_ind_old=None,
        train_iter: int = 50,
        batch_size: int = 128,
):

    x_trains = self.cond_cuda(self.cond_to_torch(x_trains))
    y_trains = self.cond_cuda(self.cond_to_torch(y_trains))
    resp_temp = self.cond_cuda(self.cond_to_torch(resp_temp))

    if f_ind_old is None:
        f_ind_old = self.f_ind_old

    N, T, D_out = y_trains.shape
    M = resp_temp.shape[1]
    assert D_out == self.n_outputs

    if not hasattr(self, "_warp_cache_full"):
        self._warp_cache_full = {}
    #     self._warp_cache_full_shape = None
    #
    # if self._warp_cache_full_shape != (N, T, self.n_outputs, M):
    #     self._warp_cache_full.clear()
    #     self._warp_cache_full_shape = (N, T, self.n_outputs, M)
    y_trains_w = y_trains.unsqueeze(-1).expand(-1, -1, -1, M)
    x_w = x_trains.unsqueeze(-1).expand(-1, -1, -1, M)
    liks_full = torch.zeros((N, M, self.n_outputs),
                            device=self.device, dtype=torch.float64)
    return y_trains_w, x_w, liks_full

