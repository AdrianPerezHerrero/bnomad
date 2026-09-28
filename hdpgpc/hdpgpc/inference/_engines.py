"""Public schedule bindings; implementations are imported on first execution."""

from ._base import InferenceEngine


class OnlineInference(InferenceEngine):
    """One observation per call, using the standard or existing fast schedule.

    Both schedules preserve their own proposal evaluation and history revisits.
    There is no common loop with offline variational re-estimation.
    """

    __slots__ = ()

    def update(self, x_train, y, with_warp=True, force_model=None, minibatch=0, classify=False):
        """Run the existing update schedule on the bound owner."""
        from . import _online

        return _online.update(self._model, x_train, y, with_warp, force_model, minibatch, classify)

    def _init_resp_cache_cold_start(self, init_state: int=0):
        """Run the existing _init_resp_cache_cold_start schedule on the bound owner."""
        from . import _online

        return _online._init_resp_cache_cold_start(self._model, init_state)

    def _ensure_fast_cache_online(self):
        """Run the existing _ensure_fast_cache_online schedule on the bound owner."""
        from . import _online

        return _online._ensure_fast_cache_online(self._model)

    def _update_q_lat_tail(self, gp, q_lat_col, update_idxs, h_ini=1.0):
        """Run the existing _update_q_lat_tail schedule on the bound owner."""
        from . import _online

        return _online._update_q_lat_tail(self._model, gp, q_lat_col, update_idxs, h_ini)

    def _append_hard_step(self, resp_prev, respPair_prev, new_state, K):
        """Run the existing _append_hard_step schedule on the bound owner."""
        from . import _online

        return _online._append_hard_step(self._model, resp_prev, respPair_prev, new_state, K)

    def update_fast(self, x_train, y, with_warp=True, force_model=None, minibatch=0, classify=False):
        """Run the existing update_fast schedule on the bound owner."""
        from . import _online

        return _online.update_fast(self._model, x_train, y, with_warp, force_model, minibatch, classify)


class OfflineInference(InferenceEngine):
    """Repeated batch local/global updates and the existing batch utility paths.

    Proposal/refinement helpers are compatibility hooks, not interchangeable
    optimizers. Classification and label reload retain their distinct schedules.
    """

    __slots__ = ()

    def fit(self, x_trains, y_trains, it_limit=None, warp=False):
        """Run the existing fit schedule on the bound owner."""
        from . import _offline

        return _offline.fit(self._model, x_trains, y_trains, it_limit, warp)

    def variational_local_terms_batch(self, M, x_trains, y_trains, y_trains_w, transTheta, startTheta, resp, respPair, q, q_lat, snr, reallocate):
        """Run the existing variational_local_terms_batch schedule on the bound owner."""
        from . import _offline

        return _offline.variational_local_terms_batch(self._model, M, x_trains, y_trains, y_trains_w, transTheta, startTheta, resp, respPair, q, q_lat, snr, reallocate)

    def estimate_q_first(self, M, x_trains, y_trains, y_trains_w_, resp, respPair, q_, q_lat_, snr_, startPi, transPi, reallocate_=False, reparam=False):
        """Run the existing estimate_q_first schedule on the bound owner."""
        from . import _offline

        return _offline.estimate_q_first(self._model, M, x_trains, y_trains, y_trains_w_, resp, respPair, q_, q_lat_, snr_, startPi, transPi, reallocate_, reparam)

    def estimate_q_all(self, M, x_trains, y_trains, y_trains_w_, resp, respPair, q_, q_lat_, snr_, startPi, transPi, q_def, elbo_def, gpmodels=None, reparam=False, post=True, f_ind_old=None):
        """Run the existing estimate_q_all schedule on the bound owner."""
        from . import _offline

        return _offline.estimate_q_all(self._model, M, x_trains, y_trains, y_trains_w_, resp, respPair, q_, q_lat_, snr_, startPi, transPi, q_def, elbo_def, gpmodels, reparam, post, f_ind_old)

    def classify_batch(self, x_trains, y_trains, learning=False, it_limit=None, warp=False):
        """Run the existing classify_batch schedule on the bound owner."""
        from . import _offline

        return _offline.classify_batch(self._model, x_trains, y_trains, learning, it_limit, warp)

    def fit_labels(self, x_trains, y_trains, labels, M, warp=False):
        """Run the existing fit_labels schedule on the bound owner."""
        from . import _offline

        return _offline.fit_labels(self._model, x_trains, y_trains, labels, M, warp)
