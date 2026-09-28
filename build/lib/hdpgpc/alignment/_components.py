"""Optional model-level policies around the existing per-cluster warpers.

All enable/disable decisions live here. Helpers receive the existing model state
to retain history/cache ownership. No optimization policy is changed.
"""

from . import _integration
from . import _identity


class GPAlignment:
    """Current AMTGP warping, honoring the legacy per-call request flags."""

    @staticmethod
    def enabled(requested):
        return requested

    @staticmethod
    def create_warper(*args, **kwargs):
        from ._amtgp import GaussianProcessWarp

        return GaussianProcessWarp(*args, **kwargs)

    @staticmethod
    def prepare_online(owner, x_train, y, t, M, y_mod, liks, requested, force_model=None, fast=False):
        method = _integration.prepare_online_fast if fast else _integration.prepare_online_standard
        return method(owner, x_train, y, t, M, y_mod, liks, requested, force_model)

    update_selected = staticmethod(_integration.update_selected)
    reset_warp_cache = staticmethod(_integration.reset_warp_cache)
    compute_warp_y = staticmethod(_integration.compute_warp_y)
    compute_warp_actual_state = staticmethod(_integration.compute_warp_actual_state)
    compute_warp_actual_state_amtgp = staticmethod(_integration.compute_warp_actual_state_amtgp)
    warp_batch_by_resp_amtgp = staticmethod(_integration.warp_batch_by_resp_amtgp)
    warp_batch_by_resp_amtgp_cached = staticmethod(_integration.warp_batch_by_resp_amtgp_cached)
    select_assigned_warp = staticmethod(_integration.select_assigned_warp)


class NoAlignment(GPAlignment):
    """Disable alignment for the model, including calls with warp flags enabled.

    Existing cached-batch callers still receive their legacy absolute-grid/view
    identity representation. Direct warper calls return zero displacements.
    """

    @staticmethod
    def enabled(requested):
        return False

    @staticmethod
    def create_warper(*args, **kwargs):
        return _identity._IdentityWarp()

    @staticmethod
    def prepare_online(owner, x_train, y, t, M, y_mod, liks, requested, force_model=None, fast=False):
        # This is precisely the old with_warp=False branch in both drivers.
        for m in range(M):
            y_mod[m].append(owner.cond_cuda(y))
        return y_mod, liks

    @staticmethod
    def update_selected(owner, ld, model, x_train, requested):
        pass

    compute_warp_y = staticmethod(_identity.candidates)
    compute_warp_actual_state = staticmethod(_identity.assigned_state)
    compute_warp_actual_state_amtgp = staticmethod(_identity.assigned_state)
    warp_batch_by_resp_amtgp = staticmethod(_identity.uncached_batch)
    warp_batch_by_resp_amtgp_cached = staticmethod(_integration.identity_cached_batch)


def configured(owner):
    """Return the explicit policy installed by native model initialization."""
    return owner.alignment
