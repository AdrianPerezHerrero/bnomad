"""Identity alignment with neutral scores and no optimizer or GP prior state."""

import torch


class _ZeroWarpScore:
    def log_sq_error(self, x_model, x_warp):
        x = torch.as_tensor(x_model)
        return torch.zeros((), device=x.device, dtype=x.dtype)

    def log_sq_error_batch(self, x_model, x_warp_batch):
        x = torch.as_tensor(x_model)
        w = torch.as_tensor(x_warp_batch)
        if w.ndim == 3:
            w = w[..., 0]
        count = w.shape[1] if w.shape[0] == x.numel() and w.shape[1] != x.numel() else w.shape[0]
        return torch.zeros(count, device=x.device, dtype=x.dtype)

    def model_to_numpy(self):
        pass

    model_to_torch = model_to_numpy
    model_to_cuda = model_to_numpy
    model_to_cpu = model_to_numpy


class _IdentityWarp:
    """Per-cluster identity object retaining the legacy warper call surface.

    Identity output preserves target tensor storage/gradients. Displacements and
    scores are zero; trace lists are empty because no optimization takes place.
    """

    def __init__(self):
        self.warp_gp = _ZeroWarpScore()

    def compute_warp(self, x_model, y_target, y_model, theta=None, noise=None,
                     visualize=False, verbose=False, train_iter=50):
        x = torch.as_tensor(x_model)
        y = torch.as_tensor(y_target)
        if y.ndim == 1:
            y = y[:, None]
        offset = torch.zeros((x.numel(), 1), device=x.device, dtype=x.dtype)
        score = torch.zeros((), device=x.device, dtype=x.dtype)
        return offset, y, score, {name: [] for name in ("loss", "data", "smooth", "amp")}

    def compute_warp_batch(self, x_model, y_target_batch, y_model, theta=None,
                           noise=None, weights=None, visualize=False, verbose=False, train_iter=50):
        x = torch.as_tensor(x_model)
        y = torch.as_tensor(y_target_batch)
        if y.ndim == 1:
            y = y[None, :, None]
        elif y.ndim == 2:
            y = y[:, :, None]
        offset = torch.zeros((y.shape[0], x.numel(), 1), device=x.device, dtype=x.dtype)
        score = torch.zeros(y.shape[0], device=x.device, dtype=x.dtype)
        return offset, y, score, {name: [] for name in ("loss", "data", "smooth", "amp")}

    def update_warp(self, x_train, x_warp):
        pass

    def reset(self):
        pass


def candidates(owner, x_train, y, strategie="standard", force_model=None, gpmodel=None, i=None, ld=0):
    y = owner.cond_cuda(owner.cond_to_torch(y))
    x = owner.cond_cuda(owner.cond_to_torch(x_train))
    offsets = [torch.zeros((len(x), 1), device=x.device, dtype=x.dtype)] * owner.M
    return [y] * owner.M, offsets, torch.zeros(owner.M, device=x.device, dtype=x.dtype)


def assigned_state(owner, x_trains, y_trains, q=None, q_lat=None, snr=None):
    y = owner.cond_cuda(owner.cond_to_torch(y_trains))
    x = owner.cond_cuda(owner.cond_to_torch(x_trains))
    owner.x_w = torch.zeros_like(y)
    owner.liks_w = torch.zeros((y.shape[0], owner.n_outputs), device=y.device, dtype=y.dtype)
    if q is not None:
        for ld in range(owner.n_outputs):
            for m, gp in enumerate(owner.gpmodels[ld]):
                q[:, m, ld] = gp.compute_sq_err_all(x, y[:, :, ld])
                q_lat[:, m, ld] = gp.compute_q_lat_all(x)
    owner.y_train = y
    return q, q_lat, True, y


def uncached_batch(owner, x_trains, y_trains, resp_temp, gpmodels=None, f_ind_old=None,
                   train_iter=50, batch_size=128):
    y = owner.cond_cuda(owner.cond_to_torch(y_trains))
    resp = owner.cond_cuda(owner.cond_to_torch(resp_temp))
    N, T, outputs = y.shape
    M = resp.shape[1]
    assert outputs == owner.n_outputs
    return (y.unsqueeze(-1).expand(-1, -1, -1, M),
            torch.zeros((N, T, outputs, M), device=owner.device, dtype=torch.float64),
            torch.zeros((N, M, outputs), device=owner.device, dtype=torch.float64))
