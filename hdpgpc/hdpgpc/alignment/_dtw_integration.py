"""Use the existing candidate-conditioned input-grid routing for point DTW-GP."""

import numpy as np
import torch

from ._dtw_gp import DTWGPWarp
from ._flow_integration import GPFlowAlignment


class DTWGPAlignment(GPFlowAlignment):
    """Fast point warps: selected observation offline, current GP mean online."""

    supports_warp_posterior = False
    uses_posterior_flow = False
    supports_batch_lds_input_warp = True

    def create_warper(self, *args, **kwargs):
        cfg = self.config
        return DTWGPWarp(*args, **kwargs, points=cfg.dtw_points,
                         inducing=cfg.dtw_inducing,
                         lengthscale=cfg.dtw_lengthscale,
                         amplitude=cfg.dtw_amplitude,
                         anchor_noise=cfg.dtw_anchor_noise)

    def warp_batch_by_resp_amtgp_cached(self, owner, x_trains, y_trains,
                                        resp_temp, f_ind_old=None,
                                        train_iter=50, batch_size=128):
        """Align each offline candidate to its selected observation seed.

        The first candidate uses f_ind_old[0] (initially observation zero).
        A birth supplies its proposed observation in the new candidate slot.
        Fitting an LDS/GP does not change this reference; only selecting a new
        f_ind_old does. The waveform remains raw and the input grid is warped.
        """
        if not owner.warp:
            owner._dtw_offline_references = {}
            return super().warp_batch_by_resp_amtgp_cached(
                owner, x_trains, y_trains, resp_temp, f_ind_old,
                train_iter, batch_size)
        x = torch.as_tensor(x_trains, dtype=torch.float64, device=owner.device)
        y = torch.as_tensor(y_trains, dtype=torch.float64, device=owner.device)
        resp = torch.as_tensor(resp_temp, dtype=torch.float64, device=owner.device)
        n, points, outputs = y.shape
        clusters = resp.shape[1]
        selected = f_ind_old if f_ind_old is not None else getattr(owner, 'f_ind_old', ())
        selected = torch.as_tensor(selected, device=owner.device).reshape(-1)
        refs = []
        for m in range(clusters):
            # Label-conditioned fits can introduce more slots than the initial
            # representative vector. Initialize those from their assigned data.
            ref = int(selected[m]) if m < len(selected) else int(torch.argmax(resp[:, m]))
            if not 0 <= ref < n:
                raise ValueError(f'f_ind_old[{m}]={ref} is outside the offline batch')
            refs.append(ref)

        class ObservationReference:
            def __init__(self, grid, values):
                self.grid = grid.detach().cpu().numpy().reshape(-1)
                self.values = values.detach().cpu().numpy().reshape(-1)

            def mean_at(self, query):
                values = np.interp(query.detach().cpu().numpy().reshape(-1),
                                   self.grid, self.values)
                return torch.as_tensor(values, dtype=query.dtype,
                                       device=query.device)[:, None]

        offsets = y.new_empty((n, points, outputs, clusters))
        results, references = {}, {}
        for m, ref in enumerate(refs):
            templates = [ObservationReference(x[ref], y[ref, :, ld])
                         for ld in range(outputs)]
            references[m] = {'index': ref, 'grid': x[ref].detach().clone(),
                             'values': y[ref].detach().clone()}
            warper = owner.wp_sys[0][min(m, len(owner.wp_sys[0]) - 1)]
            for i in range(n):
                result = warper.infer(x[i], y[i], templates)
                results[i, m] = result
                offsets[i, :, :, m] = result.mean_displacement.expand(-1, outputs)
        owner._flow_offline_posteriors = results
        owner._dtw_offline_references = references
        return (y.unsqueeze(-1).expand(-1, -1, -1, clusters), offsets,
                y.new_zeros((n, clusters, outputs)))
