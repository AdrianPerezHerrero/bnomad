"""Use the existing candidate-conditioned input-grid routing for point DTW-GP."""

from ._dtw_gp import DTWGPWarp
from ._flow_integration import GPFlowAlignment


class DTWGPAlignment(GPFlowAlignment):
    """Fast point warps from current candidate GP means, online or offline."""

    supports_warp_posterior = False
    uses_posterior_flow = False

    def create_warper(self, *args, **kwargs):
        cfg = self.config
        return DTWGPWarp(*args, **kwargs, points=cfg.dtw_points,
                         band=cfg.dtw_band, inducing=cfg.dtw_inducing,
                         lengthscale=cfg.dtw_lengthscale,
                         amplitude=cfg.dtw_amplitude,
                         anchor_noise=cfg.dtw_anchor_noise)
