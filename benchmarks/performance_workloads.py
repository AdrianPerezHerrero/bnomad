"""Deterministic performance workloads, separate from frozen test fixtures.

Construction is included. Result serialization is outside measured regions.
PERFORMANCE_SOURCE can select an isolated production source for paired comparison.
"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if os.environ.get('PERFORMANCE_SOURCE'):
    sys.path.insert(0, os.environ['PERFORMANCE_SOURCE'])
    import hdpgpc.model._state
from tests.characterization._support import (
    reset_runtime, encode, model_state, environment, np, torch, ModelState,
)
from tests.model_recipe import build_state
from hdpgpc.alignment import GaussianProcessWarp

NAMES = ('online_dynamic', 'offline_dynamic', 'offline_static', 'offline_multi',
         'warp_scalar', 'warp_batch')


def run(name):
    if name.startswith('warp_'):
        x = torch.linspace(0., 4., 64)[:, None]
        template = torch.exp(-((x - 1.8) / .65) ** 2)
        targets = torch.stack([torch.exp(-((x - (2.05 + .015*i)) / .65)**2) for i in range(16)])
        warper = GaussianProcessWarp(x, recursive=False)
        noise = torch.full((64,), .04)
        if name == 'warp_scalar':
            result = warper.compute_warp(x, targets[0], template, noise=noise, train_iter=250)
        else:
            result = warper.compute_warp_batch(x, targets, template, noise=noise, train_iter=250)
        return (result, warper)
    n, points = (8, 16) if name == 'offline_multi' else (12, 24)
    x = np.linspace(0., 4., points)[:, None]
    a = np.exp(-((x[:, 0] - 1.8)/.65)**2)
    b = -.8*np.exp(-((x[:, 0] - 2.2)/.7)**2)
    ys = np.stack([(a if i % 4 < 2 else b) + .002*(i % 3 - 1) for i in range(n)])
    ys = ys[:, :, None]
    outputs = 2 if name == 'offline_multi' else 1
    if outputs == 2:
        ys = np.concatenate([ys, .7*ys[:, ::-1] + .02], axis=2)
    model = build_state(
        x, n_outputs=outputs, model_type='static' if name == 'offline_static' else 'dynamic',
        ini_lengthscale=1.5, bound_lengthscale=(.5, 3.), ini_gamma=.03, ini_sigma=.04,
        ini_outputscale=1., bound_sigma=(.005, .08), bound_gamma=(.001, .1),
        x_basis_warp=x, noise_warp=.01, bound_noise_warp=(.001, .1), verbose=False,
        annealing=False, bayesian_params=True, cuda=False, n_explore_steps=2,
        max_models=4, free_deg_MNIV=5, reestimate_initial_params=False, use_snr=False)
    if name == 'online_dynamic':
        for y in ys:
            model.online.update(x, y, with_warp=False)
    else:
        model.offline.fit(np.repeat(x[None], n, axis=0), ys, it_limit=3, warp=False)
    return model


def snapshot(result):
    if isinstance(result, tuple):
        values, warper = result
        return encode({'result': values, 'controls': warper._u_ctrl_prev,
                       'prior_L': warper.warp_gp._cache_L,
                       'prior_logdet': warper.warp_gp._cache_logdet})
    return encode(model_state(result))
