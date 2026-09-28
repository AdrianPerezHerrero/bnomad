"""Small active-warper reference cases; optional source override for paired timing."""

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if not os.environ.get("HDPGPC_TEST_INSTALLED"):
    sys.path.insert(0, os.environ.get("HDPGPC_ALIGNMENT_REFERENCE_SOURCE", str(ROOT / "hdpgpc")))
os.environ.setdefault("MPLBACKEND", "Agg")
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from hdpgpc.alignment import GaussianProcessWarp
from hdpgpc.alignment._amtgp import WarpPriorAMTGP
from hdpgpc.model._state import ModelState
from tests.model_recipe import build_state

_threads = threadpool_limits(limits=1)


def reset():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def encode(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.tolist()}
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def waveforms(T=9, B=3):
    x = torch.linspace(0., 4., T)[:, None]
    template = torch.exp(-((x - 1.8) / .65) ** 2)
    targets = torch.stack([torch.exp(-((x - (2.05 + .08 * i)) / .65) ** 2) for i in range(B)])
    return x, template, targets


def make_model(T=9, B=4, outputs=1, **options):
    x, _, targets = waveforms(T, B)
    x_np = x.numpy()
    kwargs = dict(M=2, n_outputs=outputs, model_type="dynamic", ini_lengthscale=1.5,
                  bound_lengthscale=(.5, 3.), ini_gamma=.03, ini_sigma=.04,
                  ini_outputscale=1., bound_sigma=(.005, .08), bound_gamma=(.001, .1),
                  x_basis_warp=x_np, noise_warp=.01, bound_noise_warp=(.001, .1),
                  verbose=False, annealing=False, bayesian_params=True, cuda=False,
                  n_explore_steps=2, max_models=4, reestimate_initial_params=False,
                  use_snr=False)
    kwargs.update(options)
    model = build_state(x_np, **kwargs)
    model.f_ind_old = torch.tensor([0, 1])
    xs = x[None].repeat(B, 1, 1)
    ys = targets.repeat(1, 1, outputs)
    resp = torch.zeros(B, 2)
    resp[:, 0] = 1.
    return model, xs, ys, resp


def warper_state(warper):
    return encode({"basis": warper.x_basis, "T": warper.T, "n_ctrl": warper.n_ctrl,
                   "controls": warper._u_ctrl_prev, "theta": warper.warp_gp.theta,
                   "prior_L": warper.warp_gp._cache_L,
                   "prior_logdet": warper.warp_gp._cache_logdet,
                   "prior_key": warper.warp_gp._cache_key})


def direct(batch=False, recursive=True, bayesian=True, irregular=False, changed_grid=False):
    reset()
    x, template, targets = waveforms()
    if irregular:
        x = x ** 1.3
    warper = GaussianProcessWarp(x, noise_warp=.01, bound_noise_warp=(.001, .1),
                            recursive=recursive, bayesian=bayesian, n_ctrl=4)
    calls = []
    for number in range(2):
        if number and changed_grid:
            x, template, targets = waveforms(11)
        noise = torch.linspace(.02, .07, len(x))
        if batch:
            result = warper.compute_warp_batch(x, targets, template, theta=(1.3, .8),
                                               noise=noise, weights=torch.tensor([1., .5, -.2]), train_iter=16)
        else:
            result = warper.compute_warp(x, targets[0], template, theta=(1.3, .8), noise=noise, train_iter=16)
        calls.append({"result": encode(result), "state": warper_state(warper)})
    return calls


def cached_model():
    reset()
    model, xs, ys, resp = make_model(outputs=2)
    model.warp = True
    outputs = []
    for values in (ys, ys + 10.):
        result = model.warp_batch_by_resp_amtgp_cached(xs, values, resp, train_iter=12, batch_size=2)
        outputs.append(encode(result))
    return {"calls": outputs, "keys": sorted(model._warp_cache_full),
            "warpers": [[warper_state(w) for w in group] for group in model.wp_sys]}


def candidates(forced=False):
    reset()
    model, xs, ys, _ = make_model()
    for index, gp in enumerate(model.gpmodels[0]):
        gp.condition_observation(index, xs[index], ys[index])
    with redirect_stdout(io.StringIO()):
        result = model.compute_warp_y(xs[-1], ys[-1], strategie="standard", force_model=0 if forced else None)
    return {"result": encode(result), "warpers": [warper_state(w) for w in model.wp_sys[0]]}


def prior_cache():
    reset()
    x, _, targets = waveforms()
    prior = WarpPriorAMTGP(.01, (.001, .1))
    first = prior.log_sq_error(x, targets[0])
    L = prior._cache_L
    changed = x.clone()
    changed[1:-1] += .1
    second = prior.log_sq_error(changed, targets[0])
    return encode({"first": first, "second": second, "same_L": prior._cache_L is L,
                   "key": prior._cache_key, "L": L})


CASES = {
    "scalar_recursive": lambda: direct(),
    "scalar_irregular": lambda: direct(irregular=True),
    "scalar_no_recursion_no_prior_penalty": lambda: direct(recursive=False, bayesian=False),
    "scalar_changed_grid": lambda: direct(changed_grid=True),
    "batch_recursive": lambda: direct(batch=True),
    "batch_irregular": lambda: direct(batch=True, irregular=True),
    "batch_no_recursion_no_prior_penalty": lambda: direct(batch=True, recursive=False, bayesian=False),
    "batch_changed_grid": lambda: direct(batch=True, changed_grid=True),
    "prior_cache": prior_cache,
    "model_cached": cached_model,
    "model_candidates": candidates,
    "model_forced_candidate": lambda: candidates(forced=True),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-new", type=Path, required=True)
    args = parser.parse_args()
    if args.record_new.exists():
        parser.error("Refusing to overwrite reference")
    payload = {"source_sha256": {name: hashlib.sha256((ROOT / "hdpgpc/hdpgpc" / name).read_bytes()).hexdigest()
                                for name in ("alignment/_amtgp.py", "model/_state.py")},
               "torch": torch.__version__, "seed": 42, "cases": {}}
    for name, case in CASES.items():
        payload["cases"][name] = case()
        print(name, "captured", flush=True)
    with args.record_new.open("x", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, allow_nan=False)
        output.write("\n")
