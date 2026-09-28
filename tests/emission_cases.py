"""Tiny real GP workloads and pre-extraction reference capture; no patched math."""

import argparse
import base64
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import random
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if not os.environ.get("HDPGPC_TEST_INSTALLED"):
    sys.path.insert(0, str(ROOT / "hdpgpc"))
os.environ.setdefault("MPLBACKEND", "Agg")
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import torch
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from threadpoolctl import threadpool_limits
from hdpgpc.emissions import GaussianProcessEmission

_threads = threadpool_limits(limits=1)
MODES = ("exact", "projected", "reduced", "static")


def reset():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def make_emission(mode="exact", cls=GaussianProcessEmission):
    x = torch.arange(5, dtype=torch.float64)[:, None]
    basis = x[::2].clone() if mode in ("projected", "reduced") else x
    y = torch.tensor([0., .3, 1., .3, 0.])[:, None]
    kernel = ConstantKernel(1., (.5, 2.)) * RBF(1.5, (.5, 3.)) + WhiteKernel(.04, (.005, .08))
    model = cls(kernel, basis, annealing=False, bayesian=True, cuda=False,
                inducing_points=(mode == "reduced"), verbose=False)
    params = model.GPR_static(.04) if mode == "static" else model.GPR_dynamic(.03, .04)
    model.initial_conditions(ini_A=params[0], ini_Gamma=params[1], ini_C=params[2], ini_Sigma=params[3])
    return model, x, y


def encode(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.tolist()}
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def snapshot(model, x, y, optimizer):
    query = torch.linspace(0., 4., 7)[:, None]
    fields = ("N", "indexes", "fitted", "x_basis", "f_star", "f_star_sm", "cov_f", "cov_f_sm",
              "A", "Gamma", "C", "Sigma", "K", "ini_cov_def", "Sigma_def", "ini_kernel_theta")
    result = {name: getattr(model, name) for name in fields}
    result.update(kernel_theta=model.gp.kernel.theta, K_X_X=model.gp.K_X_X, K_inv=model.gp.K_inv,
                  alpha_ini=model.gp.alpha_ini, gamma_ini=model.gp.gamma_ini,
                  same_grid=model.observe_last(model.x_basis), query=model.observe_last(query),
                  latent=model.resample_latent_mean(query),
                  score=model.log_sq_error(x, y, i=-1), optimizer=optimizer,
                  internal=vars(model.internal_params), observation=vars(model.observation_params))
    return encode(result)


def run_case(mode, cls=GaussianProcessEmission):
    """Observe real Adam configuration/steps without replacing any optimizer work."""
    reset()
    model, x, y = make_emission(mode, cls)
    recorded = []
    original_adam = torch.optim.Adam

    def adam(*args, **kwargs):
        optimizer = original_adam(*args, **kwargs)
        record = {"steps": 0, "groups": [
            {key: value for key, value in group.items() if key != "params"}
            for group in optimizer.param_groups]}
        recorded.append(record)
        step = optimizer.step

        def observed_step(*args, **kwargs):
            record["steps"] += 1
            return step(*args, **kwargs)

        optimizer.step = observed_step
        return optimizer

    with redirect_stdout(io.StringIO()), patch.object(torch.optim, "Adam", adam):
        for i, observation in enumerate((y, y * .9)):
            model.include_weighted_sample(i, x, x, observation, 1.)
            if mode != "static":
                model.backwards_pair(1.)
            model.bayesian_new_params(1., model_type="static" if mode == "static" else "dynamic")
        if mode != "static":
            model.backwards()
    return model, snapshot(model, x, y, recorded)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture from the current implementation into a NEW file only.")
    parser.add_argument("--record-new", type=Path, required=True)
    args = parser.parse_args()
    if args.record_new.exists():
        parser.error("Refusing to overwrite an existing reference")
    cases = {}
    legacy = None
    for mode in MODES:
        model, cases[mode] = run_case(mode)
        if mode == "exact":
            legacy = base64.b64encode(pickle.dumps(model, protocol=4)).decode("ascii")
    source = ROOT / "hdpgpc/hdpgpc/emissions/_gaussian_process.py"
    payload = {"provenance": {"source": str(source.relative_to(ROOT)),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "torch": torch.__version__, "numpy": np.__version__, "seed": 42},
               "cases": cases, "legacy_pickle_b64": legacy}
    with args.record_new.open("x", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, allow_nan=False)
        output.write("\n")
    print("Recorded", list(cases), "to", args.record_new)
