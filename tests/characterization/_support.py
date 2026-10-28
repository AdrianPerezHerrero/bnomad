"""Test-only imports, deterministic setup, snapshots, and golden comparison."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from functools import wraps
import importlib.metadata
import math
import os
from pathlib import Path
import random
import sys

WORKSPACE = Path(__file__).resolve().parents[2]
REFERENCE = WORKSPACE
SOURCE = REFERENCE / "hdpgpc" / "hdpgpc"
if not os.environ.get("HDPGPC_TEST_INSTALLED"):
    sys.path.insert(0, str(REFERENCE / "hdpgpc"))
os.environ.setdefault("MPLBACKEND", "Agg")
for _thread_setting in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_setting] = "1"

import numpy as np
import torch
import pyro
from hdpgpc.model._state import ModelState
from tests.model_recipe import build_state
import inspect
SOURCE = Path(inspect.getfile(ModelState)).parents[1]
from hdpgpc.emissions import GaussianProcessEmission
from hdpgpc.alignment import GaussianProcessWarp
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from threadpoolctl import threadpool_limits

SEED = 42
RTOL = 1e-7
ATOL = 1e-9
_thread_limit = threadpool_limits(limits=1)


def reset_runtime():
    """Fix RNG and execution settings; never patch production routines."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    pyro.set_rng_seed(SEED)
    pyro.clear_param_store()
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def encode(value):
    """Retain shape/dtype and non-finite values in portable JSON snapshots."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape),
                "values": encode(value.tolist())}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("+Infinity" if value > 0 else "-Infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported snapshot type: {type(value)}")


def distribution_state(dist):
    return {k: getattr(dist, k) for k in ("m_mean", "m_r_cov", "scale", "n0", "C_fixed")
            if hasattr(dist, k)}


def gp_state(gp):
    fields = ("N", "indexes", "fitted", "x_basis", "x_train", "y_train", "f_star",
              "f_star_sm", "cov_f", "cov_f_sm", "A", "Gamma", "C", "Sigma",
              "var", "y_var", "ini_cov_def", "A_def", "Gamma_def", "C_def", "Sigma_def",
              "annealing", "bayesian", "estimation_limit", "inducing_points")
    state = {k: getattr(gp, k) for k in fields}
    state["kernel_theta"] = gp.gp.kernel.theta
    state["kernel_matrix"] = gp.gp.K_X_X
    state["internal_params"] = distribution_state(gp.internal_params)
    state["observation_params"] = distribution_state(gp.observation_params)
    return state


def model_state(model):
    fields = ("M", "T", "n_outputs", "actual_state", "rho", "omega", "transTheta",
              "startTheta", "trans_A", "q", "resp_assigned", "snr_norm", "train_elbo",
              "f_ind_old", "y_train", "x_train", "y_w", "x_w", "liks", "q_last",
              "q_lat_last", "resp_last", "respPair_last", "snr_last", "elbo_last",
              "startStateCount_last", "transStateCount_last")
    state = {k: getattr(model, k) for k in fields if hasattr(model, k)}
    state["transition_probabilities"] = model.compute_Pi()
    state["occupied_clusters"] = [sum(bool(g.indexes) for g in output) for output in model.gpmodels]
    state["gpmodels"] = [[gp_state(g) for g in output] for output in model.gpmodels]
    state["warp_cache_keys"] = sorted(getattr(model, "_warp_cache_full", {}).keys())
    return state


@contextmanager
def observe_inference(model):
    """Record real return values without replacing or recomputing numerical work.

    Instance wrappers call the original bound method exactly once and return its
    original result. Snapshots are copied to JSON values before later mutations.
    """
    trace = []
    names = ("variational_local_terms", "compute_q_elbo", "elbo_Linears", "full_LDS_elbo")

    def observer(name, original):
        @wraps(original)
        def call(*args, **kwargs):
            result = original(*args, **kwargs)
            if name == "variational_local_terms":
                # These fixtures use classify=False, whose four outputs are fixed.
                resp, logresp, pair, logpair = result
                payload = {"responsibilities": resp, "log_responsibilities": logresp,
                           "pair_responsibilities": pair, "log_pair_responsibilities": logpair}
            else:
                payload = {"result": result}
                if name == "compute_q_elbo":
                    payload.update(emission_input=args[2], latent_input=args[3],
                                   responsibilities=args[0], pair_responsibilities=args[1],
                                   post=kwargs.get("post", False),
                                   one_sample=kwargs.get("one_sample", False))
            trace.append(encode({"method": name, "M": model.M, "T": model.T, **payload}))
            return result
        return call

    for name in names:
        setattr(model, name, observer(name, getattr(model, name)))
    try:
        yield trace
    finally:
        for name in names:
            delattr(model, name)


def make_model(outputs=1, dynamics="dynamic", **kwargs):
    x = np.arange(5, dtype=np.float64)[:, None]
    options = dict(n_outputs=outputs, model_type=dynamics, ini_lengthscale=1.5,
                   bound_lengthscale=(0.5, 3.0), ini_gamma=0.03, ini_sigma=0.04,
                   ini_outputscale=1.0, bound_sigma=(0.005, 0.08), bound_gamma=(0.001, 0.1),
                   x_basis_warp=x, noise_warp=0.01, bound_noise_warp=(0.001, 0.1),
                   verbose=False, annealing=False, bayesian_params=True, cuda=False,
                   n_explore_steps=2, max_models=4, free_deg_MNIV=5,
                   reestimate_initial_params=False, use_snr=False)
    options.update(kwargs)
    return x, build_state(x, **options)


def waveform_data(outputs=1):
    """Six five-point waveforms; no disk data, no random fixture generation."""
    a = np.array([0.0, 0.4, 1.0, 0.4, 0.0])
    b = np.array([0.0, -0.6, -1.2, -0.6, 0.0])
    waves = np.stack([a, a + 0.01, b, b - 0.01, a - 0.015, b + 0.015])
    if outputs == 1:
        return waves[:, :, None]
    return np.stack([waves, 0.7 * waves[:, ::-1] + 0.02], axis=-1)


def source_hashes():
    return {p.relative_to(SOURCE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(SOURCE.rglob("*.py"))}


def environment():
    packages = ("numpy", "scipy", "torch", "gpytorch", "pyro-ppl", "scikit-learn",
                "torchmetrics", "linear-operator", "pandas")
    return {"python": sys.version, "platform": sys.platform,
            "packages": {p: importlib.metadata.version(p) for p in packages},
            "seed": SEED, "dtype": str(torch.get_default_dtype()), "device": "cpu",
            "torch_num_threads": torch.get_num_threads(), "source_sha256": source_hashes()}


def assert_snapshot(test, actual, expected, path="snapshot", exact=False):
    """Exact structure/discrete values; explicit tolerance only for finite floats."""
    test.assertIs(type(actual), type(expected), f"{path}: type")
    if isinstance(expected, dict):
        test.assertEqual(set(actual), set(expected), f"{path}: keys")
        for key in expected:
            discrete = key in ("resp_last", "respPair_last", "responsibilities", "pair_responsibilities",
                               "hard_tie", "hard_pair_tie")
            assert_snapshot(test, actual[key], expected[key], f"{path}.{key}", exact or discrete)
    elif isinstance(expected, list):
        test.assertEqual(len(actual), len(expected), f"{path}: length")
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_snapshot(test, a, e, f"{path}[{i}]", exact)
    elif isinstance(expected, float) and not exact:
        test.assertTrue(math.isclose(actual, expected, rel_tol=RTOL, abs_tol=ATOL),
                        f"{path}: {actual!r} != {expected!r} (rtol={RTOL}, atol={ATOL})")
    else:
        test.assertEqual(actual, expected, path)
