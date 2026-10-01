"""Small real inference runs shared by recording and regression tests."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import time
import traceback

from ._support import (GaussianProcessEmission, GaussianProcessWarp, ConstantKernel, RBF, WhiteKernel,
                       encode, gp_state, make_model, model_state, observe_inference, np, reset_runtime,
                       torch, waveform_data)


def online(warp=False, fast=False, dynamics="dynamic", outputs=1):
    x, model = make_model(outputs=outputs, dynamics=dynamics)
    states = []
    with observe_inference(model) as trace:
        for y in waveform_data(outputs)[:4]:
            method = model.online.update_fast if fast else model.online.update
            method(x, y, warp=warp)
            states.append(encode(model_state(model)))
    return {"steps": states, "inference_trace": trace}


def offline(outputs=1, warp=False, dynamics="dynamic", use_snr=False):
    x, model = make_model(outputs=outputs, dynamics=dynamics, use_snr=use_snr)
    y = waveform_data(outputs)
    xs = np.repeat(x[None], len(y), axis=0)
    with observe_inference(model) as trace:
        model.offline.fit(xs, y, it_limit=3, warp=warp)
    state = model_state(model)
    classification = model.classify_batch(xs, y, learning=False)
    return {"final": state, "classification": classification, "inference_trace": trace}


def emission(dynamics="dynamic"):
    x = torch.arange(5, dtype=torch.float64)[:, None]
    kernel = ConstantKernel(1.0, (0.5, 2.0)) * RBF(1.5, (0.5, 3.0)) + WhiteKernel(0.04, (0.005, 0.08))
    gp = GaussianProcessEmission(kernel, x, annealing=False, bayesian=True, cuda=False, verbose=False)
    params = gp.GPR_static(0.04) if dynamics == "static" else gp.GPR_dynamic(0.03, 0.04)
    gp.initial_conditions(ini_A=params[0], ini_Gamma=params[1], ini_C=params[2], ini_Sigma=params[3])
    states = []
    for i, y in enumerate(waveform_data()[:2]):
        gp.include_weighted_sample(i, x, x, torch.from_numpy(y), 1.0)
        if dynamics == "dynamic":
            gp.backwards_pair(1.0)
        gp.bayesian_new_params(1.0, model_type=dynamics)
        states.append(encode(gp_state(gp)))
    if dynamics == "dynamic":
        gp.backwards()
    query = torch.linspace(0, 4, 7)[:, None]
    return {"steps": states, "final": gp_state(gp), "observe_same": gp.observe_last(x),
            "observe_interpolated": gp.observe_last(query),
            "latent_interpolated": gp.resample_latent_mean(query),
            "emission_scores": [gp.log_sq_error(x, torch.from_numpy(y), i=-1) for y in waveform_data()[:3]],
            "latent_scores": gp.compute_q_lat_all(x[None].repeat(3, 1, 1))}


def warper(batch=False):
    x = torch.arange(7, dtype=torch.float64)[:, None]
    template = torch.tensor([0.0, 0.2, 0.8, 1.0, 0.4, 0.1, 0.0])[:, None]
    target = torch.tensor([0.0, 0.0, 0.2, 0.8, 1.0, 0.4, 0.0])[:, None]
    w = GaussianProcessWarp(x, noise_warp=0.01, bound_noise_warp=(0.001, 0.1), recursive=True)
    states = []
    for _ in range(2):
        if batch:
            args = (x, torch.stack([target, target * 0.9]), template)
            result = w.compute_warp_batch(*args, theta=1.5, noise=torch.full((7,), 0.04), train_iter=12)
        else:
            result = w.compute_warp(x, target, template, theta=1.5,
                                    noise=torch.full((7,), 0.04), train_iter=12)
        states.append({"result": result, "warm_start": w._u_ctrl_prev.clone(),
                       "prior_logdet": w.warp_gp._cache_logdet})
    return {"calls": states}


def hdp_messages():
    _, model = make_model(M=2)
    q = torch.tensor([[[-1.0], [-3.0]], [[-2.0], [-0.2]], [[-0.4], [-2.0]]])
    model.snr_norm = torch.ones(3, 1)
    resp, logresp, pair, logpair = model.variational_local_terms(q)
    rho, omega = model.find_optimum_rhoOmega()
    return {"responsibilities": resp, "pair_responsibilities": pair,
            "log_responsibilities": logresp, "log_pair_responsibilities": logpair,
            "rho_before": model.rho, "omega_before": model.omega,
            "rho_after": rho, "omega_after": omega,
            "beta": model.rho2beta(model.rho), "transTheta": model.transTheta,
            "startTheta": model.startTheta, "probabilities": model.compute_Pi(),
            "hard_tie": model._safe_exp(torch.zeros(2, 2)),
            "hard_pair_tie": model._safe_exp(torch.zeros(2, 2, 2))}


def known_failure(kind):
    """Characterize narrow unsupported paths; fail if their behavior changes."""
    x, model = make_model(outputs=2 if kind == "online_multi" else 1)
    expected_type = IndexError if kind == "online_multi" else RuntimeError
    for index, y in enumerate(waveform_data(model.n_outputs)[:2]):
        try:
            model.online.update(x, y, warp=(kind == "online_warp"))
        except expected_type as error:
            return {"failed_observation": index, "exception_type": type(error).__name__,
                    "message": str(error), "partial_state": model_state(model)}
    raise AssertionError(f"{kind} no longer fails: review and replace the failure characterization")


CASES = {
    "online_dynamic": lambda: online(),
    "online_fast": lambda: online(fast=True),
    "online_static": lambda: online(dynamics="static"),
    "offline_dynamic": lambda: offline(),
    "offline_multi": lambda: offline(outputs=2),
    "offline_multi_snr": lambda: offline(outputs=2, use_snr=True),
    "offline_warp": lambda: offline(warp=True),
    "offline_static": lambda: offline(dynamics="static"),
    "emission_dynamic": lambda: emission(),
    "emission_static": lambda: emission(dynamics="static"),
    "warper_scalar": lambda: warper(),
    "warper_batch": lambda: warper(batch=True),
    "hdp_messages": hdp_messages,
    "known_failure_online_warp": lambda: known_failure("online_warp"),
    "known_failure_online_multi": lambda: known_failure("online_multi"),
}

SUPPORTED_CASES = tuple(name for name in CASES if not name.startswith("known_failure_"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe actual production paths without updating baselines.")
    parser.add_argument("cases", nargs="*", choices=list(CASES))
    args = parser.parse_args()
    for name in args.cases or CASES:
        reset_runtime()
        start = time.perf_counter()
        try:
            with redirect_stdout(io.StringIO()):
                result = encode(CASES[name]())
            final = result.get("final", result.get("steps", [{}])[-1])
            print("PROBE", name, "OK", round(time.perf_counter() - start, 2),
                  "M", final.get("M"), "occupied", final.get("occupied_clusters"), flush=True)
        except Exception:
            print("PROBE", name, "FAILED", flush=True)
            traceback.print_exc()
