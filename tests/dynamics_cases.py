"""Small LDS reference workloads, recorded before moving production methods."""

import argparse
import hashlib
import json
from pathlib import Path

from emission_cases import encode, make_emission, reset, torch, ROOT


def setup(mode):
    reset()
    model, x, y = make_emission("static" if mode == "static" else "exact")
    n = len(x)
    eye = torch.eye(n)
    A = eye if mode == "static" else .85 * eye + .02 * torch.ones(n, n)
    Gamma = torch.zeros(n, n) if mode == "static" else .025 * eye
    C = eye if mode == "static" else torch.diag(torch.linspace(.8, 1.1, n))
    Sigma = torch.diag(torch.linspace(.03, .07, n))
    means = [y * scale + offset for scale, offset in ((.4, .02), (.7, -.03), (.8, .01))]
    covars = [scale * eye + .01 * torch.ones(n, n) for scale in (.3, .2, .15)]
    return model, x, y, (A, Gamma, C, Sigma), means, covars


def state(model):
    names = ("N", "indexes", "f_star", "f_star_sm", "cov_f", "cov_f_sm", "A", "Gamma", "C", "Sigma",
             "A_def", "Gamma_def", "C_def", "Sigma_def", "var", "y_var")
    result = {name: getattr(model, name) for name in names}
    for name in ("internal_params", "observation_params"):
        value = getattr(model, name)
        result[name] = {"type": type(value).__name__, "state": vars(value)}
    return encode(result)


def core(mode, task, component=None):
    model, x, y, params, means, covars = setup(mode)
    A, Gamma, C, Sigma = params
    backend = model.gp
    if task == "initial":
        if component is None:
            default = model.GPR_static() if mode == "static" else model.GPR_dynamic()
            explicit = model.GPR_static(.08) if mode == "static" else model.GPR_dynamic(.015, .08)
        else:
            default = component.initial_parameters(x)
            kwargs = {} if mode == "static" else {"gamma": .015}
            explicit = component.initial_parameters(x, sigma=.08, **kwargs)
        return encode({"default": default, "explicit": explicit})
    if task.startswith("filter"):
        mean, cov = means[0], covars[0]
        if task == "filter_prior":
            cov = model.K.clone()
        if task == "filter_warped":
            observation_x = torch.linspace(0., 4., 7)[:, None]
            warped_x = observation_x + .04 * torch.sin(observation_x)
            observation_y = torch.sin(observation_x)
        else:
            observation_x, warped_x, observation_y = x, x, y
        if task == "filter_gradients":
            mean = mean.detach().clone().requires_grad_()
            cov = cov.detach().clone().requires_grad_()
        args = (mean, cov, observation_y, A, Gamma, C, Sigma / .7)
        kwargs = {"x_train": observation_x, "x_warped": warped_x, "h": .7}
        result = backend.posterior(*args, **kwargs) if component is None else component.filter(backend, *args, **kwargs)
        output = {"result": result}
        if task == "filter_gradients":
            (result[0].sum() + result[1].sum()).backward()
            output.update(mean_gradient=mean.grad, covariance_gradient=cov.grad)
        return encode(output)
    if task == "smooth":
        # A single history entry exercises last-entry fallback in a three-state pass.
        args = ([A], [Gamma], means, covars)
        result = backend.backward(*args) if component is None else component.smooth(backend, *args)
    elif task == "smooth_pair":
        args = (A, Gamma, means, covars)
        result = backend.backward_notrange(*args) if component is None else component.smooth_pair(backend, *args)
    elif task == "estimate":
        args = (*params, [y, .9 * y, .8 * y], means, covars)
        result = backend.new_params_LDS(*args, model="static" if mode == "static" else "dynamic") if component is None else component.estimate_parameters(backend, *args)
        return encode(result)
    else:
        raise ValueError(task)
    return encode({"result": result, "same_means_list": result[0] is means,
                   "same_covars_list": result[1] is covars})


def trajectory(mode, task):
    model, x, y, _, _, _ = setup(mode)
    steps = [state(model)]
    for index in range(3):
        model.condition_observation(index, x, y * (1. - .1 * index))
        if mode != "static":
            model.backwards_pair(1.)
        model.bayesian_new_params(1., model_type="static" if mode == "static" else "dynamic")
        steps.append(state(model))
    if task == "reset":
        model.reinit_LDS(save_last=True, save_last_diag=True)
    elif task == "gates":
        model.bayesian_new_params(.999)
        model.backwards_pair(.999)
        model.backwards_pair(1., snr=.5)
        # Record the covariance/distribution behavior of the low-SNR Bayes branch.
        model.bayesian_new_params(1., snr=.5)
    elif task == "full_data":
        model.bayesian_new_params(1., full_data=True)
    elif task == "full_smoother":
        model.backwards()
    elif task == "reset_likelihood":
        model.reinit_LDS(return_likelihood=True)
    return {"steps": steps, "final": state(model)}


TASKS = ("initial", "filter_prior", "filter_regular", "filter_warped", "filter_gradients", "smooth", "smooth_pair", "estimate")
CASES = {f"{mode}_{task}": (lambda mode=mode, task=task: core(mode, task))
         for mode in ("static", "linear") for task in TASKS}
for mode in ("static", "linear"):
    for task in ("reset", "gates", "full_smoother", "full_data", "reset_likelihood"):
        CASES[f"trajectory_{mode}_{task}"] = lambda mode=mode, task=task: trajectory(mode, task)


def capture(name):
    # Exact existing failures are explicit records, not silently skipped cases.
    try:
        return {"value": CASES[name]()}
    except (RuntimeError, TypeError, UnboundLocalError, AttributeError) as error:
        return {"exception_type": type(error).__name__, "message": str(error)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-new", type=Path, required=True)
    args = parser.parse_args()
    if args.record_new.exists():
        parser.error("Refusing to overwrite a reference")
    payload = {"source_sha256": {name: hashlib.sha256((ROOT / 'hdpgpc/hdpgpc' / name).read_bytes()).hexdigest()
                                for name in ('emissions/_prior.py', 'emissions/_gaussian_process.py')},
               "torch": torch.__version__, "seed": 42, "cases": {}}
    for name in CASES:
        payload["cases"][name] = capture(name)
        result = payload["cases"][name]
        print(name, result.get("exception_type", "OK"), result.get("message", ""), flush=True)
    with args.record_new.open("x", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, allow_nan=False)
        output.write("\n")
