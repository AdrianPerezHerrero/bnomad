"""Tiny HDP-only fixtures captured before extraction, without fitting emissions."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if not os.environ.get("HDPGPC_TEST_INSTALLED"):
    sys.path.insert(0, os.environ.get("HDPGPC_HDP_REFERENCE_SOURCE", str(ROOT / "hdpgpc")))
os.environ.setdefault("MPLBACKEND", "Agg")
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import torch
from scipy.special._ufuncs import psi as digamma
from threadpoolctl import threadpool_limits
from hdpgpc.model._state import ModelState
from hdpgpc.hdp import _optimizer as optimizer

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
    if isinstance(value, bytes):
        return value.decode()
    return value


def owner(M=3, sticky=0.):
    """Only the original HDP fields/conversion context; no GP/warp objects."""
    reset()
    model = ModelState.__new__(ModelState)
    model.hmm_switch, model.verbose = True, False
    model.M, model.T, model.n_outputs = M, 4, 2
    model.cuda, model.device = False, torch.device("cpu")
    model.gamma, model.transAlpha, model.startAlpha, model.kappa = 1., 1., .1, sticky
    model.init_global_params(9, M)
    return model


def state(model):
    return encode({name: getattr(model, name) for name in
                   ("M", "gamma", "transAlpha", "startAlpha", "kappa", "rho", "omega", "transTheta", "startTheta")})


def initialization(M):
    model = owner(M)
    return {"state": state(model), "beta": encode(model.rho2beta(model.rho)),
            "pi": encode(model.compute_Pi())}


def transforms():
    model = owner()
    rho = np.array([1e-8, .35, 1.-1e-8])
    beta = model.rho2beta(rho)
    return encode({"beta": beta, "active": model.rho2beta(rho, 'K'),
                   "rho_full": model.beta2rho(beta, 3), "rho_active": model.beta2rho(beta[:-1], 3),
                   "near_exhausted": model.beta2rho(np.array([1., 1e-8, 0.]), 3)})


def theta(kind):
    model = owner(3, sticky=.7)
    counts = torch.arange(16.).reshape(4, 4) / 7.
    starts = torch.tensor([3., 2., 1., 99.])
    if kind == 'full':
        result = model._calcThetaFull(counts, starts, 4)
    elif kind == 'truncated':
        result = model._calcThetaFull(counts, starts, 3)
    else:
        model.transTheta = model.transTheta[:3, :3].clone()
        model.startTheta = model.startTheta[:3].clone()
        result = model._calcThetaPost(counts, starts, 4, model.rho)
    return encode({"result": result, "counts_after": counts, "starts_after": starts})


def reinitialization():
    model = owner(2)
    before = state(model)
    counts = torch.arange(16.).reshape(4, 4) / 5.
    starts = torch.tensor([2., 1., 0., 0.])
    temporary = model.temp_reinit_global_params(3, counts, starts)
    after_temp = state(model)
    model.reinit_global_params(3, counts, starts)
    return encode({"before": before, "temporary": temporary, "after_temp": after_temp, "committed": state(model)})


def expected_logs(component=None):
    """Independent Dirichlet expected-log formulas for every inference schedule."""
    model = owner()
    model.transTheta = torch.tensor([[.02, .3, .7, .04], [.8, .06, .5, .12], [1.1, .2, .4, .01], [.2, .2, .2, .4]])
    model.startTheta = torch.tensor([.03, .5, .15, .02])
    M, transTheta, startTheta = model.M, model.transTheta, model.startTheta
    if component is None:
        online_t = digamma(model.cond_cpu(transTheta[:M, :M])) - digamma(torch.sum(model.cond_cpu(transTheta[:M, :M+1]), axis=1))[:, None]
        online_s = digamma(model.cond_cpu(startTheta[:M])) - digamma(torch.sum(model.cond_cpu(startTheta[:M+1])))
        # All schedules now use E[log p] from the same Dirichlet posterior.
        # log(E[p]) and row-normalized digamma values are different quantities.
        batch_t, stored_t, classify_t = online_t, online_t, online_t
        batch_s, classify_s = online_s, online_s
    else:
        hdp = component(model)
        online_t = hdp.expected_log_transitions(M, transTheta)
        online_s = hdp.expected_log_start(M, startTheta)
        batch_t = hdp.batch_log_transitions(M, transTheta)
        batch_s = hdp.batch_log_start(M, startTheta)
        stored_t = hdp.stored_log_transitions(M, transTheta)
        classify_t = hdp.classification_log_transitions(M, transTheta)
        classify_s = hdp.classification_log_start(M, startTheta)
    return encode({"online_t": online_t, "online_s": online_s, "batch_t": batch_t, "batch_s": batch_s,
                   "stored_t": stored_t, "classify_t": classify_t, "classify_s": classify_s,
                   "pi": model.compute_Pi(), "message_transitions": model.compute_trans_A(3),
                   "padded_transitions": model.compute_trans_A(5), "padded_start": model.compute_trans_pi(4, online_s)})


def objectives(sticky=False):
    model = owner(3, sticky=.7 if sticky else 0.)
    rho, omega = model.rho.numpy(), model.omega.numpy()
    trans, start = model.transTheta.numpy(), model.startTheta.numpy()
    counts = np.arange(16.).reshape(4, 4) / 4.
    starts = np.array([2., 1., 0., 0.])
    result = model.calcELBO_LinearTerms(rho, omega, model.transAlpha, model.startAlpha,
                                      model.kappa, model.gamma, trans, start, starts, counts)
    resp = np.array([[1., 0., 0.], [.2, .3, .5], [0., 1., 0.]])
    pairs = np.arange(27., dtype=np.float64).reshape(3, 3, 3)/27.
    return encode({"linear": result, "mutated_counts": counts, "start_counts": starts,
                   "top": model.L_top(rho, omega, 1., .1, model.kappa, 1.),
                   "dir_vector": model.c_Dir(start), "dir_matrix": model.c_Dir(trans),
                   "dir_remainder": model.c_Dir(trans[:, :-1], trans[:, -1]),
                   "entropy": model.calcELBO_NonlinearTerms(resp, pairs)})


def growth_arrays():
    model = owner(2)
    resp = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [1., 0.]])
    pair = torch.arange(16.).reshape(4, 2, 2)
    q = torch.arange(16.).reshape(4, 2, 2) / 9.
    q_lat, snr = -q, q - 1.
    grown = model.new_group(resp, pair, q, q_lat, snr)
    removed = model.remove_last_group(*grown)
    return encode({"grown": grown, "removed": removed,
                   "new_M_unchanged": model.M,
                   "pruning_returns_views": [a.data_ptr() == b.data_ptr() for a, b in zip(removed, grown)]})


def refill():
    model = owner(3)
    resp = torch.tensor([[1., 0., 0.], [0., 0., 1.], [1., 0., 0.], [0., 0., 1.]])
    pair = torch.arange(36.).reshape(4, 3, 3)
    out = model.refill_resp(resp, pair)
    resp_only = model.refill_resp(torch.tensor([[0., 0., 1.], [0., 0., 1.]]))
    return encode({"result": out, "mutated_resp": resp, "mutated_pair": pair,
                   "same_objects": [out[0] is resp, out[1] is pair], "first_of_multiple_empty": resp_only})


def optimizer_calculations(kind):
    reset()
    rhoomega = np.array([.25, .4, .55, 2.1, 2.8, 3.3])
    c = optimizer.rhoomega2c(rhoomega)
    kwargs = dict(nDoc=4, gamma=.7, alpha=1.3, kappa=.4 if kind=='sticky' else 0.,
                  sumLogPi=np.array([-2., -3., -4., -5.]), startAlphaLogPi=np.array([-.1, -.2, -.3, -.4]))
    if kind == 'prior':
        kwargs['nDoc'] = 0
    elif kind == 'separate':
        kwargs.update(sumLogPi=None, sumLogPiActiveVec=np.array([-2.,-3.,-4.]), sumLogPiRemVec=np.array([-3.,-5.,-6.]))
    beta = np.append(rhoomega[:3],1.)
    beta[1:] *= np.cumprod(1.-rhoomega[:3])
    return encode({"c": c, "roundtrip": optimizer.c2rhoomega(c),
                   "constrained": optimizer.objFunc_constrained(rhoomega, **kwargs),
                   "unconstrained": optimizer.objFunc_unconstrained(c, **kwargs),
                   "approx_value": optimizer.objFunc_unconstrained(c, approx_grad=True, **kwargs),
                   "d_beta": optimizer.calc_dEbeta_drho(beta, rhoomega[:3], 3),
                   "psi": optimizer.calc_Psi(beta, rhoomega[:3], 3)})


def optimize(sticky=False):
    model = owner(3, sticky=.7 if sticky else 0.)
    result = model.find_optimum_rhoOmega()
    direct = optimizer.find_optimum_multiple_tries(sumLogPi=np.array([-2., -3., -4., -5.]),
                                                  nDoc=0, gamma=.7)
    return encode({"result": result, "info": model.OptimizerInfo, "prior_solution": direct})


CASES = {
    'initial_one': lambda: initialization(1),
    'initial_three': lambda: initialization(3),
    'initial_five': lambda: initialization(5),
    'stick_transforms': transforms,
    'theta_full': lambda: theta('full'),
    'theta_truncated': lambda: theta('truncated'),
    'theta_posterior': lambda: theta('post'),
    'reinitialize_and_grow': reinitialization,
    'expected_logs': expected_logs,
    'objectives': objectives,
    'objectives_sticky': lambda: objectives(True),
    'growth_and_prune_arrays': growth_arrays,
    'refill_empty': refill,
    'optimizer_regular': lambda: optimizer_calculations('regular'),
    'optimizer_sticky': lambda: optimizer_calculations('sticky'),
    'optimizer_separate': lambda: optimizer_calculations('separate'),
    'optimizer_prior': lambda: optimizer_calculations('prior'),
    'optimize_regular': optimize,
    'optimize_sticky': lambda: optimize(True),
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--record-new', type=Path, required=True)
    args = parser.parse_args()
    if args.record_new.exists():
        parser.error('Refusing to overwrite reference')
    payload = {'source_sha256': {name:hashlib.sha256((ROOT/'hdpgpc/hdpgpc'/name).read_bytes()).hexdigest()
                                for name in ('hdp/_variational.py','hdp/_optimizer.py')},
               'torch': torch.__version__, 'seed':42, 'cases':{}}
    for name, case in CASES.items():
        payload['cases'][name] = case()
        print(name, 'captured', flush=True)
    with args.record_new.open('x', encoding='utf-8') as out:
        json.dump(payload, out, indent=2)
        out.write('\n')
