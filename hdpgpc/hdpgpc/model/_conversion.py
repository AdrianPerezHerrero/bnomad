"""State representation, device conversion and private checkpoint writing.

Functions operate on the explicitly supplied mutable inference state.
Numerical expressions and mutation order are preserved from the executable specification.
"""

import numpy as np
import torch
import pandas as pd
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from scipy.special._ufuncs import psi as digamma, gammaln
from torchmetrics.audio import SignalNoiseRatio
from tqdm import trange
import pickle as plk
from ..emissions import _gaussian_process as gp
from ..alignment import _components as _alignment
from ..hdp._variational import HDPVariational
from ..hdp._optimizer import find_optimum_multiple_tries, kvec
from ..numerics._normalization import max_shift_log_scores
from ..hdp._entropy import state_entropy_terms, pair_entropy_terms


def _convert_observation_owners(self, target):
    for output, owner in enumerate(getattr(self, 'observation_policies', ())):
        owner.convert(target)
        for emission in self.gpmodels[output]:
            emission.C_def = owner.prior.get_mean()
            emission.Sigma_def = owner.prior.get_scale()
            emission.observation_params = owner.posterior


def cond_to_torch(self, x):
    if type(x) is list and type(x[0]) is np.ndarray:
        x = torch.from_numpy(np.array(x))
        x.requires_grad = False
    elif type(x) is not torch.Tensor:
        x = torch.from_numpy(np.array(x))
        x.requires_grad = False
    else:
        x.requires_grad = False
    return x


def cond_to_numpy(self, x):
    if type(x) is torch.Tensor:
        x = x.detach().numpy()
    return x


def cond_cuda(self, x):
    if self.cuda:
        if type(x) is list or type(x) is np.ndarray:
            for i in x:
                if not i.is_cuda and torch.cuda.is_available():
                    i = i.cuda()
                    i.requires_grad = False
        elif not x.is_cuda and torch.cuda.is_available():
            x = x.cuda()
            x.requires_grad = False
    return x


def cond_cpu(self, x):
    if type(x) is torch.Tensor:
        x = x.cpu()
    return x


def cond_detach(self, x):
    if type(x) is torch.Tensor:
        x = x.detach()
    return x


def full_model_to_numpy(self):
    def recursive_numpy(x):
        if type(x) is torch.Tensor:
            x = self.cond_to_numpy(x)
        elif type(x) is list:
            if len(x) > 0:
                if type(x[0]) is torch.Tensor:
                    for j, i in enumerate(x):
                        x[j] = self.cond_to_numpy(i)
                elif len(x[0]) > 0:
                    if type(x[0][0]) is torch.Tensor:
                        for i in x:
                            for j, t in enumerate(i):
                                i[j] = self.cond_to_numpy(t)
        else:
            x = x
        return x

    self.x_basis = recursive_numpy(self.x_basis)
    self.y = recursive_numpy(self.y)
    self.y_w = recursive_numpy(self.y_w)
    self.x_w = recursive_numpy(self.x_w)
    self.liks = recursive_numpy(self.liks)
    self.rho = recursive_numpy(self.rho)
    self.omega = recursive_numpy(self.omega)
    self.theta = recursive_numpy(self.theta)
    self.transTheta = recursive_numpy(self.transTheta)
    self.startTheta = recursive_numpy(self.startTheta)
    for ld in range(self.n_outputs):
        for gp in self.gpmodels[ld]:
            gp.model_to_numpy()
        for w_gp in self.wp_sys[ld]:
            w_gp.warp_gp.model_to_numpy()
    _convert_observation_owners(self, 'numpy')


def full_model_to_torch(self):
    def recursive_torch(x):
        if type(x) is np.ndarray:
            x = self.cond_to_torch(x)
        elif type(x) is list:
            if len(x) > 0:
                if type(x[0]) is np.ndarray or type(x[0]) is np.float64:
                    for j, i in enumerate(x):
                        x[j] = self.cond_to_torch(i)
        else:
            x = x
        return x

    self.x_basis = recursive_torch(self.x_basis)
    self.y = recursive_torch(self.y)
    self.y_w = recursive_torch(self.y_w)
    self.x_w = recursive_torch(self.x_w)
    self.liks = recursive_torch(self.liks)
    self.rho = recursive_torch(self.rho)
    self.omega = recursive_torch(self.omega)
    self.theta = recursive_torch(self.theta)
    self.transTheta = recursive_torch(self.transTheta)
    self.startTheta = recursive_torch(self.startTheta)
    for ld in range(self.n_outputs):
        for gp in self.gpmodels[ld]:
            gp.model_to_torch()
        for w_gp in self.wp_sys[ld]:
            w_gp.warp_gp.model_to_torch()
    _convert_observation_owners(self, 'torch')


def full_model_to_cuda(self):
    def recursive_cuda(x):
        if type(x) is torch.Tensor:
            x = x.cuda()
            x.requires_grad = False
        elif type(x) is list:
            if len(x) > 0:
                if type(x[0]) is torch.Tensor or type(x[0]) is torch.float64:
                    for j, i in enumerate(x):
                        x[j] = i.cuda()
                        x[j].requires_grad = False
        else:
            x = x
        return x

    self.x_basis = recursive_cuda(self.x_basis)
    self.y = recursive_cuda(self.y)
    self.y_w = recursive_cuda(self.y_w)
    self.x_w = recursive_cuda(self.x_w)
    self.liks = recursive_cuda(self.liks)
    self.rho = recursive_cuda(self.rho)
    self.omega = recursive_cuda(self.omega)
    self.theta = recursive_cuda(self.theta)
    self.transTheta = recursive_cuda(self.transTheta)
    self.startTheta = recursive_cuda(self.startTheta)
    for ld in range(self.n_outputs):
        for gp in self.gpmodels[ld]:
            gp.model_to_cuda()
        for w_gp in self.wp_sys[ld]:
            w_gp.warp_gp.model_to_cuda()
    _convert_observation_owners(self, 'cuda')


def full_model_to_cpu(self):
    if self.cuda:
        def recursive_cpu(x):
            if type(x) is torch.Tensor:
                x = x.cpu()
            elif type(x) is list:
                if len(x) > 0:
                    if type(x[0]) is torch.Tensor or type(x[0]) is torch.float64:
                        for j, i in enumerate(x):
                            x[j] = i.cpu()
                    elif type(x[0]) is list:
                        if type(x[0][0]) is torch.Tensor:
                            for x_ in x:
                                for j, i in enumerate(x_):
                                    x_[j] = i.cpu()
            else:
                x = x
            return x

        self.x_basis = recursive_cpu(self.x_basis)
        self.y = recursive_cpu(self.y)
        self.y_w = recursive_cpu(self.y_w)
        self.x_w = recursive_cpu(self.x_w)
        self.liks = recursive_cpu(self.liks)
        self.rho = recursive_cpu(self.rho)
        self.omega = recursive_cpu(self.omega)
        self.theta = recursive_cpu(self.theta)
        self.transTheta = recursive_cpu(self.transTheta)
        self.startTheta = recursive_cpu(self.startTheta)
        for ld in range(self.n_outputs):
            for gp in self.gpmodels[ld]:
                gp.model_to_cpu()
            for w_gp in self.wp_sys[ld]:
                w_gp.warp_gp.model_to_cpu()
        _convert_observation_owners(self, 'cpu')


def save_swgp(self, st):
    self.keep_last_all()
    self.full_model_to_cpu()
    with open(st, 'wb') as inp:
        plk.dump(self, inp)

