"""Minimal stateless policies; static retains the same real filtering equations."""

import torch
from ._kalman import filter_posterior, smooth_sequence, smooth_pair
from ._learning import estimate_lds_parameters


class LinearDynamics:
    """Current LDS initial matrices and recurrences, with explicit caller state."""

    @staticmethod
    def initial_parameters(x_basis, sigma=None, *, gamma=None):
        """ Define dynamic conditions using initial Sigma as a double.
        """
        shape = len(x_basis)
        ini_A = torch.eye(shape)
        if gamma is None:
            ini_Gamma = torch.mul(0.01, torch.eye(shape))
        else:
            ini_Gamma = torch.mul(gamma, torch.eye(shape))
        ini_C = torch.eye(shape)
        if sigma is None:
            ini_Sigma = torch.mul(0.5 ** 2, torch.eye(shape))
        else:
            ini_Sigma = torch.mul(sigma, torch.eye(shape))
        return ini_A, ini_Gamma, ini_C, ini_Sigma

    filter = staticmethod(filter_posterior)
    smooth = staticmethod(smooth_sequence)
    smooth_pair = staticmethod(smooth_pair)

    @staticmethod
    def estimate_parameters(backend, A_prior, Gamma_prior, C_prior, Sigma_prior,
                            y_samples, means, covars):
        return estimate_lds_parameters(backend, A_prior, Gamma_prior, C_prior, Sigma_prior,
                                       y_samples, means, covars, model='dynamic')


class StaticDynamics(LinearDynamics):
    """Original static formulation: identity A/C, zero Gamma, learned Sigma.

    Supplied matrices are used as-is by filtering and smoothing. No frozen-state
    shortcut, automatic covariance zeroing or replacement static regression.
    """

    @staticmethod
    def initial_parameters(x_basis, sigma=None):
        """ Define static conditions using initial Sigma as a double.
        """
        shape = len(x_basis)
        ini_A = torch.eye(shape)
        ini_Gamma = torch.zeros((shape, shape))
        ini_C = torch.eye(shape)
        if sigma is None:
            ini_Sigma = torch.mul(0.5 ** 2, torch.eye(shape))
        else:
            ini_Sigma = torch.mul(sigma, torch.eye(shape))
        return ini_A, ini_Gamma, ini_C, ini_Sigma

    @staticmethod
    def estimate_parameters(backend, A_prior, Gamma_prior, C_prior, Sigma_prior,
                            y_samples, means, covars):
        return estimate_lds_parameters(backend, A_prior, Gamma_prior, C_prior, Sigma_prior,
                                       y_samples, means, covars, model='static')


# No instance owns mutable model state or a reference back to an emission.
_LINEAR = LinearDynamics()
_STATIC = StaticDynamics()
