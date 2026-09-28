"""Structural contract for existing emission operations, without runtime policy."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EmissionModel(Protocol):
    """One cluster/output trajectory's core emission operations.

    Coordinates/basis and kernel belong to the concrete constructor. Initial
    matrices select static or LDS dynamics. Return values, score normalization,
    responsibility gates and state mutation are defined by the implementation;
    this contract does not impose a normalized Gaussian density or soft updates.
    The full legacy inference driver also accesses trajectory/LDS state and is
    not yet a generic consumer of this protocol alone.
    """

    def initial_conditions(self, ini_mean=None, ini_cov=None, ini_A=None,
                           ini_Gamma=None, ini_C=None, ini_Sigma=None) -> None:
        """Initialize trajectory priors and supplied dynamics matrices."""
        ...

    def fit_kernel_params(self, x_train, y, alpha_ini, gamma_ini,
                          valid=True) -> tuple[Any, Any]:
        """Fit/reset emission priors; return basis coordinates and covariance."""
        ...

    def observe_last(self, x_post) -> tuple[Any, Any]:
        """Return observation mean/covariance from the last smoothed state."""
        ...

    def log_sq_error(self, x_train, y, mean=None, cov=None, C=None, Sigma=None,
                     i=None, proj=False, first=False) -> Any:
        """Return the implementation's assignment score, not a density promise."""
        ...

    def include_weighted_sample(self, index, x_train, x_warped, y, h,
                                snr=None) -> Any:
        """Apply existing responsibility/SNR gates; return basis coordinates."""
        ...
