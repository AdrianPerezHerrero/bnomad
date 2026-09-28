"""Structural contract for the existing static and linear state recurrences."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DynamicsModel(Protocol):
    """Stateless operations on existing tensors and mutable smoothing lists.

    Filtering needs the existing GP projection backend (kernel, coordinates,
    device/conversion helpers and pred_dist). Smoothers/ML use its execution
    flags. This contract adds no distribution, state container or jitter policy.
    Prediction is inside filter; GP observation-only lookahead is a separate
    emission operation with different current covariance semantics.
    """

    def initial_parameters(self, x_basis, sigma=None) -> tuple[Any, Any, Any, Any]:
        """Return A, Gamma, C, Sigma; LinearDynamics also accepts keyword gamma."""
        ...

    def filter(self, backend, mean_prior, cov_prior, y_train, A, Gamma, C, Sigma,
               warped_mean=None, x_train=None, x_warped=None, embedding=True,
               h=1.0) -> tuple[Any, Any]:
        """Perform the original GP-conditioned prediction/Kalman update."""
        ...

    def smooth(self, backend, A_prior, Gamma_prior, means, covars) -> tuple[Any, Any]:
        """Mutate supplied lists using matrix histories and the full inverse pass."""
        ...

    def smooth_pair(self, backend, A_prior, Gamma_prior, means, covars) -> tuple[Any, Any]:
        """Mutate supplied lists using fixed matrices and the solve-based pass."""
        ...

    def estimate_parameters(self, backend, A_prior, Gamma_prior, C_prior, Sigma_prior,
                            y_samples, means, covars) -> tuple[Any, Any, Any, Any]:
        """One original ML parameter step for this dynamics formulation."""
        ...
