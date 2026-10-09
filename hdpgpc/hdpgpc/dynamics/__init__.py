"""Static and LDS dynamics using the existing GP-conditioned recurrences.

Public components are stateless operations on existing matrices/histories.
Private modules preserve the legacy numerical and trajectory-update contracts.
Importing this namespace alone does not initialize numerical libraries.
"""

__all__ = (
    "DynamicsModel", "StaticDynamics", "LinearDynamics",
    "LDSNoisePriorEstimate", "estimate_lds_noise_priors",
    "PCAObservationEstimate", "estimate_pca_observation",
    "FixedObservationModel", "AffineGaussianObservation",
)


def __getattr__(name):
    if name == "DynamicsModel":
        from ._interface import DynamicsModel

        return DynamicsModel
    if name in ("StaticDynamics", "LinearDynamics"):
        from ._models import StaticDynamics, LinearDynamics

        return {"StaticDynamics": StaticDynamics, "LinearDynamics": LinearDynamics}[name]
    if name in ("LDSNoisePriorEstimate", "estimate_lds_noise_priors"):
        from ._calibration import LDSNoisePriorEstimate, estimate_lds_noise_priors

        return {
            "LDSNoisePriorEstimate": LDSNoisePriorEstimate,
            "estimate_lds_noise_priors": estimate_lds_noise_priors,
        }[name]
    if name in ("PCAObservationEstimate", "estimate_pca_observation"):
        from ._pca import PCAObservationEstimate, estimate_pca_observation

        return {
            "PCAObservationEstimate": PCAObservationEstimate,
            "estimate_pca_observation": estimate_pca_observation,
        }[name]
    if name in ("FixedObservationModel", "AffineGaussianObservation"):
        from ._fixed_observation import FixedObservationModel, AffineGaussianObservation

        return {"FixedObservationModel": FixedObservationModel,
                "AffineGaussianObservation": AffineGaussianObservation}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
