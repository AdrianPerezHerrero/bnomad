"""Static and LDS dynamics using the existing GP-conditioned recurrences.

Public components are stateless operations on existing matrices/histories.
Private modules preserve the legacy numerical and trajectory-update contracts.
Importing this namespace alone does not initialize numerical libraries.
"""

__all__ = ("DynamicsModel", "StaticDynamics", "LinearDynamics")


def __getattr__(name):
    if name == "DynamicsModel":
        from ._interface import DynamicsModel

        return DynamicsModel
    if name in ("StaticDynamics", "LinearDynamics"):
        from ._models import StaticDynamics, LinearDynamics

        return {"StaticDynamics": StaticDynamics, "LinearDynamics": LinearDynamics}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
