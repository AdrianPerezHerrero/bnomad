"""HDP sticks, transition distributions and the current variational objective.

HDPVariational provides calculations over a narrow live view of existing state.
It does not schedule GP, LDS, alignment or cluster lifecycle operations. Private
modules retain the adapted optimizer and assignment entropy calculations.
Importing this namespace alone does not initialize numerical libraries.
"""

__all__ = ("HDPVariational",)


def __getattr__(name):
    if name == "HDPVariational":
        from ._variational import HDPVariational

        return HDPVariational
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
