"""Intended internal boundary for explicitly specified numerical operations.

The private _normalization module contains the legacy maximum-shift operation.
Existing solves, factorization policies, dtype conversions, and jitter constants
remain at their call sites. No generic stabilization wrapper is introduced:
similarly named routines currently implement different numerical contracts.
Future primitives should not depend on model, inference, or dataset modules.
There is currently no public API in this package.
"""

__all__ = ()
