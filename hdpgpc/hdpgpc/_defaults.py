"""Common observation/process noise prior defaults.

Dataset workflows retain their documented variance or calibrated-covariance
multiplier interpretation. Explicit recipes and fixed matrix policies override
these starting values.
"""

DEFAULT_SIGMA_PRIOR = 0.1
DEFAULT_GAMMA_PRIOR = 0.1
