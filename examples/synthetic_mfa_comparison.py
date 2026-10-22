"""Compare Bayesian DP/HDP-HMM-MFA with independently fitted finite MFA.

python examples/synthetic_mfa_comparison.py --seeds 7 19 42
Use --cases separated --seeds 7 for a shorter run. Artifacts default to agents/.
"""
from hdpgpc.experiments.mfa import main


if __name__ == '__main__':
    main()
