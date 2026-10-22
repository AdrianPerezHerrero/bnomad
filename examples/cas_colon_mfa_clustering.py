"""CAS-Colon image clustering with Bayesian factor-analysis emissions.

python examples/cas_colon_mfa_clustering.py --records 1 --max-series 150
python examples/cas_colon_mfa_clustering.py --records 1 --components 8 --initial-clusters 4
python examples/cas_colon_mfa_clustering.py --records all --resume
python examples/cas_colon_mfa_clustering.py --records 1 --no-hmm-switch

Extract Frame.rar and Label.rar into agents/data/cas_colon/CAS-ColonDB first.
Frames become grayscale pixel vectors; each MFA component learns its own image
mean and loading matrix. HMM assignments model temporal regimes by default;
factors within a regime are independent, not continuous LDS trajectories.
Anatomical annotations are used only for in-sample metrics and plots.

See docs/cas_colon_mfa.md for priors, outputs and runtime controls.
"""
from hdpgpc.experiments.cas_colon_mfa import build_parser, run_cas_colon_mfa

DATASET = 'cas_colon'
REPRESENTATION = 'mfa'
DEFAULTS = {}


def main():
    parser = build_parser(__doc__, DEFAULTS)
    args = parser.parse_args()
    reports = run_cas_colon_mfa(args)
    if any(report['status'] == 'error' for report in reports):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
