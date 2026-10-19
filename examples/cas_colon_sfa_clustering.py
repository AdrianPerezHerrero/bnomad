"""CAS-Colon ordered images with a calibrated Linear SFA affine emission.

python examples/cas_colon_sfa_clustering.py --records 1
python examples/cas_colon_sfa_clustering.py --records 9 --image-size 24 24 --components 8 --max-series 150
python examples/cas_colon_sfa_clustering.py --records all --resume

Fit slow features using genuine temporal pairs from up to 400 frames per
contiguous segment, then freeze a least-squares affine decoder. The native
HDP-GPC learns cluster-specific A/Gamma through its existing exact Gaussian
observation likelihood. Sigma defaults to fixed diagonal residual covariance;
--sigma-mode selects supported alternatives. --sigma-prior scales that Sigma
and --gamma-prior sets the process variance prior times identity.

Raw SFA features are calibration diagnostics, not noiseless LDS states. The
Gaussian encoder can differ from raw SFA coordinates. Image reports compare
raw SFA reconstruction, smoothed LDS reconstruction and one-step predictions.
See docs/linear_sfa_observation.md for mathematics, sampling and limitations.
"""
from hdpgpc.experiments.cas_colon import build_parser, make_config, run_cas_colon

DATASET = 'cas_colon'
REPRESENTATION = 'sfa'
# Ordinary CLI defaults, not another model configuration system.
DEFAULTS = {}


def main():
    parser = build_parser(__doc__, DEFAULTS, representation=REPRESENTATION)
    args = parser.parse_args()
    reports = run_cas_colon(args, parser)
    if any(report['status'] == 'error' for report in reports):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
