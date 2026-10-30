"""Xenopus expression states with a trained, fixed affine autoencoder decoder.

python examples/xenopus_autoencoder_clustering.py --download
python examples/xenopus_autoencoder_clustering.py --download --resume
python examples/xenopus_autoencoder_clustering.py --configurations independent dynamic

Use the existing tied linear orthonormal autoencoder, calibrated separately
per clutch and frozen during state inference. Defaults fit both clutches with
dynamic states and left-to-right assignments using the selected configuration
below. Observation covariance multipliers default to 0.01 for A and 0.005 for
B; an explicit --sigma-prior overrides both. CLI arguments override defaults.
Shared arguments, full-gene preparation, sample QC, reference segmentation
plots and output schemas match the other expression experiments.
"""
from hdpgpc.experiments.xenopus import build_parser, run_xenopus

REPRESENTATION = 'autoencoder'
DEFAULTS = dict(
    clutches=['A', 'B'],
    hdp_preset='more',
    sigma_prior=None,  # Resolve per clutch unless supplied explicitly on the CLI.
    gamma_prior=0.1,
    transition_topology='left_to_right',
    configurations=['dynamic'],
    lds_degrees=3,
    components=10,
    projection_components=200,
)
SIGMA_PRIOR_DEFAULTS = {'A': 0.01, 'B': 0.005}


def main():
    args = build_parser(__doc__, DEFAULTS, representation=REPRESENTATION).parse_args()
    reports = run_xenopus(args, sigma_prior_defaults=SIGMA_PRIOR_DEFAULTS)
    if any(report['status'] == 'error' for report in reports):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
