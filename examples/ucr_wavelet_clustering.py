"""UCR orthonormal Haar wavelet affine experiment.

python examples/ucr_wavelet_clustering.py --datasets ACSF1
python examples/ucr_wavelet_clustering.py --datasets Coffee --schedule online_fast
python examples/ucr_wavelet_clustering.py --resume

Defaults to ACSF1; use --datasets all for the whole archive. Calibration/normalization use TRAIN only;
TRAIN+TEST clustering metrics are transductive. See docs/experiment_workflows.md.
All variants share the same parser, model call and run/report implementation.
"""
from hdpgpc.experiments.ucr import build_parser, make_config, run_ucr

REPRESENTATION = 'wavelet'
# Add experiment-specific CLI defaults here; scientific configuration is in
# hdpgpc.experiments.ucr.make_config and always uses the ordinary HDPGPC model.
DEFAULTS = {'datasets': ['ACSF1'], 'hdp_preset': 'more', 'hard_assignments': True}


def main():
    parser = build_parser(REPRESENTATION, __doc__, DEFAULTS)
    args = parser.parse_args()
    reports = run_ucr(args, parser)
    if any(report['status'] == 'error' for report in reports):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
