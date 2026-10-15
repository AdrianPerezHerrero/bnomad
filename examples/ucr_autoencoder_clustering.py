"""UCR tied linear autoencoder affine experiment.

python examples/ucr_autoencoder_clustering.py --datasets ACSF1
python examples/ucr_autoencoder_clustering.py --datasets Coffee --schedule online_fast
python examples/ucr_autoencoder_clustering.py --resume

Omit --datasets for the whole archive. Calibration/normalization use TRAIN only;
TRAIN+TEST clustering metrics are transductive. See docs/experiment_workflows.md.
All variants share the same parser, model call and run/report implementation.
"""
from hdpgpc.experiments.ucr import build_parser, make_config, run_ucr

REPRESENTATION = 'autoencoder'
# Add experiment-specific CLI defaults here; scientific configuration is in
# hdpgpc.experiments.ucr.make_config and always uses the ordinary HDPGPC model.
DEFAULTS = {'sigma_mode': 'fixed', 'process_noise': 0.05, 'observation_noise_scale': 0.1, 'noise_floor': 1e-8, 'components': 40, 'hard_assignments': True}


def main():
    parser = build_parser(REPRESENTATION, __doc__, DEFAULTS)
    args = parser.parse_args()
    reports = run_ucr(args, parser)
    if any(report['status'] == 'error' for report in reports):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
