"""MITBIH autoencoder record experiment.

python examples/mitbih_autoencoder_clustering.py --records 100
python examples/mitbih_autoencoder_clustering.py --records 100 --schedule online_fast
python examples/mitbih_autoencoder_clustering.py --records all --resume

Set defaults below or use --help. Shared stages: prepare, calibrate, configure,
fit, report. Chronological records retain HMM transitions. Each run saves its
configuration, log, assignments, metrics and selected plots in its own folder.
See docs/experiment_workflows.md for controls and approximation constraints.
"""
from hdpgpc.experiments.records import build_parser, make_config, run_records

DATASET = 'mitbih'
REPRESENTATION = 'autoencoder'
# Overrides use the same names as CLI destinations (e.g. iterations=50).
# Scientific configuration is in hdpgpc.experiments.records.make_config.
DEFAULTS = {'gamma_prior': 0.1, 'components': 15, 'calibration_samples': 1000, 'epochs': 500, 'iterations': 50, 'max_clusters': 35, 'noise_floor': 1e-8}


def main():
    parser = build_parser(DATASET, REPRESENTATION, __doc__, DEFAULTS)
    args = parser.parse_args()
    reports = run_records(args, parser)
    if any(report['status'] == 'error' for report in reports):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
