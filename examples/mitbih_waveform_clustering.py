"""MITBIH waveform record experiment.

python examples/mitbih_waveform_clustering.py --records 100
python examples/mitbih_waveform_clustering.py --records 100 --schedule online_fast
python examples/mitbih_waveform_clustering.py --records all --resume

Set defaults below or use --help. Shared stages: prepare, calibrate, configure,
fit, report. Chronological records retain HMM transitions. Each run saves its
configuration, log, assignments, metrics and selected plots in its own folder.
See docs/experiment_workflows.md for controls and approximation constraints.
"""
from hdpgpc.experiments.records import build_parser, make_config, run_records

DATASET = 'mitbih'
REPRESENTATION = 'waveform'
# Overrides use the same names as CLI destinations (e.g. iterations=50).
# Scientific configuration is in hdpgpc.experiments.records.make_config.
DEFAULTS = {'lds_preset': 'stable', 'iterations': 50}


def main():
    parser = build_parser(DATASET, REPRESENTATION, __doc__, DEFAULTS)
    args = parser.parse_args()
    reports = run_records(args, parser)
    if any(report['status'] == 'error' for report in reports):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
