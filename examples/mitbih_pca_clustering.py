"""MIT-BIH PCA affine record experiment.

python examples/mitbih_pca_clustering.py --records 100 --components 12
python examples/mitbih_pca_clustering.py --records 100 --offline-strategy split_merge --sigma-mode cluster_diagonal
python examples/mitbih_pca_clustering.py --records 100 --schedule online_fast
python examples/mitbih_pca_clustering.py --records all --resume

PCA is calibrated on up to 400 beats, uniformly sampled offline or taken from
a prefix online (replayed during fitting). Its mean and decoder C remain fixed;
A/Gamma learn per cluster. Sigma defaults to the calibrated fixed covariance;
--sigma-prior multiplies that covariance (default 1). Select --sigma-mode to
learn it. --gamma-prior sets the Gamma prior variance times identity.
Chronological records retain HMM transitions.

Set defaults below or use --help. The shared record runner loads annotations,
fits ordinary HDPGPC and saves configuration, calibration, logs, assignments,
metrics, cluster/projection plots and LDS diagnostics in separate run folders.
Fixed affine observations require one channel, no warp and no GP fitting.
See docs/experiment_workflows.md for controls and approximation constraints.
"""
from hdpgpc.experiments.records import build_parser, make_config, run_records

DATASET = 'mitbih'
REPRESENTATION = 'pca'
# Overrides use the same names as CLI destinations (e.g. components=10).
# Scientific configuration is in hdpgpc.experiments.records.make_config.
DEFAULTS = {'components': 12, 'calibration_samples': 400, 'iterations': 50}


def main():
    parser = build_parser(DATASET, REPRESENTATION, __doc__, DEFAULTS)
    args = parser.parse_args()
    reports = run_records(args, parser)
    if any(report['status'] == 'error' for report in reports):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
