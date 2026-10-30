"""Compare expression states in the dense Xenopus GSE65785 time course.

python examples/xenopus_mfa_clustering.py --download
python examples/xenopus_mfa_clustering.py --download --clutches A B --resume
python examples/xenopus_mfa_clustering.py --projection-components 0

Use unsmoothed polyA+ gene TPMs, with all endogenous genes retained. Clutches
are fitted separately. Coarse targets are time-derived NF intervals; exact
published NF anchors are scored only where available. They are evaluation
references, not independent per-sample developmental ground truth.
"""
from hdpgpc.experiments.xenopus import build_parser, run_xenopus


def main():
    reports = run_xenopus(build_parser(__doc__).parse_args())
    if any(report['status'] == 'error' for report in reports):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
