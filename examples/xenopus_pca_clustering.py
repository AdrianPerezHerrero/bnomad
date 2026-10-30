"""Xenopus measured expression states with a calibrated, fixed PCA decoder.

python examples/xenopus_pca_clustering.py --download
python examples/xenopus_pca_clustering.py --download --clutches A B --resume

Each clutch uses all endogenous genes and is fitted separately. Independent
states fix A=0 and learn Gamma per cluster; dynamic states learn A/Gamma.
Both configurations share the PCA decoder and HMM priors. Original sample QC,
time-derived reference intervals and exact NF annotation masks are preserved.
"""
from hdpgpc.experiments.xenopus import build_parser, run_xenopus

REPRESENTATION = 'pca'
DEFAULTS = {}


def main():
    args = build_parser(__doc__, DEFAULTS, representation=REPRESENTATION).parse_args()
    if any(report['status'] == 'error' for report in run_xenopus(args)):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
