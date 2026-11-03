"""Compare temporal expression states using every measured zebrafish gene.

python examples/zebrafish_mfa_clustering.py --download
python examples/zebrafish_mfa_clustering.py --components 3 --projection-components 12 --resume
python examples/zebrafish_mfa_clustering.py --projection-components 0
python examples/zebrafish_mfa_clustering.py --transition-topology left_to_right

Both configurations use identical observations and HMM priors. Set
--projection-components 0 to pass gene-space observations directly to MFA.
Independent factors fix A=0/Gamma=I; dynamic factors learn cluster A/Gamma.
Replicates are averaged within sampling stages, never made artificial steps.
"""
from hdpgpc.experiments.zebrafish import build_parser, run_zebrafish


def main():
    args = build_parser(__doc__).parse_args()
    reports = run_zebrafish(args)
    if any(report['status'] == 'error' for report in reports):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
