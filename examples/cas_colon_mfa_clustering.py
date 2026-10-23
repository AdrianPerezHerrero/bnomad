"""CAS-Colon MFA emissions using the ordinary HDPGPC inference engines.

python examples/cas_colon_mfa_clustering.py --records 1 --image-size 24 24 --components 10
python examples/cas_colon_mfa_clustering.py --offline-strategy legacy
python examples/cas_colon_mfa_clustering.py --dynamic --hmm-switch --gamma-prior .1

Default: independent assignments, A=0, Gamma=I; learned cluster means, C and
diagonal Sigma. --dynamic permits latent LDS evolution. Output follows the
same run-folder, metric, image-gallery and trajectory conventions as CAS PCA.
"""
from hdpgpc.experiments.cas_colon import build_parser, run_cas_colon


def main():
    parser = build_parser(__doc__, representation='mfa')
    args = parser.parse_args()
    reports = run_cas_colon(args, parser)
    if any(report['status'] == 'error' for report in reports):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
