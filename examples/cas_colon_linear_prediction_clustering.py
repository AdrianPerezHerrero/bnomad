"""CAS-Colon affine features trained for reconstruction and linear prediction.

python examples/cas_colon_linear_prediction_clustering.py --records 1
python examples/cas_colon_linear_prediction_clustering.py --records 1 --components 20 --prediction-weight 0.1 --min-cluster-samples 5
python examples/cas_colon_linear_prediction_clustering.py --records 1 --prediction-weight 0
python examples/cas_colon_linear_prediction_clustering.py --records all --resume

Calibrate normalized linear features on up to 400 ordered-frame endpoints using
genuine temporal pairs. Freeze the affine decoder and let the existing HDP-GPC
learn cluster-specific A/Gamma. The calibration predictor B is saved separately
and never installed as the cluster transition matrix. Sigma defaults to fixed
diagonal reconstruction noise; supported --sigma-mode policies remain available.

--prediction-weight balances normalized reconstruction and latent prediction;
zero is the PCA control on the same calibration rows. --epochs, --learning-rate
and --prediction-whitening-components control the independent calibration.
Reports retain image galleries, trajectories, purity, metrics and plot filtering,
and add calibration loss/source/target/reconstruction/prediction diagnostics.
See docs/linear_prediction_observation.md for objective and inference semantics.
"""
from hdpgpc.experiments.cas_colon import build_parser, make_config, run_cas_colon

DATASET = 'cas_colon'
REPRESENTATION = 'linear_prediction'
# Ordinary CLI defaults, not another model configuration system.
DEFAULTS = {}


def main():
    parser = build_parser(__doc__, DEFAULTS, representation=REPRESENTATION)
    args = parser.parse_args()
    reports = run_cas_colon(args, parser)
    if any(report['status'] == 'error' for report in reports):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
