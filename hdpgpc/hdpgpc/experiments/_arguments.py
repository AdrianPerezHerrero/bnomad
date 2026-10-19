"""Common experiment controls; these are example defaults, not model defaults."""
import argparse
import math

NOISE_MODES = ('fixed', 'cluster', 'cluster_diagonal', 'shared', 'shared_diagonal')


def add_common_arguments(parser, *, defaults=None, ordered=False):
    parser.add_argument('--schedule', choices=('offline', 'online', 'online_fast'), default='offline')
    parser.add_argument('--offline-strategy', choices=('legacy', 'split_merge'), default='legacy',
                        help='Offline optimizer; legacy preserves the current birth/deletion search.')
    parser.add_argument('--offline-representer-selection', choices=('sorted', 'fixed'), default='sorted',
                        help='Reselect offline representatives by score, or preserve their initial seeds.')
    parser.add_argument('--split-merge-refine-steps', type=int, default=3,
                        help='Local refinement passes per split/merge branch (default: 3).')
    parser.add_argument('--split-merge-stability-patience', type=int, default=3,
                        help='Consecutive stable outer iterations required to stop (default: 3).')
    parser.add_argument('--split-merge-score-rtol', type=float, default=1e-3,
                        help='Relative score-change tolerance with stable responsibilities (default: 0.001).')
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--max-clusters', type=int, default=30)
    parser.add_argument('--hdp-preset', choices=(
        'less', 'balanced', 'balanced_sticky', 'more', 'more_sticky', 'max'), default='max')
    parser.add_argument('--lds-degrees', '--lds-degrees-of-freedom', type=int, default=10)
    parser.add_argument('--lds-update', choices=('conjugate_batch', 'conjugate_sequential',
                                               'natural_gradient', 'legacy'), default=None)
    parser.add_argument('--lds-scoring', choices=('final', 'historical'), default='final')
    parser.add_argument('--lds-step-size', type=float, default=.2)
    parser.add_argument('--exploration-steps', type=int, default=10)
    parser.add_argument('--hard-assignments', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--birth-lds-kl-per-dimension', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--verbose', choices=('silent', 'progress', 'debug'), default='progress')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fit-gp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--fixed-gp-hyperparameters', dest='fit_gp', action='store_false', help=argparse.SUPPRESS)
    parser.add_argument('--alignment', choices=('none', 'dtw_gp', 'gp_flow', 'gp'), default='none')
    parser.add_argument('--warp', action=argparse.BooleanOptionalAction, default=None,
                        help='Enable alignment (DTW-GP when none selected); --no-warp disables it.')
    parser.add_argument('--normalize-samples', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--plots', '--save-plots', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--min-cluster-samples', type=int, default=1,
                        help='Minimum final assigned sample count for cluster detail plots (default: 1).')
    parser.add_argument('--projection-plots', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--lds-plots', '--plot-lds', action=argparse.BooleanOptionalAction, default=ordered)
    parser.add_argument('--lds-dynamics-plots', '--plot-lds-dynamics',
                        action=argparse.BooleanOptionalAction, default=ordered)
    parser.add_argument('--warp-plots', '--plot-warps', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--birth-plots', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    if defaults:
        parser.set_defaults(**defaults)
    return parser


def alignment_policy(args):
    if args.warp is False:
        return 'none'
    return 'dtw_gp' if args.warp and args.alignment == 'none' else args.alignment


def lds_update(args):
    update = args.lds_update or ('conjugate_batch' if args.schedule == 'offline' else 'conjugate_sequential')
    if update == 'conjugate_batch' and args.schedule != 'offline':
        raise ValueError('conjugate_batch requires --schedule offline')
    return update


def validate_common(args):
    if getattr(args, 'min_cluster_samples', 1) < 1:
        raise ValueError('min_cluster_samples must be a positive integer')
    if (args.iterations < 1 or args.max_clusters < 1 or args.lds_degrees < 3
            or args.exploration_steps < 1 or args.birth_plots < 0
            or not math.isfinite(args.lds_step_size) or not 0 < args.lds_step_size <= 1
            or not 0 <= args.seed < 2**32):
        raise ValueError('Invalid iteration, cluster, LDS, exploration or seed controls')
    lds_update(args)
    if getattr(args, 'split_merge_refine_steps', 3) < 1:
        raise ValueError('split_merge_refine_steps must be positive')
    if getattr(args, 'split_merge_stability_patience', 3) < 1:
        raise ValueError('split_merge_stability_patience must be positive')
    score_rtol = getattr(args, 'split_merge_score_rtol', 1e-3)
    if not math.isfinite(score_rtol) or score_rtol < 0:
        raise ValueError('split_merge_score_rtol must be finite and nonnegative')
    if getattr(args, 'offline_strategy', 'legacy') == 'split_merge' and args.schedule != 'offline':
        raise ValueError('split_merge strategy requires --schedule offline')


def validate_affine(args):
    if alignment_policy(args) != 'none' or args.fit_gp:
        raise ValueError('Fixed affine observations require --no-warp and --no-fit-gp')
    if args.lds_scoring != 'final' or args.lds_update == 'legacy':
        raise ValueError('Fixed affine observations require final LDS scoring and a non-legacy update')
