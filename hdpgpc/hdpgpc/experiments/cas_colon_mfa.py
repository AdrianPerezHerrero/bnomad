"""CAS-Colon workflow for the public HDPMFA API and learned image emissions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ..config import HDPConfig, HDPMFAConfig, NumericalConfig
from ..datasets.cas_colon import cas_colon_annotations, load_cas_colon_frames
from ..diagnostics import cluster_label_statistics, format_confusion_matrix
from ..model import HDPMFA
from ._runs import ExperimentRun, append_summary, experiment_settings, write_json


def build_parser(description=None, defaults=None):
    parser = argparse.ArgumentParser(description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path,
        default=Path.cwd() / 'agents/data/cas_colon/CAS-ColonDB')
    parser.add_argument('--annotations', type=Path, default=None)
    parser.add_argument('--records', nargs='+', default=['1'], help='Video IDs, or all.')
    parser.add_argument('--image-size', type=int, nargs=2, default=(24, 24), metavar=('HEIGHT', 'WIDTH'))
    parser.add_argument('--crop', type=int, nargs=4, default=None, metavar=('LEFT', 'TOP', 'RIGHT', 'BOTTOM'))
    parser.add_argument('--frame-step', type=int, default=1)
    parser.add_argument('--max-series', type=int, default=None, help='Limit selected frames per video.')
    parser.add_argument('--gap-policy', choices=('split', 'error'), default='split')
    parser.add_argument('--components', '--factors', type=int, default=4,
        help='Factor rank per MFA component; not the number of clusters (default: 4).')
    parser.add_argument('--initial-clusters', type=int, default=4,
        help='Initial variational support size (default: 4).')
    parser.add_argument('--max-clusters', type=int, default=12)
    parser.add_argument('--component-strategy', choices=('fixed', 'birth_prune'), default='birth_prune')
    parser.add_argument('--hmm-switch', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--hdp-preset', choices=('less', 'balanced', 'balanced_sticky', 'more',
        'more_sticky', 'more_extra_sticky', 'max'), default='balanced')
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--tolerance', type=float, default=1e-6)
    parser.add_argument('--exploration-steps', type=int, default=2)
    parser.add_argument('--proposal-refine-steps', type=int, default=3)
    parser.add_argument('--prune-threshold', type=float, default=1.)
    parser.add_argument('--loading-prior-mean', type=float, default=0.)
    parser.add_argument('--loading-prior-precision', type=float, default=.1,
        help='Gaussian precision for augmented [image mean, loadings] rows (default: 0.1).')
    parser.add_argument('--noise-initial', type=float, default=.01,
        help='Initial shared diagonal pixel variance, not standard deviation (default: 0.01).')
    parser.add_argument('--noise-floor', type=float, default=1e-6)
    parser.add_argument('--learn-noise', action=argparse.BooleanOptionalAction, default=True,
        help='Learn one shared diagonal pixel-noise vector; --no-learn-noise fixes its initial value.')
    parser.add_argument('--chunk-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--verbose', choices=('silent', 'progress', 'debug'), default='progress')
    parser.add_argument('--plots', '--save-plots', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--frames-per-cluster', type=int, default=4)
    parser.add_argument('--min-cluster-samples', type=int, default=1,
        help='Filter image-gallery rows only; assignments, metrics and summary still include all frames.')
    parser.add_argument('--output-dir', type=Path, default=Path.cwd() / 'agents/cas_colon_mfa_runs')
    parser.add_argument('--resume', action='store_true')
    if defaults:
        parser.set_defaults(**defaults)
    return parser


def make_config(args, *, observations=None):
    """Translate CLI controls directly to the existing typed MFA recipe."""
    # An isolated/short segment can have fewer frames than initial support.
    # Persist this effective M in config.yaml; never silently change factor rank.
    support = args.initial_clusters if observations is None else min(args.initial_clusters, observations)
    return HDPMFAConfig(n_factors=args.components,
        hdp=HDPConfig(M=support, max_models=args.max_clusters,
                      hdp_hyp=args.hdp_preset, hmm_switch=args.hmm_switch),
        numerics=NumericalConfig(seed=args.seed, num_threads=args.threads),
        loading_prior_mean=args.loading_prior_mean, loading_prior_precision=args.loading_prior_precision,
        noise_initial=args.noise_initial, noise_floor=args.noise_floor, learn_noise=args.learn_noise,
        max_iter=args.iterations, tol=args.tolerance, chunk_size=args.chunk_size,
        verbose=args.verbose, component_strategy=args.component_strategy,
        n_explore_steps=args.exploration_steps, proposal_refine_steps=args.proposal_refine_steps,
        prune_threshold=args.prune_threshold)


def _run_segment(sequence, args, run, segment):
    values = sequence.images.reshape(len(sequence.images), -1)
    config = make_config(args, observations=len(values))
    (run.path / 'config.yaml').write_text(config.to_yaml(), encoding='utf-8')
    if args.verbose != 'silent':
        print(f'CAS-Colon video {sequence.video}, segment {segment}: {len(values)} ordered frames; '
              f'grayscale={tuple(args.image_size)} in [0,1]; factors={config.n_factors}')
        print(f'Offline MFA: initial support={config.hdp.M}; cap={config.hdp.max_models}; '
              f'strategy={config.component_strategy}; HMM={config.hdp.hmm_switch}; preset={config.hdp.hdp_hyp}')
        print(f'Augmented row prior: mean={args.loading_prior_mean:g}, precision={args.loading_prior_precision:g}; '
              f'shared diagonal noise: variance={args.noise_initial:g}, learn={args.learn_noise}, '
              f'floor={args.noise_floor:g}')
    started = perf_counter()
    model = HDPMFA(config).fit(values, sequence_lengths=(len(values),))
    fit_seconds = perf_counter() - started
    assignments = model.labels_
    evaluated = sequence.evaluation_mask
    if evaluated.any():
        metrics, classes, confusion, mapping = cluster_label_statistics(
            assignments[evaluated], sequence.annotations[evaluated])
    else:
        metrics, classes, confusion, mapping = {}, np.array([], dtype=str), np.zeros((0, 0), dtype=int), {}
    metrics.update(evaluation_frames=int(evaluated.sum()), excluded_evaluation_frames=int((~evaluated).sum()))
    assigned_labels = [mapping.get(int(cluster), 'unlabeled') for cluster in assignments]
    print(f'{run.dataset}: clusters={model.n_clusters_}/{model.n_components_} support slots; '
          f'ELBO={model.elbo_:.6g}; iterations={model.n_iter_}; converged={model.converged_}')
    if evaluated.any():
        print(f'Purity={metrics["purity"]:.3f}; ARI={metrics["adjusted_rand_index"]:.3f}; '
              f'NMI={metrics["normalized_mutual_information"]:.3f}')
        print(format_confusion_matrix(classes, confusion))
    print(f'In-sample label evaluation: {evaluated.sum()}/{len(values)} frames; '
          'ambiguous labels excluded. Labels were not used for fitting.')

    model.save(run.path / 'model.json')
    with (run.path / 'assignments.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['fit_row', 'second', 'filename', 'true_label', 'cluster',
            'cluster_majority_label', 'evaluation_included', 'annotation_options', 'source_variants'])
        for index, cluster in enumerate(assignments):
            writer.writerow([index, sequence.seconds[index], sequence.filenames[index],
                sequence.annotations[index], int(cluster), assigned_labels[index], bool(evaluated[index]),
                '|'.join(sequence.annotation_options[index]), '|'.join(sequence.source_variants[index])])
    # These are component-local coordinates selected with STORED training MAP
    # assignments. Do not recompute assignments or average unrelated factor bases.
    factors = model.transform_components(values)
    local_means = factors.mean[np.arange(len(values)), assignments]
    arrays = dict(labels=assignments, responsibilities=model.responsibilities_,
        occupancies=model.occupancies_, means=model.means_, loadings=model.loadings_, noise=model.noise_,
        seconds=sequence.seconds, factor_means=local_means, factor_covariance=factors.covariance,
        elbo_history=model.elbo_history_, evaluation_mask=evaluated)
    if args.hmm_switch:
        arrays.update(transition_matrix=model.transition_matrix_, start_probabilities=model.start_probabilities_)
    np.savez_compressed(run.path / 'mfa_parameters.npz', **arrays)
    write_json(run.path / 'history.json', dict(iterations=model.history_, structure=model.structure_history_))
    write_json(run.path / 'confusion_matrix.json', dict(classes=classes, counts=confusion,
        cluster_majority_labels={str(key): value for key, value in mapping.items()}))
    artifacts = ['settings.json', 'config.yaml', 'model.json', 'assignments.csv', 'mfa_parameters.npz',
                 'history.json', 'confusion_matrix.json', 'clustering.log']
    if args.plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from ..diagnostics import plot_mfa_image_summary, plot_mfa_image_gallery
        figures = {
            'clusters.png': plot_mfa_image_summary(model, values, sequence.annotations,
                times=sequence.seconds, evaluation_mask=evaluated, title=f'CAS-Colon video {sequence.video}',
                random_state=args.seed),
            'image_gallery.png': plot_mfa_image_gallery(model, sequence.images, sequence.annotations,
                factor_means=local_means, frames_per_cluster=args.frames_per_cluster,
                min_cluster_samples=args.min_cluster_samples, evaluation_mask=evaluated, random_state=args.seed),
        }
        for name, figure in figures.items():
            try:
                figure.savefig(run.path / name, dpi=150)
                artifacts.append(name)
            finally:
                plt.close(figure)
    report = dict(status='ok', dataset=run.dataset, run_id=run.run_id, experiment=run.experiment,
        fitted=len(values), classes=len(classes), clusters=model.n_clusters_, support=model.n_components_,
        metrics=metrics, seconds=perf_counter() - run.started, fit_seconds=fit_seconds,
        output_directory=str(run.path), artifacts=artifacts, elbo=model.elbo_,
        elbo_components=model.elbo_components_, iterations=model.n_iter_, converged=model.converged_,
        metadata=dict(video=sequence.video, segment=segment, image_shape=list(args.image_size),
            pixel_range=[0, 1], grayscale=True, time_units='filename seconds', frame_step=args.frame_step,
            factor_dimension=model.n_factors_, hmm_switch=args.hmm_switch,
            emission='learned cluster-specific image means/loadings; shared diagonal noise',
            factors='independent standard-normal factors conditional on regime; no continuous LDS',
            boundaries='each contiguous segment fitted independently; no transition across gaps',
            evaluation='in-sample clustering; annotations used only for evaluation and plots',
            factor_coordinates='component-local, not a common projection across clusters'))
    write_json(run.path / 'results.json', report)
    return report


def run_cas_colon_mfa(args):
    """Fit one model per contiguous segment using the existing CAS-Colon loader."""
    make_config(args).validate()
    if (args.initial_clusters < 1 or min(args.image_size) < 2 or args.frame_step < 1
            or (args.max_series is not None and args.max_series < 2)
            or args.frames_per_cluster < 1 or args.min_cluster_samples < 1):
        raise ValueError('Image, frame, cluster and gallery sizes must be positive (images/subsets at least 2)')
    if args.components > int(np.prod(args.image_size)):
        raise ValueError('Factor rank cannot exceed the flattened image dimension')
    if not np.isfinite(args.noise_initial) or args.noise_initial < args.noise_floor:
        raise ValueError('noise-initial must be a variance at least noise-floor')
    if not np.isfinite(args.loading_prior_mean) or not np.isfinite(args.loading_prior_precision) or args.loading_prior_precision <= 0:
        raise ValueError('Loading prior mean must be finite and precision positive')
    annotation_path, index = cas_colon_annotations(args.data_dir, args.annotations)
    videos = sorted(index, key=int) if args.records == ['all'] else [str(int(video)) for video in args.records]
    reports = []
    for video in dict.fromkeys(videos):
        sequences = load_cas_colon_frames(args.data_dir, video, frame_records=index,
            image_size=tuple(args.image_size), crop=args.crop, frame_step=args.frame_step,
            max_frames=args.max_series, gap_policy=args.gap_policy)
        for segment, sequence in enumerate(sequences):
            if len(sequence.images) < 2:
                continue
            digest = hashlib.sha256(annotation_path.read_bytes())
            digest.update(sequence.images.tobytes())
            digest.update(sequence.seconds.tobytes())
            digest.update('\n'.join(sequence.filenames).encode())
            run = ExperimentRun(args.output_dir, f'video_{video}_segment_{segment:02d}',
                                'cas_colon_mfa', experiment_settings(args), digest.hexdigest())
            if args.resume and run.complete():
                report = json.loads((run.path / 'results.json').read_text(encoding='utf-8'))
            else:
                try:
                    with run:
                        report = _run_segment(sequence, args, run, segment)
                except Exception:
                    report = json.loads((run.path / 'results.json').read_text(encoding='utf-8'))
            append_summary(args.output_dir, report)
            reports.append(report)
            print(f'{run.dataset}: {report["status"]}; {run.path}')
    if not reports:
        raise ValueError('No contiguous CAS-Colon segment has at least two selected frames')
    return reports
