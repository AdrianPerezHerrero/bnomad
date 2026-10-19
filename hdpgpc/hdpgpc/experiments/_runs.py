"""Run provenance, logging and common artifacts for all clustering experiments."""
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from functools import lru_cache
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from ..diagnostics import Tee

SUMMARY_FIELDS = ('dataset', 'run_id', 'experiment', 'status', 'fitted', 'classes', 'clusters',
                  'purity_in_sample', 'ari_all', 'nmi_all', 'seconds', 'output_directory', 'error')


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Cannot serialize {type(value).__name__}')


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, default=_json_default, indent=2,
                                     allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def source_digest(path):
    path = Path(path)
    digest = hashlib.sha256()
    if path.is_file():
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()
    files = sorted(path.rglob('*.txt'))
    if not files:
        raise ValueError(f'No source data found: {path}')
    for file in files:
        digest.update(str(file.relative_to(path)).encode())
        with file.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def code_digest():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def experiment_settings(args):
    return {key: value for key, value in vars(args).items()
            if key not in ('output_dir', 'resume', 'download', 'datasets', 'records',
                           'list_records', 'single_dataset', 'archive_sha256') and not key.startswith('_')}


class ExperimentRun:
    """One dataset/settings execution with a stable content-derived directory."""
    def __init__(self, root, dataset, experiment, settings, fingerprint):
        if Path(dataset).name != dataset or dataset in ('', '.', '..'):
            raise ValueError('Dataset/run label must be a single path component')
        self.root, self.dataset, self.experiment = Path(root), dataset, experiment
        self.settings, self.fingerprint = settings, fingerprint
        identity = {'experiment': experiment, 'settings': settings,
                    'source_sha256': fingerprint, 'code_sha256': code_digest()}
        self.run_id = hashlib.sha256(json.dumps(identity, sort_keys=True,
                                               default=_json_default).encode()).hexdigest()[:16]
        self.path = self.root / dataset / self.run_id
        self.identity = identity

    def complete(self):
        report_path = self.path / 'results.json'
        if not report_path.is_file():
            return False
        try:
            report = json.loads(report_path.read_text(encoding='utf-8'))
            return (report.get('status') == 'ok' and report.get('run_id') == self.run_id
                    and all((self.path / item).is_file() for item in report['artifacts']))
        except (ValueError, KeyError, TypeError):
            return False

    def __enter__(self):
        self.path.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()
        self.stack = ExitStack()
        log = self.stack.enter_context((self.path / 'clustering.log').open('w', encoding='utf-8', buffering=1))
        self.stack.enter_context(redirect_stdout(Tee(sys.stdout, log)))
        self.stack.enter_context(redirect_stderr(Tee(sys.stderr, log)))
        write_json(self.path / 'settings.json', self.identity)
        write_json(self.path / 'results.json', {'status': 'running', 'run_id': self.run_id})
        return self

    def __exit__(self, typ, value, tb):
        if typ is not None:
            traceback.print_exception(typ, value, tb)
            write_json(self.path / 'results.json', {
                'status': 'error', 'run_id': self.run_id, 'dataset': self.dataset,
                'experiment': self.experiment, 'error': f'{typ.__name__}: {value}'})
        self.stack.close()
        return False


def append_summary(root, report):
    """Upsert by dataset/run ID, avoiding duplicate rows on reruns or resume."""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    path = root / 'results.csv'
    rows = []
    if path.is_file():
        with path.open(newline='', encoding='utf-8') as stream:
            if next(csv.reader(stream), None) != list(SUMMARY_FIELDS):
                path = root / 'results_standardized.csv'
    if path.is_file():
        with path.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(SUMMARY_FIELDS):
                raise ValueError(f'Incompatible standardized results table: {path}')
            rows = list(reader)
    rows = [row for row in rows if (row['dataset'], row['run_id']) !=
            (str(report['dataset']), str(report['run_id']))]
    metrics = report.get('metrics', {})
    row = {key: report.get(key, '') for key in SUMMARY_FIELDS}
    row.update(purity_in_sample=metrics.get('purity', ''),
               ari_all=metrics.get('adjusted_rand_index', ''),
               nmi_all=metrics.get('normalized_mutual_information', ''))
    rows.append(row)
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def log_configuration(config, args, *, source='configured'):
    """One consistent initial summary, respecting silent verbosity."""
    if args.verbose == 'silent':
        return
    from ..config._models import resolved_lds_policies
    from ..diagnostics import format_lds_priors
    print(f'Schedule={args.schedule}; HMM={config.hdp.hmm_switch}; '
          f'alignment={config.alignment.policy}; fit_gp={config.emission.fit_gp}')
    if args.schedule == 'offline':
        print(f'Offline strategy={config.inference.offline_strategy}; '
              f'proposal refinement steps={config.inference.split_merge_refine_steps}; '
              f'representer selection={config.inference.offline_representer_selection}')
        if config.inference.offline_strategy == 'split_merge':
            print(f'Split/merge stopping: score rtol={config.inference.split_merge_score_rtol:g}; '
                  f'stable iterations={config.inference.split_merge_stability_patience}')
    print(f'LDS update={config.dynamics.lds_update}; scoring={config.dynamics.lds_scoring}; '
          f'hard assignments={config.inference.hard_assignments}; seed={args.seed}')
    print(format_lds_priors(dimension=len(config.x_basis), model_type=config.dynamics.model_type,
          observation_noise=config.emission.ini_sigma, process_noise=config.dynamics.ini_gamma,
          degrees_of_freedom=args.lds_degrees, source=source,
          policies=resolved_lds_policies(config)))


def save_results(run, model, values, labels, grid, args, *, ordered,
                 metadata=None, codes=None, projection_name='Latent',
                 row_metadata=None, annotation_colors=None, extra_figures=None,
                 positive_label=None, summary_plotter=None, evaluation_mask=None):
    """Common report/plot schema; optional diagnostics never refit the model."""
    import matplotlib.pyplot as plt
    from ..diagnostics import (cluster_label_statistics, format_confusion_matrix,
        plot_clustering_summary, plot_embedding_summary, model_lds_arrays,
        model_lds_dynamics, model_warp_arrays, plot_lds_diagnostics,
        plot_lds_dynamics, plot_warp_diagnostics, plot_birth_warp_diagnostic)
    from ..diagnostics._plot_selection import cluster_plot_ids, empty_cluster_figure
    values, labels = np.asarray(values), np.asarray(labels)
    min_cluster_samples = getattr(args, 'min_cluster_samples', 1)
    plotted_ids = set(int(cluster) for cluster in cluster_plot_ids(model.labels_, min_cluster_samples))
    if any(int(cluster) >= len(output) for output in model._backend.gpmodels
           for cluster in model.cluster_ids_):
        raise RuntimeError('Inference retained assignments for missing cluster components; '
                           'a consistent final report cannot be generated')
    evaluated = (np.ones(len(labels), dtype=bool) if evaluation_mask is None
                 else np.asarray(evaluation_mask, dtype=bool))
    if evaluated.shape != (len(labels),) or not evaluated.any():
        raise ValueError('Evaluation mask must select at least one aligned observation')
    scores, classes, confusion, mapping = cluster_label_statistics(
        np.asarray(model.labels_)[evaluated], labels[evaluated])
    if evaluation_mask is None:
        assigned = np.array([mapping[int(cluster)] for cluster in model.labels_])
    else:
        # A component containing only ambiguous frames has no reference class.
        for cluster in model.cluster_ids_:
            mapping.setdefault(int(cluster), None)
        assigned = np.array([mapping[int(cluster)] for cluster in model.labels_], dtype=object)
        scores.update(evaluation_frames=int(evaluated.sum()),
                      excluded_evaluation_frames=int((~evaluated).sum()))
    if positive_label is not None:
        actual, predicted = labels[evaluated] == positive_label, assigned[evaluated] == positive_label
        tp = int((actual & predicted).sum()); fp = int((~actual & predicted).sum())
        fn = int((actual & ~predicted).sum())
        scores.update(positive_label_precision_in_sample=tp / (tp + fp) if tp + fp else 0.,
                      positive_label_recall_in_sample=tp / (tp + fn) if tp + fn else None,
                      positive_label_f1_in_sample=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None)
    print(f'{run.dataset}: clusters={model.n_clusters_}, purity={scores["purity"]:.3f}, '
          f'ARI={scores["adjusted_rand_index"]:.3f}, NMI={scores["normalized_mutual_information"]:.3f}')
    print(format_confusion_matrix(classes, confusion))
    if evaluation_mask is not None:
        print(f'Label metrics: {evaluated.sum()}/{len(labels)} frames; '
              f'{(~evaluated).sum()} ambiguous/unlabeled frames excluded.')
    print('Cluster-majority mapping and label scores are evaluated in-sample.')
    (run.path / 'config.yaml').write_text(model.config.to_yaml(), encoding='utf-8')
    columns = {} if row_metadata is None else dict(row_metadata)
    if any(len(value) != len(labels) for value in columns.values()):
        raise ValueError('Assignment metadata must have one row per observation')
    with (run.path / 'assignments.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['fit_row', 'true_label', 'cluster', 'cluster_majority_label', *columns])
        for index, cluster in enumerate(model.labels_):
            writer.writerow([index, labels[index], int(cluster), assigned[index],
                             *(column[index] for column in columns.values())])
    matrices = [model_lds_arrays(model, output=output) for output in range(values.shape[2])]
    np.savez_compressed(run.path / 'lds_parameters.npz', **{
        f'output_{output}_cluster_{cluster}_{name}': matrix
        for output, group in enumerate(matrices) for cluster, parameters in group.items()
        for name, matrix in parameters.items() if name in ('A', 'C', 'Gamma', 'Sigma')})
    failures = {}
    def save(figure, name):
        try:
            figure.savefig(run.path / name, dpi=150)
        finally:
            plt.close(figure)
    if args.plots:
        summary_plotter = plot_clustering_summary if summary_plotter is None else summary_plotter
        for output, figure in enumerate(summary_plotter(
                model, values, labels, grid=grid, record=run.dataset, metrics=scores,
                ordered=ordered, warped=model.config.inference.warp,
                show_latent=ordered or args.lds_plots or args.lds_dynamics_plots,
                annotation_colors=annotation_colors,
                **({'min_cluster_samples': min_cluster_samples} if min_cluster_samples > 1 else {}))):
            save(figure, f'clusters_output_{output}.png')
        if args.projection_plots:
            projection = values.reshape(len(values), -1) if codes is None else codes
            save(plot_embedding_summary(projection, model.labels_, labels, scores,
                 title=run.dataset, projection_name=projection_name), 'projection.png')
        for output, group in enumerate(matrices):
            shown = {cluster: values for cluster, values in group.items() if cluster in plotted_ids}
            lds_title = (run.dataset if min_cluster_samples == 1 else
                         f'{run.dataset} | clusters with >= {min_cluster_samples} assigned samples')
            if args.lds_plots:
                if shown:
                    for page, figure in enumerate(plot_lds_diagnostics(shown, title=lds_title), 1):
                        save(figure, f'lds_parameters_output_{output}_{page:02d}.png')
                else:
                    save(empty_cluster_figure(lds_title, min_cluster_samples),
                         f'lds_parameters_output_{output}_01.png')
            if args.lds_dynamics_plots:
                if shown:
                    dynamics = {cluster: values for cluster, values in
                                model_lds_dynamics(model, grid=grid, output=output).items()
                                if cluster in plotted_ids}
                    figure = plot_lds_dynamics(dynamics, title=lds_title)
                else:
                    figure = empty_cluster_figure(lds_title, min_cluster_samples)
                save(figure, f'lds_dynamics_output_{output}.png')
            if args.warp_plots and model.config.inference.warp:
                offsets, std = model_warp_arrays(model, grid, output=output)
                save(plot_warp_diagnostics(grid, offsets, labels=model.labels_,
                     posterior_std=std, title=run.dataset), f'warps_output_{output}.png')
        if args.birth_plots:
            for index, record in enumerate(model.birth_diagnostics_, 1):
                if not record.get('rows'):
                    failures[f'birth_{index}'] = record.get('diagnostic_error', 'no recorded curves')
                    continue
                save(plot_birth_warp_diagnostic(record, title=run.dataset), f'birth_warp_{index:02d}.png')
        for name, figure in (extra_figures() if extra_figures else {}).items():
            save(figure, name + '.png')
    report = {'schema_version': 1, 'dataset': run.dataset, 'run_id': run.run_id,
              'experiment': run.experiment, 'status': 'ok', 'fitted': len(values),
              'classes': len(classes), 'clusters': model.n_clusters_, 'metrics': scores,
              'confusion_classes': classes, 'confusion': confusion,
              'cluster_majority_labels': mapping,
              'cluster_cap_reached': model.n_clusters_ >= args.max_clusters,
              'ordered': ordered, 'warp': model.config.inference.warp,
              'plot_selection': {'min_cluster_samples': min_cluster_samples,
                                 'shown_cluster_ids': sorted(plotted_ids),
                                 'hidden_cluster_ids': sorted(int(cluster) for cluster in model.cluster_ids_
                                                              if int(cluster) not in plotted_ids)},
              'metadata': metadata or {}, 'diagnostic_omissions': failures,
              'split_merge_history': getattr(model, 'split_merge_history_', []),
              'split_merge_convergence': getattr(model, 'split_merge_convergence_', {}),
              'seconds': round(time.perf_counter() - run.started, 3),
              'output_directory': str(run.path.resolve()),
              'artifacts': sorted(str(path.relative_to(run.path)) for path in run.path.rglob('*')
                                  if path.is_file() and path.name != 'results.json')}
    write_json(run.path / 'results.json', report)
    return report
