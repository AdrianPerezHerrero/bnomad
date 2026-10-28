"""Cluster 64 consecutive real MIT-BIH ECG beats and reproduce the README figure.

    python examples/ecg_quickstart.py --output-dir agents/ecg_quickstart/my_run

Uses core package dependencies. Reuses local archived record 208 when available;
otherwise downloads ~4.3 MB from a pinned revision and verifies both SHA-256s.
Annotations are used only after fitting. No waveform files ship in the wheel.
"""
import argparse
from contextlib import redirect_stdout
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import time
from urllib.request import urlopen

DATA_REVISION = '615109a9e474d37626abaff6c51f1c159b26124a'
DATA_BASE = f'https://raw.githubusercontent.com/AdrianPerezHerrero/HDP-GPC/{DATA_REVISION}/hdpgpc/data/mitbih'
DATA_HASHES = {
    '208.npy': '5e885c5cd595d370db68d849d5f3626742ae7085ad7935d53fa5df1df106fc7f',
    '208_labels.npy': '1905f19ff4afd8b09ed52176ae279dec5bd732697cb497804a35dfad5d6b67bf',
}
BEATS = 64
STRIDE = 2


def load_ecg(data_dir=None, *, download=True):
    """Return (X, annotations, model_grid, metadata) for archived beats [0,64).

    Archive windows are already centered per beat. Retain channel 0 and every
    second sample, then divide by one global max-absolute amplitude. No class
    balancing, annotation-based selection, beatwise rescaling or reordering.
    """
    import numpy as np
    from hdpgpc.datasets import load_mitbih_record
    if data_dir is None:
        local = Path(__file__).resolve().parents[1] / 'hdpgpc/data/mitbih'
        data_dir = local if all((local/n).is_file() for n in DATA_HASHES) else Path('agents/ecg_quickstart/data')
    data_dir = Path(data_dir)
    for name, expected in DATA_HASHES.items():
        path = data_dir/name
        if not path.exists():
            if not download:
                raise FileNotFoundError(f'{path} missing; allow download or provide --data-dir')
            print(f'Downloading pinned ECG input: {name}', flush=True)
            with urlopen(f'{DATA_BASE}/{name}', timeout=30) as response:
                payload = response.read(5_000_001)
            if len(payload) > 5_000_000 or hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError(f'ECG download size/checksum mismatch: {name}')
            data_dir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Unexpected ECG archive bytes: {path}')
    record = load_mitbih_record(data_dir, '208')
    X = record.waveforms[:BEATS, ::STRIDE, :1].copy()
    scale = float(np.max(np.abs(X)))
    if X.shape != (64, 45, 1) or not np.isfinite(X).all() or scale <= 0:
        raise ValueError('Unexpected ECG input shape, values or amplitude')
    X /= scale
    grid = np.linspace(0., 4., X.shape[1])[:, None]
    info = {'dataset': 'MIT-BIH Arrhythmia Database', 'record': '208',
            'dataset_url': 'https://physionet.org/content/mitdb/1.0.0/',
            'archive_revision': DATA_REVISION, 'archive_base_url': DATA_BASE,
            'archive_sha256': DATA_HASHES, 'archived_beat_slice': [0, BEATS],
            'channel_index': 0, 'sample_stride': STRIDE, 'original_window_samples': [-27, 63],
            'sampling_frequency_hz': 360, 'global_amplitude_divisor': scale,
            'amplitude_units': 'normalized archived ADC units',
            'model_coordinate': [0., 4.], 'omitted_trailing_annotations': record.omitted_trailing_annotations,
            'input_sha256': hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()}
    return X, record.annotations[:BEATS].copy(), grid, info


def make_config(grid):
    """Static GP means with learned diagonal observation noise per cluster."""
    from hdpgpc import HDPGPCConfig
    from hdpgpc.config import (AlignmentConfig, DynamicsConfig, GPEmissionConfig,
                               HDPConfig, InferenceConfig, MatrixPolicyConfig, MultiOutputConfig,
                               NoisePolicyConfig, NumericalConfig, ObservationConfig)
    return HDPGPCConfig(
        x_basis=grid, verbose=False,
        emission=GPEmissionConfig(ini_lengthscale=.25, bound_lengthscale=(.08, .8),
                    ini_outputscale=1., ini_sigma=.01, bound_sigma=(.001, .05)),
        dynamics=DynamicsConfig(model_type='static', ini_gamma=.03, bound_gamma=(.001, .1),
                    annealing=False, free_deg_MNIV=1, lds_update='conjugate_batch', lds_scoring='final'),
        # Batch updates also learn Sigma for static emissions. Keep C=I as in
        # the original static recipe; each cluster owns its diagonal noise.
        observation=ObservationConfig(C_policy=MatrixPolicyConfig(mode='fixed'),
                    Sigma_policy=NoisePolicyConfig(mode='cluster_diagonal')),
        hdp=HDPConfig(gamma=1., trans_alpha=1., start_alpha=.1, kappa=0., max_models=8),
        alignment=AlignmentConfig(policy='none'),
        inference=InferenceConfig(schedule='offline', it_limit=6, n_explore_steps=3,
                    offline_strategy='legacy', hard_assignments=True, birth_lds_kl_per_dimension=True),
        multi_output=MultiOutputConfig(n_outputs=1, use_snr=False),
        numerics=NumericalConfig(cuda=False, seed=42, num_threads=1, dtype='float64', deterministic=True))


def plot_result(model, X, annotations, grid, scores):
    """Use the shared MIT-BIH report, with roomy panels and pale assignments."""
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D
    import numpy as np
    from hdpgpc.diagnostics import plot_clustering_summary

    figure, = plot_clustering_summary(
        model, X, annotations, grid=grid, record='208', metrics=scores,
        channels=(0,), ordered=True, show_latent=False, min_cluster_samples=1)
    figure.set_layout_engine('constrained', w_pad=.24, h_pad=.20,
                             wspace=.08, hspace=.08, rect=(.02, .02, .96, .96))
    assignment_axis, models_axis = figure.axes[:2]
    gallery = {int(axis.get_title().split()[1]): axis for axis in figure.axes
               if axis.get_title().startswith('Cluster ')}
    cluster_ids = [int(k) for k in model.cluster_ids_]
    pale_colors = {}
    dark_colors = {}
    extremes = [float(np.abs(X).max())]
    for k, axis in gallery.items():
        # Reuse the actual RGBA of the uncertainty band, including its opacity.
        band = next(c for c in axis.collections if isinstance(c, PolyCollection))
        pale_colors[k] = band.get_facecolor()[0]
        dark_colors[k] = axis.lines[0].get_color()
        for path in band.get_paths():
            extremes.append(float(np.abs(path.vertices[:, 1]).max()))

    for k, points in zip(cluster_ids, assignment_axis.collections):
        points.set_facecolor(pale_colors[k])
        points.set_edgecolor('none')
        points.set_sizes([30])
    for k, line in zip(cluster_ids, models_axis.lines):
        line.set_color(dark_colors[k])
    # The legend represents cluster IDs; its hue matches the mean and pale dots.
    legend_axis = next(axis for axis in figure.axes if axis.get_legend() is not None
                       and axis is not models_axis and not axis.get_title()
                       and not axis.texts)
    legend_axis.get_legend().remove()
    legend_axis.legend(handles=[Line2D([], [], color=dark_colors[k], lw=2,
                                     label=f'Cluster {k}') for k in cluster_ids],
                       loc='center', frameon=False, ncol=3, fontsize=9)

    # The model uses a 0--4 coordinate; only tick labels convert it to milliseconds.
    ticks_ms = np.array([-50., 0., 50., 100., 150.])
    samples_per_unit = (X.shape[1]-1)*STRIDE/4.
    tick_positions = (ticks_ms*360/1000+27)/samples_per_unit
    amplitude_limit = max(extremes)*1.12
    for axis in [models_axis, *gallery.values()]:
        axis.set_xticks(tick_positions, [f'{t:.0f}' for t in ticks_ms])
        axis.set(xlabel='Time relative to annotation (ms)', ylabel='Normalized amplitude',
                 xlim=(-.24, 4.24), ylim=(-amplitude_limit, amplitude_limit))
        axis.grid(alpha=.15)
        axis.set_title(axis.get_title(), pad=12)
    assignment_axis.set(xlim=(-3, len(X)+2),
                        ylim=(min(cluster_ids)-.45, max(cluster_ids)+.45),
                        xlabel='Archived beat index (chronological, zero-based)')
    assignment_axis.set_title('Inferred assignments', pad=12)
    # This small example fits comfortably in SVG paths. Keep its traces vector
    # based to avoid process-dependent mixed-mode clipping IDs in the report.
    for artist in figure.findobj():
        if artist.get_rasterized():
            artist.set_rasterized(False)
    figure._suptitle.set_text(
        f'MIT-BIH 208 · channel 0 · {len(X)} consecutive beats · {len(cluster_ids)} clusters\n'
        f'In-sample purity {scores["purity"]:.3f} · ARI {scores["adjusted_rand_index"]:.3f} · '
        f'NMI {scores["normalized_mutual_information"]:.3f}')
    return figure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('agents/ecg_quickstart/run'))
    parser.add_argument('--data-dir', type=Path, help='Directory containing the two pinned record-208 arrays')
    parser.add_argument('--no-download', action='store_true', help='Require local input files')
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error('Choose a new output directory; previous runs are not overwritten')
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update({'svg.hashsalt': 'hdpgpc-ecg-208-v1', 'font.family': 'DejaVu Sans', 'font.size': 9})
    import matplotlib.pyplot as plt
    import numpy as np
    from threadpoolctl import threadpool_limits
    from hdpgpc import HDPGPC
    from hdpgpc.diagnostics import cluster_label_statistics
    X, annotations, grid, data = load_ecg(args.data_dir, download=not args.no_download)
    config = make_config(grid)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/'config.json').write_text(config.to_yaml(), encoding='utf-8')
    start = time.perf_counter()
    with (args.output_dir/'fit.log').open('w', encoding='utf-8') as log:
        with threadpool_limits(limits=1), redirect_stdout(log):
            model = HDPGPC(config).fit(X, grid=grid)
    seconds = time.perf_counter()-start
    scores, _, _, _ = cluster_label_statistics(model.labels_, annotations)
    clusters = []
    fitted_observations = model.observation_parameters_['outputs'][0]
    for cluster in model.cluster_ids_:
        k = int(cluster)
        prediction = model.predictive(k, grid=grid)
        symbols, counts = np.unique(annotations[model.labels_ == k], return_counts=True)
        clusters.append({'id': k, 'mean': prediction.mean.tolist(), 'covariance': prediction.covariance.tolist(),
                         'sigma_diagonal': np.diag(fitted_observations[k]['Sigma']).tolist(),
                         'annotation_counts': dict(zip(symbols.tolist(), counts.tolist()))})
    result = {'labels': model.labels_.tolist(), 'n_clusters': model.n_clusters_, 'scores': scores, 'clusters': clusters}
    figure = plot_result(model, X, annotations, grid, scores)
    figure.savefig(args.output_dir/'ecg_clustering.svg', metadata={'Date': None})
    figure.savefig(args.output_dir/'ecg_clustering.png', dpi=180)
    plt.close(figure)
    np.savez_compressed(args.output_dir/'input.npz', X=X, annotations=annotations, grid=grid)
    report = {'data': data, 'config': config.to_dict(), 'scientific_result': result,
              'annotations_used_for_fit': False, 'evaluation': 'in-sample, same 64 beats as fit',
              'uncertainty_band': 'mean +/-2 sqrt(diag(learned cluster Sigma)); observation noise only',
              'observation_noise': {'mode': config.observation.Sigma_policy.mode,
                  'update': config.dynamics.lds_update, 'initial_variance': config.emission.ini_sigma},
              'fit_seconds': seconds, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                 'seed': 42, 'dtype': 'float64', 'device': 'cpu', 'threads': 1,
                 'versions': {p: metadata.version(p) for p in ('hdpgpc', 'numpy', 'scipy', 'torch', 'gpytorch',
                             'linear-operator', 'pyro-ppl', 'scikit-learn', 'matplotlib')}}}
    report['figure_sha256'] = hashlib.sha256((args.output_dir/'ecg_clustering.svg').read_bytes()).hexdigest()
    (args.output_dir/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f'{BEATS} ECG beats; {model.n_clusters_} clusters; purity={scores["purity"]:.3f}, '
          f'ARI={scores["adjusted_rand_index"]:.3f}, NMI={scores["normalized_mutual_information"]:.3f}; '
          f'fit={seconds:.2f}s; saved {args.output_dir}')


if __name__ == '__main__':
    main()
