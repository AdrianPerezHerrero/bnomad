"""Generate the eight consistent, public-API research tutorials (no execution)."""
from pathlib import Path
import json
import textwrap

ROOT = Path(__file__).resolve().parents[1]

SETUP = '''from pathlib import Path
import runpy
import zipfile
import hashlib
import json

# A local checkout or the previously uploaded bundle is reused automatically.
roots = [Path.cwd() / '.hdpgpc-colab-bundle', Path.cwd(), *Path.cwd().parents]
tutorial_root = next((p for p in roots if (p / 'examples/colab_setup.py').is_file()), None)
if tutorial_root is None:
    from google.colab import files
    print('Upload hdpgpc-colab.zip (the current wheel, shared helpers and record 100).')
    uploaded = files.upload()
    archive_path = next(Path(name) for name in uploaded if name.endswith('.zip'))
    tutorial_root = (Path.cwd() / '.hdpgpc-colab-bundle').resolve()
    with zipfile.ZipFile(archive_path) as archive:
        assert sum(item.file_size for item in archive.infolist()) < 20_000_000
        for item in archive.infolist():
            assert (tutorial_root / item.filename).resolve().is_relative_to(tutorial_root)
        archive.extractall(tutorial_root)
    manifest = json.loads((tutorial_root / 'manifest.json').read_text())
    for name, expected in manifest.items():
        path = (tutorial_root / name).resolve()
        assert path.is_relative_to(tutorial_root)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
runpy.run_path(str(tutorial_root / 'examples/colab_setup.py'))['start'](tutorial_root)
'''

IMPORTS = '''import json
import time
import io
from contextlib import redirect_stdout
from dataclasses import replace
import numpy as np
import matplotlib.pyplot as plt
from hdpgpc import HDPGPC, HDPGPCConfig
from hdpgpc.config import (HDPConfig, GPEmissionConfig, DynamicsConfig,
                          AlignmentConfig, InferenceConfig, MultiOutputConfig, NumericalConfig)

numerics = NumericalConfig(cuda=False, seed=42, num_threads=1,
                           dtype='float64', deterministic=True)
numerics.apply()  # Python, NumPy, PyTorch and Pyro RNGs; float64 and one CPU thread
versions = utils['environment']()
'''

DATA = '''record, annotations = utils['load_record']()
start, n_beats, stride = 1700, 6, 6
raw = record[start:start + n_beats]
sampled = raw[:, ::stride, :]
# Freeze preprocessing using the first observed beat; do not peek at future beats.
center = sampled[0].mean(axis=0)
scale = sampled[0].std(axis=0)
assert np.all(scale > 0)
scaled = (sampled - center) / scale
grid = np.linspace(0., 4., sampled.shape[1])[:, None]
X = scaled[:, :, :N_OUTPUTS].copy()
print('Model input (observations, points, outputs):', X.shape)
print('Center:', center, 'Scale:', scale)
utils['plot_input'](raw, grid, X)
plt.show()
'''

SPECS = [
('00_quickstart', 'A first HDP-GPC analysis', 'static', 'offline', 1, False,
 'Learn the complete workflow: load real waveforms, specify a static model, fit and inspect its public results.',
 'The simplest baseline here is static, single-channel, unaligned batch clustering. Static refers to component dynamics, not independence of cluster assignments.'),
('01_static_clustering', 'Static waveform clusters', 'static', 'offline', 1, False,
 'Understand the actual static formulation and distinguish it from an independent mixture.',
 'The static formulation starts with identity A and C and zero process covariance. Observation noise can still be learned. HMM dependence remains enabled; we do not silently collapse transitions.'),
('02_dynamic_clustering', 'Evolving cluster representatives', 'dynamic', 'offline', 1, False,
 'Inspect the latest filtered and smoothed latent representatives of an LDS model.',
 'Cluster representatives may evolve across assigned observations. Each component has its own LDS state and parameter updates; the public API exposes state snapshots, not A/C histories.'),
('03_online_inference', 'Clustering one observation at a time', 'dynamic', 'online', 1, False,
 'Stream six beats and inspect each committed assignment, allocation event and latent snapshot.',
 'One update commits one observation through the existing online schedule. It is not equivalent to repeatedly refitting a batch; earlier cached state and labels may be reconsidered by that schedule.'),
('04_offline_inference', 'Repeated batch inference', 'dynamic', 'offline', 1, False,
 'Distinguish batch local/global iteration from streaming and save a reproducible recipe.',
 'All six observations are available together. Batch fitting revisits assignments and component estimates. We explicitly cap outer iterations at two; this is a teaching budget, not a convergence assertion.'),
('05_temporal_warping', 'Timing variation and GP alignment', 'dynamic', 'offline', 1, True,
 'Enable offline temporal alignment and visualize a fitted monotone warp through the public alignment component.',
 'Similar waveforms may have timing differences. The model uses its existing assignment-dependent alignment schedule. A separate pairwise demonstration exposes offsets without reading private model caches.'),
('06_multioutput', 'Two channels, shared assignments', 'dynamic', 'offline', 2, False,
 'Cluster both recorded channels and compare SNR-based with uniform output weighting.',
 'Both outputs share cluster assignments while retaining output-specific GP/LDS states. This is not a model with a full cross-output covariance. share_gp=False does not mean shared or unshared C.'),
('07_fast_online', 'The fast online approximation', 'dynamic', 'online_fast', 1, False,
 'Run the supported fast schedule and compare its result with ordinary online inference on the same observations.',
 'Fast online is a separate approximation with its own history/update behavior. It is not an optimized implementation guaranteed to match ordinary online or batch inference.'),
]

GUIDANCE = {
 '00_quickstart': ['Start here when learning the public fit/results workflow.', 'Use the small record to check installation and understand waveform shapes.', 'Inspect representatives before interpreting cluster labels.', 'Move to a specialized tutorial before applying the recipe to a full recording.'],
 '01_static_clustering': ['Use when within-cluster representatives can be treated as stationary.', 'Use as a baseline before paying for parameter evolution.', 'Retain HMM dependence if adjacent assignments should influence one another.', 'Choose dynamic modeling if drift matters; static does not freeze all uncertainty learning.'],
 '02_dynamic_clustering': ['Use when representative shapes evolve over the observation sequence.', 'Inspect filtered versus stored smoothed latent states, not just labels.', 'Provide enough repeated observations per cluster to inform LDS estimates.', 'Expect more estimation work and prior sensitivity than the static baseline.'],
 '03_online_inference': ['Use for sequential arrivals when the whole future record is unavailable.', 'Freeze preprocessing from an initial calibration observation or past-only window.', 'Inspect per-update allocations; the constructor-created first cluster is not a new birth.', 'Do not assume online results equal a later batch refit, or that an offline fit can be continued online.'],
 '04_offline_inference': ['Use when the complete analysis window is available.', 'Use repeated batch updates when earlier assignments may be reconsidered.', 'Distinguish iteration budgets from a validated convergence criterion.', 'Investigate the documented empty-cluster edge case before increasing this subset’s iteration cap.'],
 '05_temporal_warping': ['Use when timing variation should not define distinct waveform groups.', 'First inspect unaligned data to avoid aligning away meaningful differences.', 'Budget for optimization beyond the dynamic baseline; begin with few observations.', 'Interpret the public pairwise warp separately from the model’s unavailable internal warp history.'],
 '06_multioutput': ['Use when simultaneous channels should share one cluster identity.', 'Scale channels deliberately; weighting does not replace a units decision.', 'Compare uniform and SNR weighting when channels differ in quality.', 'Inspect every output; this model does not expose full cross-channel predictive covariance.'],
 '07_fast_online': ['Use only after understanding standard online inference.', 'Treat it as a distinct approximation and compare assignments/posteriors on your data.', 'Measure longer streams before claiming a speed advantage from this short example.', 'Keep alignment disabled and a single output in this supported streaming demonstration.'],
}


def markdown(text):
    return dict(cell_type='markdown', metadata={}, source=textwrap.dedent(text).strip() + '\n')


def code(text, worker=True):
    return dict(cell_type='code', metadata={}, execution_count=None, outputs=[],
                source=('%%hdpgpc\n' if worker else '') + textwrap.dedent(text).strip() + '\n')


def build(spec):
    name, title, dynamics, schedule, outputs, warp, objective, assumption = spec
    cells = [markdown(f'# {title}\n\n{objective}\n\n{assumption}'),
      markdown('''## Colab setup
      CPU only. Run all cells in order. No source edits are needed. Until the new
      package is published, upload `hdpgpc-colab.zip` when prompted; GitHub main
      currently contains the old API. Locally, open this notebook from the checkout
      after installing `.[tutorials]`. See the README for the bundle/build command.

      The setup reuses a supported local installation or installs the wheel in a
      private environment. On an unsupported host Python it downloads Python 3.11
      with uv. Dependencies need internet and may take several minutes on first use.
      The bundle contains only the small record 100, not the full dataset.

      `%%hdpgpc` executes ordinary Python in one persistent isolated worker and
      displays its figures here. This avoids changing Colab's preloaded NumPy/Torch
      or requiring a restart. Keep the prefix when adding a cell. Host-cell variables
      and worker variables are separate. Model code below uses public APIs only.
      '''), code(SETUP, worker=False), markdown('## Imports and reproducibility'), code(IMPORTS),
      markdown('''## Load and inspect the record
      We use six consecutive beats from the same region as `test_step.ipynb`, not
      the original 250-beat demonstration. The stored tensor is (beats, points,
      channels). Channel names and physical units are not encoded in this array;
      we call them channel 0 and channel 1 rather than inventing metadata.

      Every sixth point is retained for 15-point CPU fits. The grid is rescaled to
      0–4 and amplitudes use first-beat centering/scaling. These are explicit teaching
      choices: coarse subsampling can discard morphology and is not a production
      resampling recommendation. Hyperparameters below are in these scaled units.
      The annotation file has a different length, so it is not used as ground truth.
      '''), code(f'N_OUTPUTS = {outputs}\n' + DATA),
      markdown(f'''## Configure the model
      {assumption}

      The GP starts with length scale 0.5 (bounds 0.1–2), output scale 1,
      observation covariance scale 0.04 and process scale 0.03. For static models
      the static policy initializes zero process covariance despite the supplied
      positive legacy `ini_gamma`. Bounds and Bayesian learning are explicit below.
      These are small teaching settings, not package defaults or the old notebook's
      raw-amplitude output scale 300. HDP concentration preset `balanced` and
      `max_models=4` bound this small exploration; the latter is not a known class count.

      Two batch outer iterations avoid a known empty-cluster reallocation failure
      reached with three on this subset (see inventory). They still allow repeated
      local/global updates. Do not interpret this cap as convergence. Inner GP and
      warp optimizer behavior is unchanged. Online schedules do not use this cap.
      '''), code(f'''config = HDPGPCConfig(
    x_basis=grid, verbose=False,
    hdp=HDPConfig(M=1, hdp_hyp='balanced', hmm_switch=True, max_models=4),
    emission=GPEmissionConfig(ini_lengthscale=0.5, bound_lengthscale=(0.1, 2.0),
        ini_outputscale=1.0, ini_sigma=0.04, bound_sigma=(0.005, 0.08),
        inducing_points=False),
    dynamics=DynamicsConfig(model_type={dynamics!r}, ini_gamma=0.03,
        bound_gamma=(0.001, 0.1), annealing=False, bayesian_params=True,
        estimation_limit=None, reest_conditions=[1, 20, 5]),
    alignment=AlignmentConfig(policy={'gp' if warp else 'none'!r}, x_basis_warp=grid,
        noise_warp=0.01, bound_noise_warp=(0.001, 0.1), recursive_warp=False,
        warp_updating=False, method_compute_warp='greedy', mode_warp='rough'),
    inference=InferenceConfig(schedule={schedule!r}, warp={warp},
        it_limit={2 if schedule == 'offline' else None}, n_explore_steps=2,
        reestimate_initial_params=False),
    multi_output=MultiOutputConfig(n_outputs={outputs}, use_snr={outputs == 2},
        share_gp=False, reduce_outputs=False),
    numerics=numerics,
)
config.validate(for_build=True)
print(json.dumps(config.to_dict(), indent=2))'''),
      markdown('## Run inference\nLegacy progress prints are captured to keep the tutorial readable; inspect `progress.getvalue()` if needed. No exceptions are suppressed.')]
    run = '''model = HDPGPC(config)
progress = io.StringIO()
extra = dict(preprocessing=dict(start=start, n_beats=n_beats, stride=stride,
                               center=center.tolist(), scale=scale.tolist()))
started = time.perf_counter()
with redirect_stdout(progress):
'''
    if schedule == 'offline':
        run += '    model.fit(X)\n'
    else:
        run += '''    updates = []
    for waveform in X:
        result = model.update(waveform, include_prediction=True, include_posterior=True)
        updates.append(dict(cluster=result.assigned_cluster, new_cluster=result.new_cluster,
            active_clusters=model.n_clusters_, responsibilities=result.responsibilities.tolist(),
            assignment_probabilities=result.assignment_probabilities,
            filtered_mean=result.posterior[0].filtered_mean.tolist()))
extra['updates'] = updates
for index, update in enumerate(updates):
    print(index, {key: value for key, value in update.items() if key != 'filtered_mean'})
'''
    run += "inference_seconds = time.perf_counter() - started\nprint('Labels:', model.labels_, '; represented clusters:', model.n_clusters_)"
    cells += [code(run), markdown('''## Inspect and plot public results
      Cluster IDs are current indices, not stable biological labels. `n_clusters_`
      counts represented clusters rather than reserved empty slots. Soft posterior
      assignment probabilities are not retained by this API: online
      `assignment_probabilities` is `None`, and `responsibilities` is hard.

      `predictive()` returns the current observation distribution, not a future
      forecast. Shading is ±2 square roots of the returned covariance diagonal;
      at the native grid this is the implementation's observation Sigma, not a
      newly reconstructed full posterior uncertainty or calibrated clinical interval.
      `posterior()` copies the latest stored filtered/smoothed states. Querying it
      does not run another smoother. No A/C matrices or transition probabilities
      are exposed by the facade, so we do not reach into private state to plot them.
      '''), code("utils['plot_results'](model, X)\nplt.show()\nutils['plot_latent'](model)\nplt.show()")]
    if warp:
        cells += [markdown('''## Visualize a public pairwise warp
          The facade does not expose its per-observation warp cache. This separate
          public-component fit aligns the second real beat to the first; it is not
          claimed to be the exact warp chosen by batch clustering. No artificial
          shift is added. A small or identity warp is a valid result. This uses 50
          component optimization steps and does not alter the model's optimizer.
          '''), code('''import torch
from hdpgpc.alignment import GaussianProcessWarp
numerics.apply()
warper = GaussianProcessWarp(torch.from_numpy(grid), recursive=False)
warp_started = time.perf_counter()
offsets, aligned, score, losses = warper.compute_warp(
    torch.from_numpy(grid), torch.from_numpy(X[1]), torch.from_numpy(X[0]), train_iter=50)
print('Pairwise warp seconds:', time.perf_counter() - warp_started)
g = grid.ravel() + offsets.detach().numpy().ravel()
assert np.all(np.diff(g) >= 0), 'Expected monotone fitted grid'
fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), constrained_layout=True)
axes[0].plot(grid.ravel(), grid.ravel(), '--', label='Identity')
axes[0].plot(grid.ravel(), g, label='Fitted g(x)')
axes[0].set(xlabel='Model coordinate', ylabel='Warped coordinate', title='Pairwise warp')
axes[0].legend(frameon=False)
axes[1].plot(grid.ravel(), X[0, :, 0], label='Template: beat 1700')
axes[1].plot(grid.ravel(), X[1, :, 0], ':', label='Target: beat 1701')
axes[1].plot(grid.ravel(), aligned.detach().numpy().ravel(), label='Aligned target')
axes[1].set(xlabel='Model coordinate', ylabel='Scaled amplitude')
axes[1].legend(frameon=False)
plt.show()
extra['pairwise_warp'] = dict(offsets=offsets.tolist(), aligned=aligned.tolist(),
                             score=float(score), losses=losses)
''')]
    if outputs == 2 or schedule == 'online_fast':
        comparison = "replace(config, multi_output=replace(config.multi_output, use_snr=False))" if outputs == 2 else "replace(config, inference=replace(config.inference, schedule='online'))"
        comparison_run = "comparison.fit(X)" if outputs == 2 else "\n    for waveform in X:\n        comparison.update(waveform)"
        cells += [markdown('## A controlled comparison\nChange only output weighting.' if outputs == 2 else
                           '## Compare with standard online\nChange only the schedule. Similar labels on six beats do not establish equivalence or a speedup.'),
                  code(f'''comparison_config = {comparison}
comparison = HDPGPC(comparison_config)
comparison_started = time.perf_counter()
with redirect_stdout(io.StringIO()):
    {comparison_run}
print('Comparison seconds:', time.perf_counter() - comparison_started)
print('Main labels:', model.labels_, 'Comparison labels:', comparison.labels_)
extra['comparison'] = utils['snapshot'](comparison)
''')]
    changes = {
        'static': 'Static is the baseline. It retains GP posterior learning and HMM assignment dependence; it is not just averaging observations.',
        'dynamic': 'Compared with static clustering, model_type changes to dynamic, allowing evolving component states and parameter updates. Filtering/smoothing and LDS estimation add work; posterior snapshots show the latest latent estimates.',
    }
    change = changes[dynamics]
    if schedule != 'offline': change += ' Streaming replaces fit(X) with successive update(x) calls and exposes assignment/allocation events; it does not promise batch-equivalent labels.'
    if warp: change += ' Alignment additionally changes policy to gp and requests warp=True. It adds expensive optimization; six beats and 15 points keep this example short. Model representatives are in aligned coordinates.'
    if outputs == 2: change += ' A second output adds separate GP/LDS work and channel-specific plots. Both channels still share one label per beat; SNR weighting is a model choice, not preprocessing.'
    cells += [markdown('## What changed relative to the simplest baseline?\n\n' + change),
      markdown('## Save the reproducible experiment\nThe JSON contains the full configuration, record/input hashes, environment, labels and public posterior quantities. Rerunning replaces only this tutorial’s generated report.'),
      code(f"report = utils['save_report']({name!r}, model, X, config, versions, inference_seconds, extra)\nassert len(report['scientific_result']['labels']) == len(X)"),
      markdown('## When should I use this mode?\n\n' + '\n'.join('- ' + point for point in GUIDANCE[name])),
      markdown('''## Optional exercises (outside the reproducible main flow)
      - Copy the configuration with `dataclasses.replace` and compare `hdp_hyp='less'`
        or `'more'`; leave the data and seed fixed and report resulting labels.
      - Try a different length scale or an explicit supported scikit-learn GP kernel;
        adjust its units to the grid and retain positive covariance bounds.
      - Compare `hmm_switch=False` with True. This changes assignment dependence,
        independently of static versus dynamic component behavior.
      - For alignment, compare `policy='none', warp=False`; for two channels, inspect
        the uniform/SNR comparison. Do not reinterpret `share_gp` as a C-sharing switch.
      - Increase observations or grid resolution in a separate copy and measure runtime.
        The known three-pass empty-cluster failure must be investigated before simply
        raising the outer iteration cap on this subset.

      PCA, user transform hooks and shared-C/fixed-LDS presets are not currently
      implemented by this public API; no corresponding exercise is silently simulated.
      '''), code("plt.close('all')")]
    for i, cell in enumerate(cells): cell['id'] = f'cell-{i:02d}'
    return dict(cells=cells, nbformat=4, nbformat_minor=5,
                metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                              language_info=dict(name='python'), colab=dict(name=name+'.ipynb', provenance=[])))


def main():
    folder = ROOT / 'tutorials'
    folder.mkdir(exist_ok=True)
    for spec in SPECS:
        (folder / (spec[0] + '.ipynb')).write_text(json.dumps(build(spec), indent=1, ensure_ascii=False) + '\n', encoding='utf8')
    print('Generated', len(SPECS), 'tutorials')


if __name__ == '__main__':
    main()
