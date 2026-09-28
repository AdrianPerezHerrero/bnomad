"""Data, public-result plotting and reporting shared by the user tutorials.

No inference, parameter selection or access to private model state lives here.
"""
from pathlib import Path
import hashlib
import json
import platform
from importlib import metadata
from urllib.request import urlopen

import numpy as np
import matplotlib.pyplot as plt

DATA_REVISION = '615109a9e474d37626abaff6c51f1c159b26124a'
DATA_HASHES = {
    '100.npy': 'cf3c01ac96a1ae9683122a4af57f71c44d3e89eec090ed9d53c502b8ff466b89',
    '100_labels.npy': 'ea7161896ee6a6ce762bda26ae7aa9e9a4debf9543ced3f4948e18f0c3df9917',
}


def load_record(record_name="100"):
    """Load the specified MIT-BIH record; fetch at most 3.3 MB if no local copy is present."""
    data_file = f"{record_name}.npy"
    labels_file = f"{record_name}_labels.npy"
    required_files = [data_file, labels_file]
    
    candidates = [p / 'hdpgpc/data/mitbih' for p in
                  (Path(__file__).resolve().parents[1], Path.cwd(), *Path.cwd().parents)]
    directory = next((p for p in candidates if (p / data_file).exists()),
                     Path.cwd() / '.tutorial-data')
    directory.mkdir(parents=True, exist_ok=True)
    
    for name in required_files:
        expected = DATA_HASHES.get(name)
        path = directory / name
        if not path.exists():
            url = f'https://raw.githubusercontent.com/AdrianPerezHerrero/HDP-GPC/{DATA_REVISION}/hdpgpc/data/mitbih/{name}'
            print(f'Downloading {name} from the pinned repository record')
            with urlopen(url, timeout=60) as response:
                payload = response.read(3_400_001)
            if len(payload) > 3_400_000:
                raise ValueError('Record download size exceeded')
            if expected and hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError('Record checksum mismatch')
            path.write_bytes(payload)
        if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Unexpected record bytes: {path}')
            
    data = np.load(directory / data_file, allow_pickle=False)
    annotations = np.load(directory / labels_file, allow_pickle=False)
    
    print('Waveforms:', data.shape, data.dtype, '; annotations:', annotations.shape)
    print('Annotation lengths differ: no ground-truth pairing or accuracy claim.')
    return data, annotations


def environment():
    result = {'python': platform.python_version(), 'platform': platform.platform(),
              'versions': {n: metadata.version(n) for n in
                           ('hdpgpc', 'numpy', 'scipy', 'torch', 'gpytorch', 'linear-operator', 'pyro-ppl', 'scikit-learn')}}
    print(json.dumps(result, indent=2))
    return result


def plot_input(raw, grid, values):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), constrained_layout=True)
    for i in range(min(3, len(raw))):
        axes[0].plot(np.arange(raw.shape[1]), raw[i, :, 0], label=f'Beat {1700+i}')
        axes[1].plot(grid.ravel(), values[i, :, 0], marker='.', label=f'Beat {1700+i}')
    axes[0].set(xlabel='Original point index', ylabel='Stored amplitude', title='Record channel 0')
    axes[1].set(xlabel='Rescaled coordinate (0–4)', ylabel='Scaled amplitude', title='15-point teaching input')
    axes[0].legend(frameon=False)
    return fig


def plot_results(model, values):
    """Current observation distributions; shading uses the returned covariance."""
    count = values.shape[2]
    fig, axes = plt.subplots(1, count + 1, figsize=(5 * (count + 1), 3.5), constrained_layout=True)
    axes[0].step(np.arange(len(values)), model.labels_, where='mid', marker='o')
    axes[0].set(xlabel='Observation in subset', ylabel='Cluster index', title='Current hard assignments')
    axes[0].set_yticks(model.cluster_ids_)
    for cluster in model.cluster_ids_:
        prediction = model.predictive(int(cluster))
        for output in range(count):
            axis = axes[output + 1]
            x = prediction.grid.ravel()
            mean = prediction.mean[:, output]
            # Clipping protects plotting only; never alter returned covariances.
            sd = np.sqrt(np.maximum(np.diag(prediction.covariance[output]), 0))
            line, = axis.plot(x, mean, label=f'Cluster {cluster}')
            axis.fill_between(x, mean - 2 * sd, mean + 2 * sd, color=line.get_color(), alpha=.18)
            axis.set(xlabel='Rescaled coordinate', ylabel='Scaled amplitude', title=f'Channel {output}: representatives')
            axis.legend(frameon=False)
    return fig


def plot_latent(model):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), constrained_layout=True)
    for cluster in model.cluster_ids_:
        for state in model.posterior(int(cluster)):
            label = f'C{cluster}, channel {state.output}, t={state.observation_index}'
            axes[0].plot(state.grid.ravel(), state.filtered_mean, label=label)
            axes[1].plot(state.grid.ravel(), state.smoothed_mean, label=label)
    for axis, title in zip(axes, ('Latest stored filtered mean', 'Latest stored smoothed mean')):
        axis.set(xlabel='Rescaled coordinate', ylabel='Latent mean', title=title)
        axis.legend(frameon=False, fontsize=8)
    return fig


def snapshot(model):
    """JSON-ready public posterior quantities, excluding timing/environment."""
    components = []
    for c in model.cluster_ids_:
        p = model.predictive(int(c))
        states = [dict(output=s.output, observation_index=s.observation_index,
                       filtered_mean=s.filtered_mean.tolist(), filtered_covariance=s.filtered_covariance.tolist(),
                       smoothed_mean=s.smoothed_mean.tolist(), smoothed_covariance=s.smoothed_covariance.tolist())
                  for s in model.posterior(int(c))]
        components.append(dict(cluster=int(c), mean=p.mean.tolist(), covariance=p.covariance.tolist(), latent=states))
    return dict(labels=model.labels_.tolist(), n_clusters=model.n_clusters_, components=components)


def save_report(name, model, values, config, versions, seconds, extra=None, record_name="100"):
    report = dict(config=config.to_dict(), environment=versions, inference_seconds=seconds,
                  data_revision=DATA_REVISION, record_sha256=DATA_HASHES.get(f'{record_name}.npy', 'unknown_hash'),
                  input_sha256=hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest(),
                  scientific_result=snapshot(model), extra=extra or {})
    path = Path('tutorial-results') / (name + '.json')
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print('Saved', path, '; inference seconds:', round(seconds, 3))
    return report