"""Calibration loss and reconstruction diagnostics for fixed observation maps."""
import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_autoencoder_calibration(values, network, indices, losses):
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[..., 0]
    selected = np.asarray(indices)[np.linspace(0, len(indices) - 1, min(4, len(indices)), dtype=int)]
    with torch.no_grad():
        decoded = network(torch.as_tensor(values[selected], dtype=torch.float64)).numpy()
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(np.arange(1, len(losses) + 1), losses)
    axes[0].set(xlabel='Epoch', ylabel='Scaled MSE', title='Autoencoder calibration')
    for index, y, fitted in zip(selected, values[selected], decoded):
        line, = axes[1].plot(y, alpha=.6, label=f'Sample {index}')
        axes[1].plot(fitted, '--', color=line.get_color())
    axes[1].set(xlabel='Waveform point', ylabel='Amplitude', title='Observed / dashed reconstruction')
    axes[1].legend(fontsize=8)
    return figure
