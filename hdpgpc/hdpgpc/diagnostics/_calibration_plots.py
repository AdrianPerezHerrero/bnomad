"""Calibration loss and reconstruction diagnostics for fixed observation maps."""
import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_autoencoder_calibration(values, network, indices, losses, *, image_shape=None):
    """Show training loss and reconstructions, optionally as grayscale images."""
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[..., 0]
    selected = np.asarray(indices)[np.linspace(0, len(indices) - 1, min(4, len(indices)), dtype=int)]
    with torch.no_grad():
        decoded = network(torch.as_tensor(values[selected], dtype=torch.float64)).numpy()
    if image_shape is not None:
        shape = tuple(image_shape)
        if len(shape) != 2 or int(np.prod(shape)) != values.shape[1]:
            raise ValueError('Image shape must match flattened autoencoder observations')
        figure = plt.figure(figsize=(4 + 2 * len(selected), 5), constrained_layout=True)
        layout = figure.add_gridspec(2, len(selected) + 1, width_ratios=[2] + [1] * len(selected))
        loss_axis = figure.add_subplot(layout[:, 0])
        loss_axis.plot(np.arange(1, len(losses) + 1), losses)
        loss_axis.set(xlabel='Epoch', ylabel='Scaled MSE', title='Autoencoder calibration')
        for column, (index, fitted) in enumerate(zip(selected, decoded), 1):
            for row, (name, frame) in enumerate((('Observed', values[index]), ('Reconstructed', fitted))):
                axis = figure.add_subplot(layout[row, column])
                axis.imshow(np.clip(frame.reshape(shape), 0., 1.), cmap='gray', vmin=0, vmax=1)
                axis.set_xticks([])
                axis.set_yticks([])
                if column == 1:
                    axis.set_ylabel(name)
                if row == 0:
                    axis.set_title(f'Frame {index}')
        return figure
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(np.arange(1, len(losses) + 1), losses)
    axes[0].set(xlabel='Epoch', ylabel='Scaled MSE', title='Autoencoder calibration')
    for index, y, fitted in zip(selected, values[selected], decoded):
        line, = axes[1].plot(y, alpha=.6, label=f'Sample {index}')
        axes[1].plot(fitted, '--', color=line.get_color())
    axes[1].set(xlabel='Waveform point', ylabel='Amplitude', title='Observed / dashed reconstruction')
    axes[1].legend(fontsize=8)
    return figure
