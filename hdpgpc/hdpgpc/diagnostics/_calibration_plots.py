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


def plot_linear_prediction_calibration(values, estimate, *, image_shape=None):
    """Plot loss history and target reconstruction/prediction on calibration pairs.

    Uses the saved calibration B and raw encoder, not a cluster LDS posterior.
    This is an in-sample calibration diagnostic, not a held-out forecast score.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    selected = np.linspace(0, len(estimate.pair_indices) - 1,
                           min(4, len(estimate.pair_indices)), dtype=int)
    pairs = estimate.pair_indices[selected]
    source, target = values[pairs[:, 0]], values[pairs[:, 1]]
    decoder = estimate.model
    reconstructed = decoder.offset + estimate.transform(target) @ decoder.C.T
    predicted = decoder.offset + estimate.transform(source) @ estimate.predictor.T @ decoder.C.T

    def losses_on(axis):
        steps = np.arange(len(estimate.losses))
        axis.plot(steps, estimate.losses[:, 0], label='Total objective')
        axis.plot(steps, estimate.losses[:, 1], label='Reconstruction / image variance')
        axis.plot(steps, estimate.losses[:, 2] + estimate.losses[:, 3],
                  label='Prediction + ridge (unweighted)')
        axis.axvline(estimate.best_iteration, color='0.4', linestyle='--', label='Selected basis')
        axis.set(title=f'Calibration | prediction weight={estimate.prediction_weight:g}',
                 xlabel='Optimizer step (0 = PCA control)', ylabel='Normalized loss')
        axis.legend(fontsize=7)

    if image_shape is not None:
        shape = tuple(image_shape)
        if len(shape) != 2 or int(np.prod(shape)) != values.shape[1]:
            raise ValueError('Image shape must match flattened calibration observations')
        figure = plt.figure(figsize=(5 + 2 * len(pairs), 7), constrained_layout=True)
        layout = figure.add_gridspec(4, len(pairs) + 1, width_ratios=[2.5] + [1] * len(pairs))
        losses_on(figure.add_subplot(layout[:, 0]))
        for row, (name, images) in enumerate((('Source', source), ('Observed target', target),
                    ('Target reconstruction', reconstructed), ('B prediction', predicted))):
            for column, image in enumerate(images, 1):
                axis = figure.add_subplot(layout[row, column])
                axis.imshow(np.clip(image.reshape(shape), 0., 1.), cmap='gray', vmin=0, vmax=1)
                axis.set_xticks([])
                axis.set_yticks([])
                if column == 1:
                    axis.set_ylabel(name)
                if row == 0:
                    axis.set_title(f'Frames {pairs[column - 1, 0]} -> {pairs[column - 1, 1]}')
    else:
        figure, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
        losses_on(axes[0])
        for index, pair in enumerate(pairs):
            color = plt.get_cmap('tab10')(index)
            for axis, fitted in ((axes[1], reconstructed), (axes[2], predicted)):
                axis.plot(target[index], color=color, alpha=.6, label=f'{pair[0]} -> {pair[1]}')
                axis.plot(fitted[index], color=color, linestyle='--')
        for axis, title in ((axes[1], 'Target / dashed reconstruction'),
                            (axes[2], 'Target / dashed B prediction')):
            axis.set(title=title, xlabel='Observation coordinate', ylabel='Amplitude')
            axis.legend(fontsize=7)
    figure.suptitle('Fixed linear prediction representation | calibration pairs, not held-out forecasts')
    return figure
