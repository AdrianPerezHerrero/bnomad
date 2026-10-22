"""Tied linear autoencoder calibration for exact fixed affine observations."""
from dataclasses import dataclass
import numpy as np
import torch
from ._fixed_observation import AffineGaussianObservation

class TiedLinearAutoencoder(torch.nn.Module):
    """Learn an orthonormal latent subspace with an affine Gaussian decoder.

    Q has orthonormal columns. Encoder: z = (y - offset) Q / scale.
    Decoder: y_hat = offset + scale Q z. This ties the two maps and lets the
    Gaussian LDS use exactly the same latent coordinates as the encoder.
    """

    def __init__(self, points: int, components: int, mean, scale: float):
        super().__init__()
        self.raw_basis = torch.nn.Parameter(torch.randn(points, components, dtype=torch.float64))
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float64).reshape(1, -1))
        self.scale = float(scale)

    def basis(self):
        return torch.linalg.qr(self.raw_basis, mode='reduced').Q

    def encode(self, waveform):
        return ((waveform - self.mean) / self.scale) @ self.basis()

    def decode(self, latent):
        return self.mean + self.scale * (latent @ self.basis().T)

    def forward(self, waveform):
        basis = self.basis()
        latent = ((waveform - self.mean) / self.scale) @ basis
        return self.mean + self.scale * (latent @ basis.T)


@dataclass(frozen=True)
class AutoencoderObservationEstimate:
    model: AffineGaussianObservation
    network: TiedLinearAutoencoder
    indices: np.ndarray
    losses: np.ndarray
    residual: np.ndarray


def estimate_autoencoder_observation(observations, grid, *, components=12,
                                      max_samples=400, epochs=60, learning_rate=0.02,
                                      random_state=42, sampling='uniform',
                                      noise_structure='isotropic', noise_floor=1e-8,
                                      noise_floor_fraction=0.05,
                                      process_noise_scale=1., observation_noise_scale=1.):
    """Calibrate the existing tied linear AE and freeze its affine decoder.

    Calibration data must be supplied explicitly (TRAIN for UCR, or a prefix
    for ordered online records). No labels or clustering posteriors are used.
    Covariance arguments are variances; the returned model uses the same exact
    Gaussian observation interface as PCA/Fourier/wavelets.
    """
    values = np.asarray(observations, dtype=np.float64)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2 or not np.isfinite(values).all() or len(values) < 2:
        raise ValueError('Autoencoder calibration needs finite waveform rows and at least two samples')
    if (not isinstance(components, int) or components < 1 or
            not isinstance(max_samples, int) or max_samples < 2 or
            not isinstance(epochs, int) or epochs < 1):
        raise ValueError('Invalid autoencoder dimension, calibration size or epochs')
    if sampling not in ('uniform', 'prefix') or noise_structure not in ('isotropic', 'diagonal'):
        raise ValueError('Unsupported calibration sampling or noise structure')
    if any(not np.isfinite(x) or x <= 0 for x in
           (learning_rate, noise_floor, process_noise_scale, observation_noise_scale)):
        raise ValueError('Learning rate and variance scales must be positive and finite')
    if not np.isfinite(noise_floor_fraction) or noise_floor_fraction < 0:
        raise ValueError('noise_floor_fraction must be finite and nonnegative')
    count = min(max_samples, len(values))
    dimension = min(components, values.shape[1] - 1, count - 1)
    if dimension < 1:
        raise ValueError('Autoencoder requires at least two waveform points')
    indices = (np.arange(count) if sampling == 'prefix' else np.sort(
        np.random.default_rng(random_state).choice(len(values), count, replace=False)))
    batch = values[indices]
    mean = batch.mean(axis=0)
    scale = float(np.sqrt(np.mean((batch - mean) ** 2)))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError('Autoencoder calibration has zero variation')
    torch.manual_seed(random_state)
    network = TiedLinearAutoencoder(values.shape[1], dimension, mean, scale)
    training = torch.as_tensor(batch, dtype=torch.float64)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = (((network(training) - training) / scale) ** 2).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    network.eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        basis = network.basis().numpy()
        codes = network.encode(training).numpy()
        residual = batch - network(training).numpy()
    covariance = np.atleast_2d(np.cov(codes, rowvar=False))
    ridge = max(1e-8, 1e-5 * float(np.trace(covariance)) / dimension)
    covariance = (covariance + ridge * np.eye(dimension)) * process_noise_scale
    floor = max(noise_floor, noise_floor_fraction * float(np.var(batch)))
    sigma = (max(float(np.mean(residual ** 2)), floor) * np.eye(values.shape[1])
             if noise_structure == 'isotropic' else
             np.diag(np.maximum(residual.var(axis=0, ddof=1), floor)))
    decoder = AffineGaussianObservation(
        C=scale * basis, Sigma=observation_noise_scale * sigma,
        observation_grid=np.asarray(grid), offset=mean, latent_covariance=covariance)
    return AutoencoderObservationEstimate(decoder, network, indices,
                                          np.asarray(losses), residual)
