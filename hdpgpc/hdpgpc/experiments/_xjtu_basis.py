"""Channel-wise fixed Fourier/Haar observations for the shared XJTU workflow.

Rank selection uses only the balanced, unlabeled calibration windows. It keeps
coordinates of the existing orthonormal bases, without learning a projection.
Fourier sine/cosine pairs are indivisible so phase rotations stay in the basis.
"""
from dataclasses import dataclass

import numpy as np

from ..datasets.xjtu_sy import XJTU_SAMPLE_RATE
from ..emissions import AffineGaussianObservation, real_fourier_basis, real_wavelet_basis


@dataclass(frozen=True)
class BasisObservationEstimate:
    model: AffineGaussianObservation
    indices: np.ndarray
    candidate_energy: np.ndarray
    selected_columns: tuple
    coefficient_metadata: tuple
    explained_variance_ratio: float


def _candidate_basis(args):
    count = args.window_samples
    grid = np.arange(count, dtype=float)[:, None] / XJTU_SAMPLE_RATE
    groups, metadata = [], []
    if args.representation == 'wavelet':
        basis = real_wavelet_basis(grid, wavelet=args.wavelet)
        metadata.append(dict(kind='scaling', start_sample=0, length_samples=count))
        length = count
        while length >= 2:
            for start in range(0, count, length):
                metadata.append(dict(kind='detail', start_sample=start, length_samples=length))
            length //= 2
        groups = [(i,) for i in range(count)]
    else:
        window_size = count if args.window_size is None else args.window_size
        basis = real_fourier_basis(grid, window_size=window_size,
            n_harmonics=args.harmonics, frequency_bins=args.frequency_bins)
        highest = min(count, window_size) // 2
        bins = (sorted(args.frequency_bins) if args.frequency_bins is not None else
                range((args.harmonics if args.harmonics is not None else highest) + 1))
        for start in range(0, count, window_size):
            length = min(window_size, count - start)
            for frequency_bin in bins:
                if frequency_bin > length // 2:
                    continue
                kinds = (('dc',) if frequency_bin == 0 else ('nyquist',)
                         if length % 2 == 0 and frequency_bin == length // 2 else ('cos', 'sin'))
                columns = []
                for kind in kinds:
                    columns.append(len(metadata))
                    metadata.append(dict(kind=kind, start_sample=start, length_samples=length,
                        frequency_bin=int(frequency_bin), frequency_hz=frequency_bin*XJTU_SAMPLE_RATE/length))
                groups.append(tuple(columns))
    if len(metadata) != basis.shape[1]:
        raise RuntimeError('Basis metadata does not match the native decoder columns')
    return basis, groups, metadata


def _select_columns(energy, groups, rank):
    """Maximum calibration energy at an exact rank, retaining whole groups."""
    if rank == len(energy):
        return np.arange(rank, dtype=int)
    scores = np.full((len(groups)+1, rank+1), -np.inf)
    scores[0, 0] = 0.
    take = np.zeros(scores.shape, dtype=bool)
    for i, group in enumerate(groups):
        cost = len(group)
        scores[i+1] = scores[i]
        if cost <= rank:
            candidate = scores[i, :rank-cost+1] + energy[list(group)].sum()
            improved = candidate > scores[i+1, cost:]
            scores[i+1, cost:][improved] = candidate[improved]
            take[i+1, cost:][improved] = True
    if not np.isfinite(scores[-1, rank]):
        raise ValueError('Requested --components cannot retain whole Fourier pairs in each channel; '
                         'change the rank or include DC/Nyquist among the candidate bins')
    columns, remaining = [], rank
    for i in range(len(groups), 0, -1):
        if take[i, remaining]:
            columns.extend(groups[i-1])
            remaining -= len(groups[i-1])
    return np.asarray(sorted(columns), dtype=int)


def validate_basis_options(args):
    if args.components < 2 or args.components > 2*args.window_samples:
        raise ValueError('Channel-wise bases require 2 <= --components <= 2*--window-samples')
    basis, groups, _ = _candidate_basis(args)
    for rank in ((args.components+1)//2, args.components//2):
        if rank > basis.shape[1]:
            raise ValueError('Requested --components exceeds the available basis columns')
        _select_columns(np.zeros(basis.shape[1]), groups, rank)


def estimate_basis_observation(batch, args):
    """Calibrate offsets/noise and retain basis coordinates, separately per sensor."""
    values = np.asarray(batch, dtype=float)
    if (values.ndim != 2 or values.shape[1] != 2*args.window_samples
            or len(values) < 2 or not np.isfinite(values).all()):
        raise ValueError('Basis calibration needs at least two finite two-channel raw windows')
    basis, groups, metadata = _candidate_basis(args)
    offset = values.mean(0)
    centered = values-offset
    decoder = np.zeros((values.shape[1], args.components))
    selected, energies, coefficients = [], [], []
    column = 0
    for channel, rank in enumerate(((args.components+1)//2, args.components//2)):
        codes = centered[:, channel::2] @ basis
        energy = np.mean(codes**2, axis=0)
        chosen = _select_columns(energy, groups, rank)
        decoder[channel::2, column:column+rank] = basis[:, chosen]
        selected.append(chosen)
        energies.append(energy)
        for index in chosen:
            coefficients.append(dict(channel=channel, channel_name=('horizontal', 'vertical')[channel],
                basis_column=int(index), **metadata[index]))
        column += rank
    codes = centered @ decoder
    residual = centered-codes @ decoder.T
    covariance = np.atleast_2d(np.cov(codes, rowvar=False))
    ridge = max(1e-8, 1e-5*float(np.trace(covariance))/args.components)
    covariance += ridge*np.eye(args.components)
    floor = max(args.noise_floor, args.noise_floor_fraction*float(np.var(values)))
    noise = np.diag(np.maximum(residual.var(axis=0, ddof=1), floor))
    variance = float(np.sum(centered**2))
    explained = 1-float(np.sum(residual**2))/variance if variance else 0.
    model = AffineGaussianObservation(decoder, noise,
        np.arange(values.shape[1], dtype=float)[:, None], offset, covariance)
    return BasisObservationEstimate(model, np.arange(len(values)), np.stack(energies),
                                    tuple(selected), tuple(coefficients), explained)
