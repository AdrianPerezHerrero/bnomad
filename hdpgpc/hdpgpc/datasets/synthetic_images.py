"""Toy physical fields and tissue images with known linear Gaussian dynamics.

A shared affine decoder supports different persistent regime appearances and
temporal evolution. The original equal-appearance experiment is also available.
No clipping enters either generative model.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class SwitchingLDSImages:
    images: np.ndarray
    clean_images: np.ndarray
    states: np.ndarray
    labels: np.ndarray
    segment_ids: np.ndarray
    A: np.ndarray
    Gamma: np.ndarray
    C: np.ndarray
    Sigma_diagonal: np.ndarray
    offset: np.ndarray
    initial_covariance: np.ndarray
    previous_indices: np.ndarray
    regime_names: tuple[str, ...]
    state_mode: str
    seed: int
    scenario: str = 'dynamics_only'
    initial_means: np.ndarray | None = None
    initial_covariances: np.ndarray | None = None

    @property
    def values(self):
        """Package input: (frames, flattened pixels, one output)."""
        return self.images.reshape(len(self.images), -1, 1)

    @property
    def grid(self):
        """Pixel indices for the affine interface, not physical 1D distances."""
        return np.arange(self.C.shape[0], dtype=np.float64).reshape(-1, 1)

    @property
    def initial_state_means(self):
        return (np.zeros((len(self.regime_names), self.C.shape[1])) if self.initial_means is None
                else self.initial_means)

    @property
    def initial_state_covariances(self):
        return (np.repeat(self.initial_covariance[None], len(self.regime_names), axis=0)
                if self.initial_covariances is None else self.initial_covariances)

    @property
    def regime_mean_images(self):
        """Nominal appearances before zero-mean evolution/observation noise."""
        return (self.offset + self.initial_state_means @ self.C.T).reshape(
            len(self.regime_names), *self.images.shape[1:])

    def observation_model(self):
        """Return the existing exact fixed-affine observation implementation.

        This is an oracle decoder for controlled experiments, not a learned PCA.
        Constructing it materializes a full pixel covariance matrix.
        """
        from ..emissions import AffineGaussianObservation
        return AffineGaussianObservation(
            C=self.C, Sigma=np.diag(self.Sigma_diagonal),
            observation_grid=self.grid, offset=self.offset,
            latent_covariance=self.initial_covariance)


def load_switching_lds_images(path):
    """Load the generator's dataset.npz, including its ground truth (no pickle).

    PNG/GIF previews must not be used as inputs: they are quantized and scaled.
    Validate the array contract before exposing images to an experiment.
    """
    with np.load(path, allow_pickle=False) as archive:
        data = SwitchingLDSImages(
            images=archive['images'].copy(), clean_images=archive['clean_images'].copy(),
            states=archive['latent_states'].copy(), labels=archive['labels'].copy(),
            segment_ids=archive['segment_ids'].copy(), A=archive['A'].copy(),
            Gamma=archive['Gamma'].copy(), C=archive['C'].copy(),
            Sigma_diagonal=archive['Sigma_diagonal'].copy(), offset=archive['offset'].copy(),
            initial_covariance=archive['initial_covariance'].copy(),
            previous_indices=archive['previous_indices'].copy(),
            regime_names=tuple(archive['regime_names'].astype(str).tolist()),
            state_mode=str(archive['state_mode'].item()), seed=int(archive['seed'].item()),
            scenario=str(archive['scenario'].item()) if 'scenario' in archive else 'dynamics_only',
            initial_means=archive['initial_state_means'].copy() if 'initial_state_means' in archive else None,
            initial_covariances=archive['initial_state_covariances'].copy() if 'initial_state_covariances' in archive else None)
    n = len(data.images)
    if data.images.ndim != 3 or n < 2 or data.states.ndim != 2:
        raise ValueError('Synthetic images require at least two frames and a state matrix')
    pixels, dimension, regimes = int(np.prod(data.images.shape[1:])), data.states.shape[1], len(data.regime_names)
    if (data.clean_images.shape != data.images.shape or dimension < 1 or regimes < 1
            or data.states.shape[0] != n or data.labels.shape != (n,)
            or data.segment_ids.shape != (n,) or data.previous_indices.shape != (n,)
            or data.C.shape != (pixels, dimension) or data.offset.shape != (pixels,)
            or data.Sigma_diagonal.shape != (pixels,)
            or data.A.shape != (regimes, dimension, dimension)
            or data.Gamma.shape != data.A.shape
            or data.initial_covariance.shape != (dimension, dimension)
            or data.initial_state_means.shape != (regimes, dimension)
            or data.initial_state_covariances.shape != (regimes, dimension, dimension)):
        raise ValueError('Synthetic dataset dimensions disagree')
    for value in (data.images, data.clean_images, data.states, data.A, data.Gamma,
                  data.C, data.Sigma_diagonal, data.offset, data.initial_covariance,
                  data.initial_state_means, data.initial_state_covariances):
        if not np.isfinite(value).all():
            raise ValueError('Synthetic dataset contains nonfinite values')
    for value in (data.labels, data.segment_ids, data.previous_indices):
        if not np.issubdtype(value.dtype, np.integer):
            raise ValueError('Synthetic labels and indices must be integer arrays')
    if (np.any(data.labels < 0) or np.any(data.labels >= regimes)
            or np.any(data.segment_ids < 0) or np.any(np.diff(data.segment_ids) < 0)
            or np.any(data.Sigma_diagonal <= 0)
            or data.state_mode not in ('switching', 'per_regime')
            or data.scenario not in ('dynamics_only', 'appearance_dynamics', 'physical_fields')
            or (data.scenario != 'dynamics_only' and data.state_mode != 'per_regime')):
        raise ValueError('Invalid synthetic labels, noise variances or state mode')
    for covariance in (*data.Gamma, data.initial_covariance, *data.initial_state_covariances):
        if not np.allclose(covariance, covariance.T) or np.linalg.eigvalsh(covariance).min() <= 0:
            raise ValueError('Synthetic state covariances must be positive definite')
    last = {}
    for t, k in enumerate(data.labels):
        expected = t - 1 if data.state_mode == 'switching' else last.get(int(k), -1)
        if data.previous_indices[t] != expected:
            raise ValueError('Synthetic previous-state indices disagree with the declared state mode')
        last[int(k)] = t
    if not np.allclose(data.clean_images.reshape(n, -1), data.offset + data.states @ data.C.T):
        raise ValueError('Synthetic clean images disagree with the saved affine decoder')
    return data


def _generate_dynamics_only(*, image_size=(32, 32),
                                  regime_order=(0, 1, 2, 0, 2, 1),
                                  segment_lengths=120, rho=0.995,
                                  contrast=0.12, observation_std=0.01,
                                  state_mode='switching', seed=42, rotation_rates=None):
    """Generate four-dimensional latent oscillators observed as grayscale images.

    ``switching`` is a conventional SLDS: z[t] = A[label[t]] z[t-1] + e[t].
    ``per_regime`` resumes the last state of the selected regime; its first
    occurrence draws from N(0,I). This matches the package's per-cluster state
    sequence, whose time step counts successive assigned observations.

    Each A consists of two damped 2D rotation blocks. Gamma=(1-rho**2) I
    makes N(0,I) stationary for every regime, including arbitrary predetermined
    switches. Thus every regime has the same marginal image distribution, while
    lagged covariances differ. Known scheduled labels are not drawn from an HMM.

    contrast is the largest latent-induced marginal pixel standard deviation;
    observation_std is a standard deviation, squared exactly once for Sigma.
    segment_lengths can be an integer or one positive integer per segment.
    All returned arrays are float64 except index/label arrays.
    """
    if (len(image_size) != 2 or any(not isinstance(n, Integral) or isinstance(n, bool)
                                  or n < 8 for n in image_size)):
        raise ValueError('image_size requires two integer dimensions >= 8')
    if state_mode not in ('switching', 'per_regime'):
        raise ValueError('state_mode must be switching or per_regime')
    if (not np.isfinite([rho, contrast, observation_std]).all()
            or not 0 < rho < 1 or contrast <= 0 or observation_std <= 0):
        raise ValueError('Require 0 < rho < 1 and positive finite contrast/observation_std')
    order = tuple(regime_order)
    if not order or any(not isinstance(k, Integral) or isinstance(k, bool)
                        or k not in (0, 1, 2) for k in order):
        raise ValueError('regime_order requires a nonempty sequence of regime IDs 0, 1, 2')
    lengths = ((segment_lengths,) * len(order) if isinstance(segment_lengths, Integral)
               else tuple(segment_lengths))
    if len(lengths) != len(order) or any(not isinstance(n, Integral) or isinstance(n, bool)
                                       or n <= 0 for n in lengths):
        raise ValueError('segment_lengths requires positive integers, one per segment')
    height, width = image_size
    yy, xx = np.meshgrid(np.linspace(-1, 1, height), np.linspace(-1, 1, width), indexing='ij')
    # Localized quadrature pairs: phase evolution moves light/dark bands inside
    # two patches. This is exactly linear in the four coefficients, unlike
    # translating or rotating a rasterized object in pixel space.
    left = np.exp(-((xx + .43)**2 / .15 + yy**2 / .55))
    right = np.exp(-((xx - .43)**2 / .15 + yy**2 / .55))
    templates = np.stack((left * np.cos(4 * yy), left * np.sin(4 * yy),
                          right * np.cos(4 * xx + 3 * yy),
                          right * np.sin(4 * xx + 3 * yy)), axis=-1).reshape(-1, 4)
    basis, triangular = np.linalg.qr(templates)
    basis *= np.where(np.diag(triangular) < 0, -1., 1.)
    C = basis * contrast / np.sqrt(np.max(np.sum(basis**2, axis=1)))
    rates = ((.10, .17), (.42, .63), (-.20, -.31)) if rotation_rates is None else rotation_rates
    A = np.zeros((3, 4, 4))
    for k, pair in enumerate(rates):
        for block, angle in enumerate(pair):
            rotation = np.array(((np.cos(angle), -np.sin(angle)),
                                 (np.sin(angle), np.cos(angle))))
            A[k, 2*block:2*block+2, 2*block:2*block+2] = rho * rotation
    Gamma = np.tile((1 - rho**2) * np.eye(4), (3, 1, 1))
    segment_ids = np.repeat(np.arange(len(order)), lengths)
    labels = np.repeat(np.asarray(order, dtype=np.int64), lengths)
    rng = np.random.default_rng(seed)
    states = np.empty((len(labels), 4))
    previous_indices = np.full(len(labels), -1, dtype=np.int64)
    last = {}
    process_std = np.sqrt(1 - rho**2)
    for t, k in enumerate(labels):
        previous = t - 1 if state_mode == 'switching' else last.get(int(k), -1)
        previous_indices[t] = previous
        states[t] = (rng.standard_normal(4) if previous < 0 else
                     A[k] @ states[previous] + process_std * rng.standard_normal(4))
        last[int(k)] = t
    offset = np.full(height * width, .5)
    clean = offset + states @ C.T
    noisy = clean + observation_std * rng.standard_normal(clean.shape)
    return SwitchingLDSImages(
        images=noisy.reshape(-1, height, width), clean_images=clean.reshape(-1, height, width),
        states=states, labels=labels, segment_ids=segment_ids, A=A, Gamma=Gamma, C=C,
        Sigma_diagonal=np.full(height * width, observation_std**2), offset=offset,
        initial_covariance=np.eye(4), previous_indices=previous_indices,
        regime_names=('slow', 'fast', 'reverse'), state_mode=state_mode, seed=int(seed))


def generate_switching_lds_images(*, image_size=(32, 32),
                                  regime_order=(0, 1, 2, 0, 2, 1),
                                  segment_lengths=120, rho=.995, contrast=None,
                                  observation_std=.01, state_mode='per_regime', seed=42,
                                  scenario='physical_fields', appearance_strength=1.,
                                  appearance_drift=None):
    """Generate overlapping physical fields (default), tissue or dynamics-only data.

    physical_fields uses a common broad background, weak gradient anomalies,
    plane-wave and localized rotating-dipole modes. Regimes have moderate speed
    and direction differences, with overlapping fluctuation amplitudes. Initial
    appearance SD=.55 versus mean coefficient norm .65, and appearance drift
    SD=.04, prevent nearly deterministic regime templates. This is a Gaussian
    field illustration, not a calibrated climate/PDE simulator.

    appearance_dynamics has six coordinates: two persistent appearance
    coefficients, then two 2D oscillators. A_k = blockdiag(I_2, rotations_k).
    Distinct initial means identify folds, rings and patches; small Gaussian
    random-walk noise lets their appearance drift slowly. No transition bias or
    nonlinear pixel warp is introduced. This case requires per_regime histories.

    All regimes share y=offset+C*z+noise. They have different initial latent
    means/covariances, oscillator speeds/directions and activity amplitudes.
    The two appearance coordinates are random walks, not stationary processes.
    The common initial_covariance is a pooled initialization for model fitting;
    the actual generator distributions are in initial_state_means/covariances.

    contrast controls maximum oscillator-induced pixel SD before the activity
    multipliers (which are <=1), default .06 for fields, .05 for tissue and .12
    for dynamics_only. appearance_strength scales nominal pattern differences.
    appearance_drift is
    a latent random-walk standard deviation, squared once in Gamma. All raw
    images remain unclipped. dynamics_only retains the original four-state
    algorithm and can also use state_mode='switching'.
    """
    if scenario not in ('physical_fields', 'appearance_dynamics', 'dynamics_only'):
        raise ValueError('scenario must be physical_fields, appearance_dynamics or dynamics_only')
    if scenario != 'dynamics_only' and state_mode != 'per_regime':
        raise ValueError(f'{scenario} requires state_mode=per_regime; use dynamics_only for a global switching state')
    appearance_drift = (.04 if scenario == 'physical_fields' else .002) if appearance_drift is None else appearance_drift
    if (not np.isfinite([appearance_strength, appearance_drift]).all()
            or appearance_strength <= 0 or appearance_drift <= 0):
        raise ValueError('appearance_strength and appearance_drift must be positive and finite')
    contrast = dict(physical_fields=.06, appearance_dynamics=.05, dynamics_only=.12)[scenario] if contrast is None else contrast
    rotation_rates = ((.08, .12), (.14, .17), (-.10, -.14)) if scenario == 'physical_fields' else None
    base = _generate_dynamics_only(image_size=image_size, regime_order=regime_order,
        segment_lengths=segment_lengths, rho=rho, contrast=contrast,
        observation_std=observation_std, state_mode=state_mode, seed=seed,
        rotation_rates=rotation_rates)
    if scenario == 'dynamics_only':
        return base
    height, width = image_size
    yy, xx = np.meshgrid(np.linspace(-1, 1, height), np.linspace(-1, 1, width), indexing='ij')
    radius = np.sqrt(xx**2 + yy**2)
    envelope = np.exp(-.65*radius**2)
    if scenario == 'physical_fields':
        common = .5 + .075*np.cos(np.pi*(.9*xx+.3*yy))*envelope + .035*np.sin(np.pi*yy)
        gradient = np.stack((xx, yy), axis=-1)*np.exp(-radius[..., None]**2/.9)
        gradient /= np.max(np.abs(gradient), axis=(0, 1))
        offsets = np.array(((1., 0.), (-.5, np.sqrt(3)/2), (-.5, -np.sqrt(3)/2)))
        anomalies = .018*appearance_strength*np.einsum('hwd,kd->khw', gradient, offsets)
        nominal = (common+anomalies).reshape(3, -1)
        mean_norm, initial_sd = .65, .55
    else:
        lumen = .5 - .20*np.exp(-radius**2/.055) + .05*np.exp(-(radius-.7)**2/.08)
        textures = appearance_strength * np.stack((
            .20*np.cos(3*np.pi*yy + .35*np.sin(np.pi*xx))*envelope,
            .20*np.cos(4*np.pi*radius)*envelope,
            .19*np.sin(3*np.pi*xx)*np.sin(3*np.pi*yy)*envelope))
        nominal = (lumen + textures).reshape(3, -1)
        mean_norm, initial_sd = 2., .03
    offset = nominal.mean(0)
    centered = nominal-offset
    left, singular, right = np.linalg.svd(centered, full_matrices=False)
    # Three centered nominal images span at most two coordinates. Choose their
    # The physical field means are much less separated relative to their
    # within-regime appearance variation than the tissue templates.
    appearance_scale = np.sqrt(np.sum(singular[:2]**2)/(3*mean_norm**2))
    C_appearance = right[:2].T*appearance_scale
    means = np.zeros((3, 6))
    means[:, :2] = left[:, :2]*singular[:2]/appearance_scale
    # Preserve an identifiable full-rank six-dimensional shared decoder.
    if scenario == 'physical_fields':
        phase = np.pi*(1.2*xx+.6*yy)
        eddy = np.exp(-((xx+.22)**2+(yy-.1)**2)/.35)
        dynamic = np.stack((envelope*np.cos(phase), envelope*np.sin(phase),
                            (xx+.22)*eddy, (yy-.1)*eddy), axis=-1).reshape(-1, 4)
    else:
        dynamic = base.C
    dynamic = dynamic-right[:2].T@(right[:2]@dynamic)
    dynamic, triangular = np.linalg.qr(dynamic)
    dynamic *= np.where(np.diag(triangular)<0, -1., 1.)
    C_dynamic = dynamic*contrast/np.sqrt(np.max(np.sum(dynamic**2, axis=1)))
    C = np.column_stack((C_appearance, C_dynamic))
    activity = (np.array(((1., 1., .7, .7), (.85, .85, 1., 1.), (.95, .95, .85, .85)))
                if scenario == 'physical_fields' else
                np.array(((1., 1., .15, .15), (.15, .15, 1., 1.), (.65, .65, .65, .65))))
    states = np.empty((len(base.states), 6))
    states[:, 2:] = base.states*np.sqrt(activity[base.labels])
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,)))
    for t, (k, previous) in enumerate(zip(base.labels, base.previous_indices)):
        states[t, :2] = (means[k, :2] + initial_sd*rng.standard_normal(2) if previous < 0 else
                         states[previous, :2] + appearance_drift*rng.standard_normal(2))
    A = np.zeros((3, 6, 6))
    A[:, :2, :2] = np.eye(2)
    A[:, 2:, 2:] = base.A
    Gamma = np.zeros_like(A)
    Gamma[:, :2, :2] = appearance_drift**2*np.eye(2)
    Gamma[:, 2:, 2:] = base.Gamma*activity[:, :, None]
    initial = np.zeros_like(A)
    initial[:, :2, :2] = initial_sd**2*np.eye(2)
    initial[:, 2:, 2:] = np.stack([np.diag(row) for row in activity])
    pooled_covariance = initial.mean(0)+means.T@means/len(means)
    clean = offset + states@C.T
    # Retain the base independent observation-noise draws, in raw float units.
    noisy = clean + (base.images-base.clean_images).reshape(len(states), -1)
    return SwitchingLDSImages(
        images=noisy.reshape(-1, height, width), clean_images=clean.reshape(-1, height, width),
        states=states, labels=base.labels, segment_ids=base.segment_ids,
        A=A, Gamma=Gamma, C=C, Sigma_diagonal=base.Sigma_diagonal,
        offset=offset, initial_covariance=pooled_covariance,
        previous_indices=base.previous_indices,
        regime_names=(('slow_wave', 'fast_wave', 'reverse_wave') if scenario == 'physical_fields' else
                      ('folds', 'rings', 'patches')),
        state_mode=state_mode, seed=int(seed), scenario=scenario,
        initial_means=means, initial_covariances=initial)
