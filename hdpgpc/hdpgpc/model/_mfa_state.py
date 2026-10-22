"""State and data-driven initialization for DP/HDP-HMM-MFA."""
from __future__ import annotations

import numpy as np
import torch

from ..emissions import FactorAnalysisEmission
from ..hdp import HDPVariational
from ..hdp._markov import markov_counts, update_markov_weights
from ..inference._mfa_assignments import initial_pairs


class MFAState:
    """Own FA components, one learned noise vector and a live native HDP view.

    No GP/LDS trajectories are manufactured. Only HMM mode owns transitions. Calls to
    the standalone emissions always override their fallback noise with `noise`.
    """
    def __init__(self, config, values, sequence_lengths):
        self.config = config
        self.device, self.dtype = values.device, values.dtype
        self.M, self.hmm_switch = config.hdp.M, config.hdp.hmm_switch
        self.observation_dim = values.shape[1]
        self.sequence_lengths = tuple(sequence_lengths)
        self.hdp = HDPVariational(self)
        self.hdp.set_hyperparameters(config.hdp.hdp_hyp)
        self.components = [FactorAnalysisEmission(
            self.observation_dim, config.n_factors,
            loading_prior_mean=config.loading_prior_mean,
            loading_prior_precision=config.loading_prior_precision,
            noise=config.noise_initial, noise_floor=config.noise_floor,
            dtype=self.dtype, device=self.device) for _ in range(self.M)]
        self.noise = self.components[0].noise
        self._initialize(values)
        self.history, self.elbo_history = [], []
        self.structure_history = []
        self.seen_partitions = set()
        self.n_iter, self.converged = 0, False

    def cond_to_torch(self, values):
        """Numerical context required by the native HDP stick-moment helper."""
        return torch.as_tensor(values, dtype=self.dtype, device=self.device)

    def cond_cuda(self, values):
        return values.to(self.device)

    def cond_cpu(self, values):
        return values.detach().cpu() if torch.is_tensor(values) else values

    def cond_to_numpy(self, values):
        return values.detach().cpu().numpy() if torch.is_tensor(values) else np.asarray(values)

    def update_weights(self, *, initialize=False):
        if self.hmm_switch:
            update_markov_weights(self.hdp, *markov_counts(self), initialize=initialize)
        else:
            self.hdp.update_iid_weights(self.resp)

    def _initialize(self, values):
        from sklearn.cluster import KMeans
        from sklearn.utils.extmath import randomized_svd

        n, d = values.shape
        seed = 0 if self.config.numerics.seed is None else int(self.config.numerics.seed)
        # Hard memberships only seed proper Gaussian parameters. Every actual
        # variational assignment update uses normalized soft probabilities.
        labels = KMeans(n_clusters=self.M, n_init=1, max_iter=100,
                        random_state=seed).fit_predict(values.detach().cpu().numpy())
        labels = torch.as_tensor(labels, device=self.device)
        self.resp = values.new_full((n, self.M), .05 / self.M)
        self.resp[torch.arange(n, device=self.device), labels] += .95
        center = values.mean(0)
        centered = values - center
        r = self.config.n_factors

        def initial_basis(matrix, rank, offset=0):
            # Only initialization is randomized. Conjugate updates/evidence are
            # exact; no full d-by-d singular-vector basis is allocated here.
            _, singular, directions = randomized_svd(matrix.detach().cpu().numpy(),
                n_components=rank, n_iter=4, random_state=(seed + offset) % (2**32))
            return self.cond_to_torch(singular.copy()), self.cond_to_torch(directions.copy())

        singular, vh = initial_basis(centered, min(r, *centered.shape))
        global_w = values.new_zeros((d, r))
        rank = min(r, len(singular))
        global_scale = (singular[:rank].square() / n - self.noise.mean()).clamp_min(
            max(float(centered.square().mean()) * .01, self.config.noise_floor)).sqrt()
        global_w[:, :rank] = vh[:rank].T * global_scale
        p = r + 1
        eye = torch.eye(p, dtype=self.dtype, device=self.device)
        for k, component in enumerate(self.components):
            group = values[labels == k]
            mean = group.mean(0) if len(group) else center
            w = global_w.clone()
            if len(group) > r:
                scale, directions = initial_basis(group - mean, r, k + 1)
                variance = (scale[:r].square() / len(group) - self.noise.mean())
                scale = variance.clamp_min(max(float(centered.square().mean()) * .001,
                                              self.config.noise_floor)).sqrt()
                w = directions[:r].T * scale
            b = torch.cat((mean[:, None], w), 1)
            precision = component.prior_precision + max(len(group), 1) * eye / self.noise[:, None, None]
            covariance = torch.cholesky_solve(eye.expand(d, p, p), torch.linalg.cholesky(precision))
            component.set_posterior(b, covariance)
        if self.hmm_switch:
            self.respPair = initial_pairs(self.resp, self.sequence_lengths)
        self.update_weights(initialize=True)
