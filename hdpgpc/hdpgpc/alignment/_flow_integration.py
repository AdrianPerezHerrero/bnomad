"""Model-state integration for the variational GP-flow alignment policy."""

from __future__ import annotations

import numpy as np
import torch

from ._components import GPAlignment
from ._gp_flow import GPRepresenter, VariationalFlowWarp


class GPFlowAlignment(GPAlignment):
    """Fit a warp posterior against the current GP state for each candidate.

    Warp posteriors are keyed by observations, not owned by cluster warpers.
    Cluster warpers remain as compatibility handles and optimizer settings.
    """

    uses_posterior_flow = True
    uses_candidate_input_warp = True
    supports_warp_posterior = True

    def __init__(self, config):
        self.config = config

    def reset_warp_cache(self, owner):
        super().reset_warp_cache(owner)
        owner._flow_offline_posteriors = {}
        owner._flow_online_posteriors = {}
        owner._flow_online_scores = {}
        owner._flow_online_grids = {}
        owner._flow_pending_by_warper = {}

    def create_warper(self, *args, **kwargs):
        cfg = self.config
        return VariationalFlowWarp(
            *args, **kwargs, n_inducing=cfg.flow_inducing,
            n_features=cfg.flow_features, n_samples=cfg.flow_samples,
            ode_steps=cfg.flow_steps, train_iter=cfg.flow_train_iter,
            lengthscale=cfg.flow_lengthscale, amplitude=cfg.flow_amplitude,
            lr=cfg.flow_lr)

    def posterior_next(self, owner, grid, observation, *, candidate=None):
        """Infer p(warp | next observation, current candidate GP representers).

        Returns one posterior per candidate and one score per output channel.
        No model state is committed by this method.
        """
        x = torch.as_tensor(grid, dtype=torch.float64, device=owner.device).reshape(-1, 1)
        y = torch.as_tensor(observation, dtype=torch.float64, device=owner.device)
        if y.ndim == 1:
            y = y[:, None]
        if y.shape != (len(x), owner.n_outputs):
            raise ValueError('Expected one observation with shape (points, outputs)')
        candidates = (range(len(owner.gpmodels[0])) if candidate is None else (int(candidate),))
        posteriors = {}
        scores = {}
        for m in candidates:
            representers = [GPRepresenter(owner.gpmodels[ld][m], x.device)
                            for ld in range(owner.n_outputs)]
            warper = owner.wp_sys[0][m]
            result = warper.infer(x, y, representers)
            parts = torch.stack([gp.log_prob(result.grid_samples, y[:, ld]).mean()
                                 for ld, gp in enumerate(representers)])
            parts = parts - result.kl / owner.n_outputs
            posteriors[m] = result
            scores[m] = parts.detach()
        return posteriors, scores

    def prepare_birth(self, owner, source, grid, observation):
        """Infer the new-cluster candidate against its reinitialized GP prior."""
        x = torch.as_tensor(grid, dtype=torch.float64, device=owner.device).reshape(-1, 1)
        y = torch.as_tensor(observation, dtype=torch.float64, device=owner.device)
        if y.ndim == 1:
            y = y[:, None]
        prior_gps = []
        for ld in range(owner.n_outputs):
            gp = owner.gpmodel_deepcopy(owner.gpmodels[ld][int(source)])
            gp.reinit_GP(save_last=False)
            gp.reinit_LDS(save_last=False)
            prior_gps.append(GPRepresenter(gp, x.device))
        candidate = len(owner.gpmodels[0])
        result = owner.wp_sys[0][int(source)].infer(x, y, prior_gps)
        parts = torch.stack([gp.log_prob(result.grid_samples, y[:, ld]).mean()
                             for ld, gp in enumerate(prior_gps)])
        owner._flow_online_posteriors[candidate] = result
        owner._flow_online_scores[candidate] = (parts - result.kl / owner.n_outputs).detach()
        owner._flow_online_grids[candidate] = x + result.mean_displacement

    def bind_birth(self, owner, candidate):
        """Preserve posterior identity if the newly created cluster is reordered."""
        result = getattr(owner, '_flow_online_posteriors', {}).get(int(candidate))
        if result is not None:
            owner._flow_pending_by_warper[id(owner.wp_sys[0][-1])] = result

    def prepare_online(self, owner, x_train, y, t, M, y_mod, liks,
                       requested, force_model=None, fast=False):
        if not requested:
            for m in range(M):
                y_mod[m].append(y)
            owner._flow_online_posteriors = {}
            owner._flow_online_scores = {}
            owner._flow_online_grids = {}
            owner._flow_pending_by_warper = {}
            return y_mod, liks

        posteriors, scores = self.posterior_next(owner, x_train, y)
        owner._flow_online_posteriors = posteriors
        owner._flow_online_scores = scores
        owner._flow_online_grids = {
            m: torch.as_tensor(x_train, dtype=torch.float64,
                               device=owner.device).reshape(-1, 1) + result.mean_displacement
            for m, result in posteriors.items()}
        owner._flow_pending_by_warper = {
            id(owner.wp_sys[0][m]): result for m, result in posteriors.items()}
        for m in range(M):
            y_mod[m].append(y)
        owner.y_w.append([y] * M)
        owner.x_w.append([posteriors[m].mean_displacement for m in range(M)])
        owner.liks.append(np.asarray(liks))
        return y_mod, liks

    def update_selected(self, owner, ld, model, x_train, requested):
        if ld != 0 or not requested:
            return
        pending = getattr(owner, '_flow_pending_by_warper', {})
        result = pending.get(id(owner.wp_sys[0][model]))
        if result is None:
            result = getattr(owner, '_flow_online_posteriors', {}).get(model)
        if result is not None:
            if not hasattr(owner, 'flow_posteriors'):
                owner.flow_posteriors = []
            owner.flow_posteriors.append(result)
        owner._flow_pending_by_warper = {}

    def grid_for_candidate(self, owner, index, original):
        return getattr(owner, '_flow_online_grids', {}).get(index, original)

    def samples_for_candidate(self, owner, index):
        posteriors = getattr(owner, '_flow_online_posteriors', {})
        result = posteriors.get(int(index))
        if result is None and posteriors:
            result = posteriors[max(posteriors)]
        return None if result is None else result.grid_samples

    def proposal_score(self, owner, candidate, output, gp, observation, fallback):
        """Score a provisional GP with the candidate's sampled warp posterior."""
        result = getattr(owner, '_flow_online_posteriors', {}).get(candidate)
        if result is None:
            return fallback
        y = torch.as_tensor(observation, dtype=torch.float64,
                            device=owner.device)
        if y.ndim == 2:
            y = y[:, output]
        y = y.reshape(-1)
        representer = GPRepresenter(gp, result.grid_samples.device)
        return (representer.log_prob(result.grid_samples, y).mean()
                - result.kl / owner.n_outputs)

    def warp_batch_by_resp_amtgp_cached(self, owner, x_trains, y_trains,
                                        resp_temp, f_ind_old=None,
                                        train_iter=50, batch_size=128):
        """Infer candidate-conditional posteriors from the current batch state.

        Return the legacy tensor layout.  In this policy, ``y_trains_w`` is the
        unmodified observation and ``x_w`` contains posterior mean offsets.
        The richer sampled grids are retained for emission score evaluation.
        """
        x = torch.as_tensor(x_trains, dtype=torch.float64, device=owner.device)
        y = torch.as_tensor(y_trains, dtype=torch.float64, device=owner.device)
        resp = torch.as_tensor(resp_temp, dtype=torch.float64, device=owner.device)
        n, points, outputs = y.shape
        clusters = resp.shape[1]
        if not owner.warp:
            owner._flow_offline_posteriors = {}
            return (y.unsqueeze(-1).expand(-1, -1, -1, clusters),
                    x.unsqueeze(-1).expand(-1, -1, outputs, clusters),
                    y.new_zeros((n, clusters, outputs)))

        offsets = y.new_empty((n, points, outputs, clusters))
        result_map = {}
        for m in range(clusters):
            representatives = [GPRepresenter(
                owner.gpmodels[ld][min(m, len(owner.gpmodels[ld]) - 1)], x.device)
                for ld in range(outputs)]
            warper = owner.wp_sys[0][min(m, len(owner.wp_sys[0]) - 1)]
            for i in range(n):
                result = warper.infer(x[i], y[i], representatives)
                result_map[i, m] = result
                offsets[i, :, :, m] = result.mean_displacement.expand(-1, outputs)
        owner._flow_offline_posteriors = result_map
        return (y.unsqueeze(-1).expand(-1, -1, -1, clusters), offsets,
                y.new_zeros((n, clusters, outputs)))

    def emission_grid_batch(self, owner, x_trains, offsets, output, candidate):
        x = torch.as_tensor(x_trains, dtype=torch.float64, device=owner.device)
        return x + offsets[:, :, output, candidate].unsqueeze(-1)

    def score_offline(self, owner, gp, fallback, observations, output, candidate):
        """MC expected data score minus exactly one candidate warp KL."""
        result_map = getattr(owner, '_flow_offline_posteriors', {})
        if not result_map:
            return fallback
        if candidate < 0:
            candidate += max((m for _, m in result_map), default=-1) + 1
        representer = GPRepresenter(gp, fallback.device)
        score = fallback.clone()
        for i in range(len(score)):
            result = result_map.get((i, candidate))
            if result is None:
                continue
            score[i] = (representer.log_prob(result.grid_samples,
                         observations[i, :, output]).mean()
                        - result.kl / owner.n_outputs)
        return score
