"""Offline coordinate ascent for variational DP/HDP-HMM-MFA."""
import math

import torch

from ._base import InferenceEngine
from ._independent_assignments import _iid_messages
from ._mfa_assignments import markov_assignments, chain_entropy
from ..hdp._markov import markov_counts, markov_linear_elbo
from ..config._verbosity import normalize_verbosity
from ..emissions import FactorAnalysisEmission


def _evidence(state, values):
    return torch.stack([component.local_evidence(values, noise=state.noise,
                        chunk_size=state.config.chunk_size) for component in state.components], 1)


def objective_components(state, evidence):
    """Unscaled variational objective: IID ELBO or native HDP-HMM bound.

    `evidence` already optimizes q(z|s) and includes its entropy. The native
    IID uses expected allocation minus stick KL; HMM uses the native lower
    bound on HDP prior normalizers and exact chain entropy. Do not
    add factor entropy, a second stick KL, LDS terms or dimension scaling.
    """
    resp = state.resp
    local = (resp * evidence).sum()
    if state.hmm_switch:
        allocation = markov_linear_elbo(state.hdp, *markov_counts(state))
        entropy = chain_entropy(resp, state.respPair, state.sequence_lengths)
    else:
        allocation = state.hdp.iid_linear_elbo(resp)
        entropy = -torch.special.xlogy(resp, resp).sum()
    parameter_kl = torch.stack([component.parameter_kl() for component in state.components]).sum()
    total = local + allocation + entropy - parameter_kl
    return {'local_evidence': local, 'allocation': allocation,
            'assignment_entropy': entropy, 'parameter_kl': parameter_kl, 'total': total}


def _checked_phase(state, evidence, before, name):
    parts = {key: float(value) for key, value in objective_components(state, evidence).items()}
    after = parts['total']
    tolerance = 1e-8 + 1e-10 * max(1., abs(before), abs(after))
    if not math.isfinite(after) or after < before - tolerance:
        raise FloatingPointError(f'DP-MFA {name} coordinate decreased ELBO: '
                                 f'{before:.12g} -> {after:.12g} (tolerance={tolerance:.6g})')
    return parts, dict(phase=name, delta_elbo=after - before, **parts)


def coordinate_sweep(state, values, evidence, parts):
    """One complete CAVI sweep, also used for isolated structural proposals."""
    config, previous = state.config, parts["total"]
    phases = []
    # 1. q(s), then 2. q(sticks)/q(Pi), using the native assignment/HDP cores.
    if state.hmm_switch:
        state.resp, state.respPair, _ = markov_assignments(state, evidence, state.sequence_lengths)
    else:
        state.resp = _iid_messages(state, evidence)[0]
    state.update_weights()
    parts, phase = _checked_phase(state, evidence, previous, 'assignments_and_sticks')
    phases.append(phase)
    # 3. q(B) from original priors. Optimize q(z|s) again after B, so
    # the noise step uses CURRENT latent moments and current B moments.
    noise_pairs = []
    new_evidence = torch.empty_like(evidence)
    for k, component in enumerate(state.components):
        stats = component.sufficient_statistics(values, state.resp[:, k],
            noise=state.noise, chunk_size=config.chunk_size)
        component.update_from_statistics(stats, noise=state.noise)
        if config.learn_noise:
            factors = component.posterior_factors(values, noise=state.noise,
                                                  chunk_size=config.chunk_size)
            new_evidence[:, k] = factors.log_evidence
            stats = component.sufficient_statistics(values, state.resp[:, k],
                posterior=factors, chunk_size=config.chunk_size)
            noise_pairs.append(component.noise_statistics(stats))
        else:
            new_evidence[:, k] = component.local_evidence(values, noise=state.noise,
                                                         chunk_size=config.chunk_size)
    evidence = new_evidence
    parts, phase = _checked_phase(state, evidence, parts['total'], 'loadings_and_factors')
    phases.append(phase)
    # 4. One shared constrained noise M-step, followed by optimized factors.
    if config.learn_noise:
        state.noise = FactorAnalysisEmission.estimate_shared_noise(noise_pairs,
            noise_floor=config.noise_floor, previous_noise=state.noise)
        evidence = _evidence(state, values)
        parts, phase = _checked_phase(state, evidence, parts['total'], 'shared_noise_and_factors')
        phases.append(phase)
    return evidence, parts, phases


class MFAOfflineInference(InferenceEngine):
    """CAVI with shared point-noise, soft labels and optional birth/pruning."""

    @torch.no_grad()
    def fit(self, values):
        state, config = self.model, self.model.config
        verbosity = normalize_verbosity(config.verbose)
        evidence = _evidence(state, values)
        parts = {key: float(value) for key, value in objective_components(state, evidence).items()}
        if not math.isfinite(parts['total']):
            raise FloatingPointError('Initial DP-MFA ELBO is not finite')
        state.initial_elbo = parts['total']
        state.elbo_history = [state.initial_elbo]
        for iteration in range(1, config.max_iter + 1):
            previous, old_resp = parts['total'], state.resp
            old_pair = state.respPair if state.hmm_switch else None
            evidence, parts, phases = coordinate_sweep(state, values, evidence, parts)
            accepted = False
            if config.component_strategy == 'birth_prune':
                from ..model._mfa_lifecycle import explore_structure
                before_structure = parts['total']
                evidence, parts, accepted = explore_structure(state, values, evidence, parts, iteration)
                if accepted:
                    parts, phase = _checked_phase(state, evidence, before_structure, 'structure')
                    phases.append(phase)
            change = (float((state.resp - old_resp).abs().max())
                      if state.resp.shape == old_resp.shape else 1.)
            pair_change = 0.
            if state.hmm_switch:
                pair_change = (float((state.respPair - old_pair).abs().max())
                               if state.respPair.shape == old_pair.shape else 1.)
            gain = parts['total'] - previous
            state.n_iter = iteration
            state.elbo_history.append(parts['total'])
            state.history.append(dict(iteration=iteration, delta_elbo=gain,
                responsibility_change=change, pair_change=pair_change,
                structure_accepted=accepted, phases=phases, **parts))
            if verbosity != 'silent':
                counts = [round(float(value), 2) for value in state.resp.sum(0)]
                print(f'Iteration {iteration}: responsibilities={counts}; ELBO={parts["total"]:.8g}; '
                      f'delta_ELBO={gain:+.6g}', flush=True)
            if verbosity == 'debug':
                for phase in phases:
                    print(f'  {phase}', flush=True)
            if verbosity != 'silent':
                for move in state.structure_history:
                    if move['iteration'] == iteration and (verbosity == 'debug' or move['accepted']):
                        print(f'  Structure: {move}', flush=True)
            if (not accepted and abs(gain) <= config.tol * max(1., abs(previous))
                    and max(change, pair_change) <= config.tol):
                state.converged = True
                break
        state.evidence, state.elbo_parts = evidence, parts
        return state
