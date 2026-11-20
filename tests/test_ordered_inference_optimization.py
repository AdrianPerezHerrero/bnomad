"""Independent path checks and invalidation contracts for ordered inference."""
import gc
import itertools
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np
import torch

from hdpgpc.hdp import _topology
from hdpgpc.inference import _assignments, _ordered_messages as messages
from hdpgpc.inference import _proposal_families as families, _proposal_search as screen


def paths(length, count):
    for increments in itertools.product((0, 1), repeat=length-1):
        labels = (0, *itertools.accumulate(increments))
        if labels[-1] < count:
            yield labels


def exhaustive_posterior(resp, pair):
    """Enumerate the Markov joint defined by the state/pair marginals."""
    ranked = []
    for labels in paths(*resp.shape):
        score = float(resp[0, 0].log())
        for t in range(1, len(labels)):
            conditional = pair[t, labels[t-1], labels[t]] / resp[t-1, labels[t-1]]
            score += float(conditional.log()) if bool(conditional > 0) else -float('inf')
        # Dense decoding resolves ties backwards, beginning with final state.
        ranked.append((-score, tuple(reversed(labels)), labels))
    return min(ranked)[-1]


def chain(seed=7, length=5, count=3, dtype=torch.float64):
    possibilities = list(paths(length, count))
    weights = np.random.default_rng(seed).uniform(.05, 1., len(possibilities))
    weights /= weights.sum()
    resp = torch.zeros(length, count, dtype=dtype)
    pair = torch.zeros(length, count, count, dtype=dtype)
    for labels, weight in zip(possibilities, weights):
        for t, label in enumerate(labels):
            resp[t, label] += weight
            if t:
                pair[t, labels[t-1], label] += weight
    return resp, pair


class OrderedDecodingTests(unittest.TestCase):
    def test_banded_posterior_matches_exhaustive_joint_search(self):
        for dtype in (torch.float32, torch.float64):
            for seed in range(12):
                with self.subTest(dtype=dtype, seed=seed):
                    resp, pair = chain(seed, dtype=dtype)
                    self.assertEqual(tuple(messages.posterior_path(resp, pair).tolist()),
                                     exhaustive_posterior(resp, pair))

    def test_posterior_and_potential_ties_keep_their_distinct_decoding_rules(self):
        initial = torch.tensor([0., -torch.inf], dtype=torch.float64)
        transitions = torch.tensor([[0., 0.], [-torch.inf, 0.]], dtype=torch.float64)
        unary = torch.zeros(3, 2, dtype=torch.float64)
        unary[-1, 0] = -torch.inf
        resp, pair, direct = messages.marginals(initial, transitions, unary)
        self.assertEqual(direct.tolist(), [0, 1, 1])  # Stay wins potential ties.
        self.assertEqual(messages.posterior_path(resp, pair).tolist(), [0, 0, 1])

    def test_zero_mass_singleton_and_forbidden_edges_keep_existing_behavior(self):
        resp, pair = _assignments.from_labels([0, 0, 1, 1], 3, torch.empty(0, dtype=torch.float64))
        self.assertEqual(messages.posterior_path(resp, pair).tolist(), [0, 0, 1, 1])
        self.assertEqual(messages.posterior_path(resp[:1], None).tolist(), [0])
        pair.zero_()
        with self.assertRaises(FloatingPointError):
            messages.posterior_path(resp, pair)
        resp, pair = _assignments.from_labels([0, 2], 3, resp)
        with self.assertRaises(ValueError):
            messages.posterior_path(resp, pair)

    def test_hard_path_bypasses_soft_messages_and_preserves_joint_pairs(self):
        initial = torch.tensor([0., -torch.inf], dtype=torch.float64)
        transitions = torch.tensor([[.7, .3], [0., 1.]], dtype=torch.float64).log()
        unary = torch.tensor([[.9, .1], [.6, .4], [.1, .9]], dtype=torch.float64).log()
        ranked = []
        for labels in paths(3, 2):
            score = sum(float(unary[t, label]) for t, label in enumerate(labels))
            score += sum(float(transitions[a, b]) for a, b in zip(labels, labels[1:]))
            ranked.append((score, labels))
        expected = max(ranked)[1]
        with patch.object(torch, 'logaddexp', side_effect=AssertionError('soft messages')):
            resp, pair, path = messages.marginals(initial, transitions, unary, hard=True)
        self.assertEqual(tuple(path.tolist()), expected)
        torch.testing.assert_close(pair[1:].sum(2), resp[:-1], rtol=0, atol=0)
        torch.testing.assert_close(pair[1:].sum(1), resp[1:], rtol=0, atol=0)

    def test_soft_optional_path_does_not_change_marginals(self):
        initial = torch.tensor([0., -torch.inf], dtype=torch.float64)
        transitions = torch.tensor([[.7, .3], [0., 1.]], dtype=torch.float64).log()
        unary = torch.zeros(4, 2, dtype=torch.float64)
        expected = messages.marginals(initial, transitions, unary)
        with patch.object(messages, 'decode', side_effect=AssertionError('unused path')):
            resp, pair, path = messages.marginals(initial, transitions, unary, decode_path=False)
        self.assertIsNone(path)
        torch.testing.assert_close(resp, expected[0], rtol=0, atol=0)
        torch.testing.assert_close(pair, expected[1], rtol=0, atol=0)

    def test_hard_and_soft_reject_infeasible_or_nonfinite_initial_evidence(self):
        for hard in (False, True):
            for first in ((-torch.inf, -torch.inf), (0., float('nan')), (0., float('inf'))):
                with self.subTest(hard=hard, first=first), self.assertRaises(FloatingPointError):
                    messages.marginals(torch.zeros(2, dtype=torch.float64),
                        torch.zeros(2, 2, dtype=torch.float64),
                        torch.tensor([first, (0., 0.)], dtype=torch.float64), hard=hard)


class OrderedPathCacheTests(unittest.TestCase):
    def test_fitted_label_history_keeps_independent_snapshots(self):
        from hdpgpc import HDPGPC, HDPGPCConfig
        from hdpgpc.config import (AlignmentConfig, DynamicsConfig, GPEmissionConfig,
            HDPConfig, InferenceConfig, NumericalConfig, with_fixed_observation_model)
        from hdpgpc.emissions._fixed_observation import AffineGaussianObservation
        from sklearn.gaussian_process.kernels import RBF
        decoder = AffineGaussianObservation(np.array([[1.], [.5]]), .1*np.eye(2),
            np.arange(2.)[:, None], np.zeros(2), np.eye(1))
        config = HDPGPCConfig(emission=GPEmissionConfig(kernels=[RBF(1.)], fit_gp=False, ini_sigma=.1),
            dynamics=DynamicsConfig(ini_gamma=.1, annealing=False),
            alignment=AlignmentConfig(policy='none'),
            hdp=HDPConfig(M=1, max_models=1, transition_topology='left_to_right',
                          gamma=1., trans_alpha=1., start_alpha=.1),
            inference=InferenceConfig(schedule='offline', offline_strategy='split_merge',
                hard_assignments=True, n_explore_steps=0, it_limit=2),
            numerics=NumericalConfig(seed=3, num_threads=1))
        model = HDPGPC(with_fixed_observation_model(config, decoder)).fit(
            np.array([[1., .5], [1.1, .6], [.9, .4], [1., .5]]))
        history = model._backend.resp_assigned
        self.assertEqual(len(history), 2)
        self.assertNotEqual(history[0].data_ptr(), history[1].data_ptr())
        history[0][0] = 9
        self.assertEqual(int(history[1][0]), 0)
        self.assertIsNone(model._backend._offline_merge_features)

    def test_cache_detects_in_place_view_and_identity_changes(self):
        cache = messages.PathCache()
        resp, pair = _assignments.from_labels([0, 0, 1], 2, torch.empty(0, dtype=torch.float64))
        original = messages._posterior_path
        with patch.object(messages, '_posterior_path', wraps=original) as compute:
            for _ in range(2):
                self.assertEqual(messages.posterior_path(resp, pair, cache=cache).tolist(), [0, 0, 1])
            self.assertEqual(compute.call_count, 1)
            new_resp, new_pair = _assignments.from_labels([0, 1, 1], 2, resp)
            resp.copy_(new_resp); pair.copy_(new_pair)
            self.assertEqual(messages.posterior_path(resp, pair, cache=cache).tolist(), [0, 1, 1])
            self.assertEqual(compute.call_count, 2)
            pair[1:].mul_(1.)
            messages.posterior_path(resp, pair, cache=cache)
            self.assertEqual(compute.call_count, 3)
            messages.posterior_path(resp.clone(), pair.clone(), cache=cache)
            self.assertEqual(compute.call_count, 4)

    def test_cache_is_bounded_and_does_not_retain_histories(self):
        cache = messages.PathCache(maxsize=2)
        for seed in range(5):
            resp, pair = chain(seed)
            messages.posterior_path(resp, pair, cache=cache)
        references = weakref.ref(resp), weakref.ref(pair)
        del resp, pair
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))
        self.assertEqual(len(cache._entries), 2)

    def test_inference_mode_tensors_are_not_cached(self):
        cache = messages.PathCache()
        with torch.inference_mode():
            resp, pair = _assignments.from_labels([0, 0, 1], 2, torch.empty(0, dtype=torch.float64))
            for _ in range(2):
                self.assertEqual(messages.posterior_path(resp, pair, cache=cache).tolist(), [0, 0, 1])
        self.assertEqual(len(cache._entries), 0)


class OrderedProposalWorkTests(unittest.TestCase):
    def test_ordered_candidate_pools_never_compute_unused_observation_scores(self):
        features = torch.arange(12, dtype=torch.float64).reshape(6, 2)
        state = SimpleNamespace(offline_candidate_search='combined', proposal_family_weights=(1., 1., 0.),
            _offline_score_features=features, _offline_proposal_geometry=SimpleNamespace(features=features),
            _offline_dynamics_features=None, weight_mean=lambda value, snr: value)
        resp, _ = _assignments.from_labels([0, 0, 0, 1, 1, 1], 2, features)
        with patch.object(screen, 'observation_scores', side_effect=AssertionError('unused likelihood')):
            candidates = families.pools(state, resp, torch.zeros_like(resp), None, ordered=True)
        self.assertTrue(candidates['score_guided'])
        self.assertTrue(candidates['noise_aware'])

    def test_merge_feature_cache_resets_on_new_fit_even_when_shape_matches(self):
        state = SimpleNamespace(offline_candidate_search='score_guided')
        values = torch.arange(24, dtype=torch.float64).reshape(6, 4).square()
        resp, _ = _assignments.from_labels([0, 0, 0, 1, 1, 1], 2, values)
        screen.initialize(state, values)
        expected = families.merge_pools(state, resp, ordered=True)
        cached = state._offline_merge_features
        self.assertEqual(families.merge_pools(state, resp, ordered=True), expected)
        self.assertIs(state._offline_merge_features, cached)
        replacement = values.flip(0).clone()
        screen.initialize(state, replacement)
        self.assertIsNone(state._offline_merge_features)
        families.merge_pools(state, resp, ordered=True)
        reference = (replacement-replacement.mean(0))/replacement.std(0, unbiased=False).clamp_min(1e-8)
        torch.testing.assert_close(state._offline_merge_features, reference, rtol=0, atol=0)


class OrderedFixedObjectiveTests(unittest.TestCase):
    def test_precomputed_row_terms_are_exact_at_both_quadrature_orders(self):
        state = SimpleNamespace(rho=np.array([.4, .6]), omega=np.array([3., 5.]),
            gamma=1., transAlpha=1., kappa=0., device='cpu')
        counts = np.array([[5., 1.], [0., 3.]])
        theta, _ = _topology.posterior_matrices(state, counts)
        matrix = theta.numpy()
        index = np.arange(2)
        posterior = np.stack((matrix[index, index], matrix[index, index+1]), 1)
        fixed = (_topology.allowed_counts(counts, 2), *_topology._row_posterior_terms(posterior))
        for order in (128, 256):
            for rho in (state.rho, np.array([.3, .7])):
                expected = _topology.linear_terms(state, counts, rho, state.omega, theta, order=order)
                actual = _topology.linear_terms(state, counts, rho, state.omega, theta, order=order, _fixed_rows=fixed)
                self.assertEqual(actual, expected)

    def test_stick_optimizer_prepares_its_fixed_rows_once(self):
        state = SimpleNamespace(rho=np.array([.5]), omega=np.array([2.]),
            gamma=1., transAlpha=1., kappa=0., device='cpu')
        theta, _ = _topology.posterior_matrices(state, np.array([[5.]]))
        with patch.object(_topology, '_row_posterior_terms', wraps=_topology._row_posterior_terms) as prepare:
            rho, omega, info = _topology.optimize_sticks(state, theta)
        self.assertEqual(prepare.call_count, 1)
        self.assertTrue(bool(torch.isfinite(rho).all() and torch.isfinite(omega).all()))
        self.assertTrue(info['quadrature_accepted'])


if __name__ == '__main__':
    unittest.main()
