"""Exact-work reuse, invalidation and isolation for physical-clock trajectories."""
from copy import deepcopy
from dataclasses import replace
import gc
import pickle
import unittest
from unittest.mock import patch
import weakref

import torch

from hdpgpc.dynamics import _clock, _clock_cache
from test_dynamics_clock import emission, recipe


def series(constant=False):
    cfg = recipe(fixed=False)
    cfg = replace(cfg, dynamics=replace(cfg.dynamics, constant_input=constant))
    gp = emission(cfg)
    grid = gp.direct_observation_grid
    for t, weight in enumerate([1., 0., .6, 0., 1., 0.]):
        value = torch.tensor([[1.-.3*t], [.2+.1*t]])
        gp.include_weighted_sample(t, grid, grid, value, weight)
    gp.backwards()
    return gp


class ClockOptimizationTests(unittest.TestCase):
    def assert_exact(self, first, second):
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def assert_fresh_scores(self, gp):
        reused = _clock.latent_scores(gp)
        _clock_cache.discard(gp)
        fresh = _clock.latent_scores(gp)
        self.assert_exact(reused, fresh)

    def test_missing_ticks_do_not_rebuild_retained_views(self):
        gp = series()
        grid = gp.direct_observation_grid
        views = [gp.indexes, gp.x_train, gp.y_train, gp.f_star, gp.cov_f]
        with patch.object(_clock, '_views', wraps=_clock._views) as rebuilt:
            for t in range(6, 60):
                gp.include_weighted_sample(t, grid, grid, torch.zeros(2, 1), 0.)
            rebuilt.assert_not_called()
        self.assertEqual(gp.N, 3)
        self.assertEqual(gp.indexes, [0, 2, 4])
        for name, original in zip(('indexes', 'x_train', 'y_train', 'f_star', 'cov_f'), views):
            self.assertIs(getattr(gp, name), original)

    def test_out_of_order_replacement_rejection_and_rebound_views(self):
        gp = series()
        grid = gp.direct_observation_grid
        gp.include_weighted_sample(3, grid, grid, torch.ones(2, 1), .4)
        gp.include_weighted_sample(2, grid, grid, torch.full((2, 1), 2.), .8)
        gp.include_weighted_sample(6, grid, grid, torch.ones(2, 1), 1., snr=0.)
        gp.f_star = []  # Checkpoint/copy clients may replace the retained views.
        gp.include_weighted_sample(7, grid, grid, torch.ones(2, 1), 0.)
        self.assertEqual(gp.indexes, [0, 2, 3, 4])
        state = _clock.trajectory(gp)
        for position, t in enumerate(gp.indexes):
            self.assert_exact(gp.f_star[position+1], state.filtered_means[t])
            self.assert_exact(gp.y_train[position], state.observations[t][1])

    def test_smoothing_and_statistics_share_factors_and_repeated_scores_are_free(self):
        gp = series()
        _clock_cache.discard(gp)
        with patch.object(gp, '_chol_spd', wraps=gp._chol_spd) as chol:
            gp.backwards()
            calls = chol.call_count
            self.assertEqual(calls, 5)
            _clock.transition_statistics(gp)
            self.assertEqual(chol.call_count, calls)
            expected = _clock.latent_scores(gp)
            calls = chol.call_count
            gp.backwards()
            actual = _clock.latent_scores(gp)
            self.assertEqual(chol.call_count, calls)
            self.assert_exact(actual, expected)

    def test_new_measurement_still_smooths_the_entire_history(self):
        gp = series()
        state = _clock.trajectory(gp)
        old = state.means[0].clone()
        grid = gp.direct_observation_grid
        gp.include_weighted_sample(6, grid, grid, torch.full((2, 1), 8.), 1.)
        gp.backwards()
        self.assertFalse(torch.equal(state.means[0], old))
        reused = torch.stack(state.means)
        _clock_cache.discard(gp)
        gp.backwards()
        self.assert_exact(reused, torch.stack(state.means))

    def test_scores_invalidate_for_every_mutable_dependency(self):
        for target in ('mean', 'covariance', 'filtered', 'prediction', 'historical_A',
                       'historical_Q', 'historical_b', 'A', 'Q', 'b', 'prior', 'scoring'):
            with self.subTest(target=target):
                gp = series(constant=True)
                _clock.latent_scores(gp)
                state = _clock.trajectory(gp)
                targets = {'mean': state.means[2], 'covariance': state.covariances[2],
                    'filtered': state.filtered_covariances[1],
                    'prediction': state.predicted_covariances[2],
                    'historical_A': state.matrices[2], 'historical_Q': state.noises[2],
                    'historical_b': state.offsets[2], 'A': gp.A[-1], 'Q': gp.Gamma[-1],
                    'b': gp.b[-1], 'prior': state.prior_covariance}
                if target == 'scoring':
                    gp.lds_scoring = 'historical'
                else:
                    targets[target].add_(.015)
                self.assert_fresh_scores(gp)
                # A different count cannot return a previous score vector.
                self.assert_exact(_clock.latent_scores(gp, 3), _clock.latent_scores(gp)[:3])

    def test_replaced_tensors_and_mutated_results_do_not_poison_caches(self):
        gp = series()
        scores = _clock.latent_scores(gp)
        expected = scores.clone()
        scores.fill_(100.)
        self.assert_exact(_clock.latent_scores(gp), expected)
        stats = _clock.transition_statistics(gp)
        expected_stats = tuple(value.clone() for value in stats[1:])
        stats[1].fill_(100.)
        for value, expected in zip(_clock.transition_statistics(gp)[1:], expected_stats):
            self.assert_exact(value, expected)
        state = _clock.trajectory(gp)
        state.means[2] = state.means[2]+.1
        self.assert_fresh_scores(gp)
        reused = _clock.transition_statistics(gp, start=2, end=4)
        _clock_cache.discard(gp)
        fresh = _clock.transition_statistics(gp, start=2, end=4)
        for first, second in zip(reused[1:], fresh[1:]):
            self.assert_exact(first, second)

    def test_smoother_rechecks_modified_inputs_and_outputs(self):
        for target in ('filtered', 'predicted', 'mean', 'covariance'):
            with self.subTest(target=target):
                gp = series()
                state = _clock.trajectory(gp)
                {'filtered': state.filtered_covariances, 'predicted': state.predicted_means,
                 'mean': state.means, 'covariance': state.covariances}[target][2].add_(.1)
                gp.backwards()
                reused = torch.stack(state.means), torch.stack(state.covariances)
                _clock_cache.discard(gp)
                gp.backwards()
                self.assert_exact(reused[0], torch.stack(state.means))
                self.assert_exact(reused[1], torch.stack(state.covariances))

    def test_factor_cache_compares_values_and_respects_autograd(self):
        gp = series()
        cache = _clock_cache.work(gp, _clock.trajectory(gp))
        matrix = torch.eye(2)
        calls = []
        def factor(value):
            calls.append(None)
            return torch.linalg.cholesky(value)
        cache.factor('test', matrix, factor)
        cache.factor('test', matrix.clone(), factor)
        self.assertEqual(len(calls), 1)
        # Even mutations made through an alias with a separate version counter.
        matrix.numpy()[0, 0] = 2.
        self.assert_exact(cache.factor('test', matrix, factor), factor(matrix))
        matrix.requires_grad_()
        first = cache.factor('test', matrix, factor)
        second = cache.factor('test', matrix, factor)
        self.assertIsNot(first, second)
        self.assertIsNone(_clock_cache.inputs_key((matrix,)))
        with torch.inference_mode():
            self.assertIsNone(_clock_cache.inputs_key((torch.ones_like(matrix),)))
            self.assertIsNone(_clock_cache.inputs_key((torch.eye(2),)))

    def test_shared_policies_skip_unused_statistics_without_skipping_annealing(self):
        gp = series()
        gp.transition_policy = object()
        gp.observation_policy = object()
        gp.annealing = True
        expected = gp.Gamma[-1]+gp.Gamma_def/gp.N**2
        with patch.object(_clock, 'transition_statistics', side_effect=AssertionError('unused')), \
             patch.object(_clock, 'observation_statistics', side_effect=AssertionError('unused')):
            _clock.update_parameters(gp, batch=True)
        self.assert_exact(gp.Gamma[-1], expected)

    def test_copy_pickle_reset_and_conversions_have_independent_work(self):
        gp = series(constant=True)
        expected = _clock.latent_scores(gp)
        original = _clock_cache.work(gp, _clock.trajectory(gp))
        for copied in (deepcopy(gp), pickle.loads(pickle.dumps(gp))):
            self.assertIsNot(_clock_cache.work(copied, _clock.trajectory(copied)), original)
            self.assert_exact(_clock.latent_scores(copied), expected)
        gp.model_to_numpy()
        gp.model_to_torch()
        self.assertIsNot(_clock_cache.work(gp, _clock.trajectory(gp)), original)
        self.assert_exact(_clock.latent_scores(gp), expected)
        gp.reinit_GP()
        self.assertIsNot(_clock_cache.work(gp, _clock.trajectory(gp)), original)

    def test_memory_limit_only_causes_recomputation_and_owner_is_weak(self):
        gp = series()
        expected = _clock.latent_scores(gp)
        _clock_cache.discard(gp)
        with patch.object(_clock_cache, '_RTS_BYTES', 0):
            gp.backwards()
            self.assert_exact(_clock.latent_scores(gp), expected)
            self.assertFalse(_clock_cache.work(gp, _clock.trajectory(gp)).rts)
        reference = weakref.ref(gp)
        del gp
        gc.collect()
        self.assertIsNone(reference())


if __name__ == '__main__':
    unittest.main()
