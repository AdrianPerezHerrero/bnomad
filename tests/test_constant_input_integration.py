"""Affine latent transitions across the existing inference and emission families."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import unittest

import numpy as np
import torch

from hdpgpc import HDPGPC
from hdpgpc.config import (LDS_PRESETS, MatrixPolicyConfig, NoisePolicyConfig,
                          with_lds_preset, with_fixed_observation_model, with_mfa_emission)
from hdpgpc.diagnostics import model_lds_arrays
from hdpgpc.dynamics import _clock
from hdpgpc.emissions import (AffineGaussianObservation, estimate_autoencoder_observation,
                              estimate_pca_observation)
from test_dynamics_clock import recipe
from test_public_api import config as gp_recipe, waves


def enable(config):
    return replace(config, dynamics=replace(config.dynamics, constant_input=True))


class ConstantInputIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.output = redirect_stdout(io.StringIO())
        self.output.__enter__()

    def tearDown(self):
        self.output.__exit__(None, None, None)

    def assert_affine_model(self, model, count):
        self.assertEqual(len(model.labels_), count)
        for output in model._backend.gpmodels:
            for cluster in model.cluster_ids_:
                gp = output[int(cluster)]
                b = torch.as_tensor(gp.b[-1])
                self.assertEqual(tuple(b.shape), (gp.A[-1].shape[0], 1))
                self.assertTrue(torch.isfinite(b).all())
                prediction = model.predictive(int(cluster))
                self.assertTrue(np.isfinite(prediction.mean).all())
                self.assertTrue(np.isfinite(prediction.covariance).all())
                if _clock.enabled(gp):
                    self.assertEqual(len(_clock.trajectory(gp).means), count)
        for output in range(len(model._backend.gpmodels)):
            snapshot = model_lds_arrays(model, output=output)
            for cluster, parameters in snapshot.items():
                self.assertIn('b', parameters)
                gp = model._backend.gpmodels[output][cluster]
                before = torch.as_tensor(gp.b[-1]).clone()
                parameters['b'][:] = 1000
                torch.testing.assert_close(torch.as_tensor(gp.b[-1]), before)

    def test_schedules_updates_and_both_clocks(self):
        data = np.array([[1., .3], [1.1, .35], [.9, .25], [1.2, .3], [1., .3]])
        for clock in ('cluster-clock', 'physical-clock'):
            for schedule, updates in (
                    ('offline', ('conjugate_batch', 'conjugate_sequential', 'natural_gradient', 'legacy')),
                    ('online', ('conjugate_sequential', 'natural_gradient', 'legacy')),
                    ('online_fast', ('conjugate_sequential', 'natural_gradient', 'legacy'))):
                for update in updates:
                    with self.subTest(clock=clock, schedule=schedule, update=update):
                        cfg = enable(recipe(clock, schedule=schedule, update=update, fixed=False))
                        self.assert_affine_model(HDPGPC(cfg).fit(data), len(data))

    def test_constant_level_learns_nonzero_input_with_fixed_dynamics(self):
        data = np.repeat([[2., .6]], 15, axis=0)
        for clock in ('cluster-clock', 'physical-clock'):
            with self.subTest(clock=clock):
                model = HDPGPC(enable(recipe(clock, fixed=True))).fit(data)
                self.assert_affine_model(model, len(data))
                cluster = int(model.labels_[-1])
                gp = model._backend.gpmodels[0][cluster]
                self.assertGreater(float(gp.b[-1][0, 0]), .05)
                # Fixed A/Gamma stay fixed while the intercept is learned.
                np.testing.assert_allclose(gp.A[-1], [[.8]])
                np.testing.assert_allclose(gp.Gamma[-1], [[.1]])

    def test_hdp_presets_hmm_and_independent_assignments(self):
        data = np.array([[1., .3], [.9, .25], [-1., -.3], [-.9, -.25]])
        for clock in ('cluster-clock', 'physical-clock'):
            for preset in ('less', 'less_extra_sticky', 'balanced', 'balanced_sticky',
                           'more', 'more_sticky', 'more_extra_sticky', 'max'):
                for hmm in (False, True):
                    with self.subTest(clock=clock, preset=preset, hmm=hmm):
                        cfg = enable(recipe(clock, fixed=False))
                        cfg = replace(cfg, hdp=replace(cfg.hdp, hdp_hyp=preset, hmm_switch=hmm),
                                      inference=replace(cfg.inference, it_limit=2))
                        self.assert_affine_model(HDPGPC(cfg).fit(data), len(data))

    def test_single_observation_estimation_limits_and_online_minibatches(self):
        for clock in ('cluster-clock', 'physical-clock'):
            cfg = enable(recipe(clock, fixed=False))
            with self.subTest(clock=clock, case='single'):
                self.assert_affine_model(HDPGPC(cfg).fit(np.array([[1., .3]])), 1)
            with self.subTest(clock=clock, case='estimation_limit'):
                limited = replace(cfg, dynamics=replace(cfg.dynamics, estimation_limit=2))
                self.assert_affine_model(HDPGPC(limited).fit(np.repeat([[1., .3]], 5, axis=0)), 5)
            for schedule in ('online', 'online_fast'):
                with self.subTest(clock=clock, case='minibatch', schedule=schedule):
                    online = enable(recipe(clock, schedule=schedule, update='conjugate_sequential',
                                           fixed=False, hard=True))
                    online = replace(online, inference=replace(online.inference, minibatch=2))
                    self.assert_affine_model(HDPGPC(online).fit(np.repeat([[1., .3]], 5, axis=0)), 5)

    def test_all_lds_presets_and_shared_ownership(self):
        for clock in ('cluster-clock', 'physical-clock'):
            for preset in LDS_PRESETS:
                with self.subTest(clock=clock, preset=preset):
                    cfg = gp_recipe()
                    cfg = replace(cfg, verbose=False,
                        hdp=replace(cfg.hdp, max_models=2),
                        emission=replace(cfg.emission, fit_gp=False),
                        dynamics=replace(cfg.dynamics, clock=clock, constant_input=True,
                                         lds_update='conjugate_batch'),
                        inference=replace(cfg.inference, schedule='offline', it_limit=2, n_explore_steps=1))
                    model = HDPGPC(with_lds_preset(cfg, preset)).fit(waves()[:4])
                    self.assert_affine_model(model, 4)
                    if model._backend.transition_policies:
                        for output in model._backend.gpmodels:
                            used = [output[int(cluster)] for cluster in model.cluster_ids_]
                            for gp in used[1:]:
                                torch.testing.assert_close(gp.b[-1], used[0].b[-1])

    def test_inducing_gp_basis_and_learned_kernel(self):
        for clock in ('cluster-clock', 'physical-clock'):
            with self.subTest(clock=clock):
                cfg = gp_recipe()
                grid = np.asarray(cfg.x_basis).copy()
                cfg = replace(cfg, verbose=False, x_basis=grid[[0, 2, 4]],
                    hdp=replace(cfg.hdp, max_models=2),
                    emission=replace(cfg.emission, fit_gp=True, inducing_points=True),
                    dynamics=replace(cfg.dynamics, constant_input=True, clock=clock),
                    inference=replace(cfg.inference, schedule='offline', it_limit=2, n_explore_steps=1))
                model = HDPGPC(cfg).fit(waves()[:4], grid=grid)
                self.assert_affine_model(model, 4)
                for cluster in model.cluster_ids_:
                    gp = model._backend.gpmodels[0][int(cluster)]
                    self.assertEqual(gp.b[-1].shape, (3, 1))
                    prediction = model.predictive(int(cluster), grid=grid)
                    self.assertEqual(prediction.mean.shape, (5, 1))

    def test_native_gp_alignment_multi_output_ml_and_historical(self):
        for clock in ('cluster-clock', 'physical-clock'):
            for kind in ('dynamic', 'static', 'ml', 'multi', 'dtw', 'flow', 'historical', 'legacy'):
                with self.subTest(clock=clock, kind=kind):
                    cfg = gp_recipe('static' if kind == 'static' else 'dynamic')
                    cfg = replace(cfg, verbose=False,
                        hdp=replace(cfg.hdp, max_models=2),
                        emission=replace(cfg.emission, fit_gp=False),
                        dynamics=replace(cfg.dynamics, clock=clock, constant_input=True),
                        inference=replace(cfg.inference, schedule='offline', it_limit=2, n_explore_steps=1))
                    if kind == 'ml':
                        cfg = replace(cfg, dynamics=replace(cfg.dynamics, bayesian_params=False),
                                      inference=replace(cfg.inference, hard_assignments=True))
                    if kind == 'multi':
                        cfg = replace(cfg, multi_output=replace(cfg.multi_output, n_outputs=2))
                    if kind in ('dtw', 'flow'):
                        cfg = replace(cfg, alignment=replace(cfg.alignment,
                            policy='dtw_gp' if kind == 'dtw' else 'gp_flow',
                            flow_steps=1, flow_samples=2, flow_train_iter=0, flow_features=4,
                            flow_inducing=3, dtw_points=5, dtw_inducing=3),
                            inference=replace(cfg.inference, warp=True))
                    if kind == 'historical':
                        cfg = replace(cfg, dynamics=replace(cfg.dynamics, lds_scoring='historical'))
                    if kind == 'legacy':
                        cfg = replace(cfg, dynamics=replace(cfg.dynamics, lds_update='legacy'))
                    self.assert_affine_model(HDPGPC(cfg).fit(waves(2 if kind == 'multi' else 1)[:4]), 4)

    def test_calibrated_pca_autoencoder_and_factor_analysis(self):
        grid = np.arange(4.)[:, None]
        values = np.array([[1., .3, -.2, .1], [.8, .25, -.1, .05], [1.1, .4, -.3, .15],
                           [-1., -.3, .2, -.1], [-.8, -.2, .1, -.05], [-1.1, -.4, .3, -.15]])
        pca = estimate_pca_observation(values, grid, components=2, max_beats=6, random_state=7)
        pca_decoder = AffineGaussianObservation(pca.components, pca.residual_covariance,
            grid, pca.mean, np.diag(pca.latent_variances))
        ae_decoder = estimate_autoencoder_observation(values, grid, components=2,
            max_samples=6, epochs=3, random_state=7).model
        for clock in ('cluster-clock', 'physical-clock'):
            for name, decoder in (('pca', pca_decoder), ('autoencoder', ae_decoder),
                                  ('mfa', ae_decoder), ('shared_mfa', ae_decoder)):
                for dynamic in ((False, True) if 'mfa' in name else (True,)):
                    with self.subTest(clock=clock, emission=name, dynamic=dynamic):
                        cfg = enable(recipe(clock, fixed=False))
                        if 'mfa' in name:
                            cfg = with_mfa_emission(cfg, decoder, dynamic=dynamic,
                                                    shared=name == 'shared_mfa')
                        else:
                            cfg = with_fixed_observation_model(cfg, decoder)
                        self.assert_affine_model(HDPGPC(cfg).fit(values), len(values))

    def test_static_noop_and_independent_factor_location(self):
        initial = AffineGaussianObservation(np.array([[1.], [.3]]), np.diag([.2, .4]),
            np.arange(2.)[:, None], np.zeros(2), np.eye(1))
        for clock in ('cluster-clock', 'physical-clock'):
            with self.subTest(clock=clock, kind='zero_process_static'):
                cfg = gp_recipe('static')
                cfg = replace(cfg, verbose=False, dynamics=replace(cfg.dynamics,
                    constant_input=True, clock=clock), hdp=replace(cfg.hdp, max_models=2),
                    inference=replace(cfg.inference, schedule='offline', it_limit=2, n_explore_steps=1))
                model = HDPGPC(cfg).fit(waves()[:4])
                self.assert_affine_model(model, 4)
                for cluster in model.cluster_ids_:
                    gp = model._backend.gpmodels[0][int(cluster)]
                    torch.testing.assert_close(gp.b[-1], torch.zeros_like(gp.b[-1]))
            with self.subTest(clock=clock, kind='independent_factor_location'):
                cfg = with_mfa_emission(enable(recipe(clock, fixed=False)), initial,
                    dynamic=False, shared=False, mean_precision=1000., loading_precision=1000.)
                # Anchored observation priors make the latent location identifiable
                # in this fixture; A=0 means b is a location rather than drift.
                model = HDPGPC(cfg).fit(np.repeat([[2., .6]], 10, axis=0))
                self.assert_affine_model(model, 10)
                gp = model._backend.gpmodels[0][int(model.labels_[-1])]
                torch.testing.assert_close(gp.A[-1], torch.zeros_like(gp.A[-1]))
                self.assertGreater(float(gp.b[-1][0, 0]), .1)

    def test_structural_proposals_topologies_deepcopy_and_conversion(self):
        values = np.array([[4., 1.2]] * 3 + [[-4., -1.2]] * 3 + [[4., 1.2]] * 3)
        for clock in ('cluster-clock', 'physical-clock'):
            for topology in ('ergodic', 'left_to_right'):
                with self.subTest(clock=clock, topology=topology):
                    cfg = enable(recipe(clock, fixed=True, hard=True))
                    cfg = replace(cfg,
                        dynamics=replace(cfg.dynamics,
                            A_policy=MatrixPolicyConfig(mode='fixed', value=np.array([[.99]])),
                            Gamma_policy=NoisePolicyConfig(mode='fixed', value=np.array([[.005]]))),
                        observation=replace(cfg.observation,
                            Sigma_policy=NoisePolicyConfig(mode='fixed', value=.01 * np.eye(2))),
                        hdp=replace(cfg.hdp, max_models=3, transition_topology=topology),
                        inference=replace(cfg.inference, offline_strategy='split_merge',
                            it_limit=4, n_explore_steps=2, split_merge_refine_steps=1))
                    model = HDPGPC(cfg).fit(values)
                    self.assert_affine_model(model, len(values))
                    self.assertTrue(model.split_merge_history_)
                    self.assertTrue(any(move['kind'] == 'split' and move['accepted']
                                        for move in model.split_merge_history_))
                    self.assertTrue(any(move['kind'] == 'merge' for move in model.split_merge_history_))
                    gp = model._backend.gpmodels[0][int(model.labels_[0])]
                    saved = gp.b[-1].clone()
                    copied = model._backend.gpmodel_deepcopy(gp)
                    torch.testing.assert_close(copied.b[-1], saved)
                    copied.b[-1].add_(10.)
                    torch.testing.assert_close(gp.b[-1], saved)
                    model._backend.full_model_to_numpy()
                    self.assertIsInstance(gp.b[-1], np.ndarray)
                    np.testing.assert_allclose(gp.b[-1], saved.numpy())
                    model._backend.full_model_to_torch()
                    torch.testing.assert_close(gp.b[-1], saved)


if __name__ == '__main__':
    unittest.main()
