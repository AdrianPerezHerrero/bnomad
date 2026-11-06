"""Physical-clock endpoint plots retain initial gaps and omit trailing propagation."""
from types import SimpleNamespace
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from hdpgpc import HDPGPC
from hdpgpc.diagnostics import plot_clustering_summary, plot_latent_trajectories
from hdpgpc.dynamics import _clock
from test_dynamics_clock import recipe, emission


def fitted_histories(labels):
    """Known assignments isolate the plotting cutoff from structural search."""
    labels = np.asarray(labels)
    model = HDPGPC(recipe())
    models = []
    for cluster in np.unique(labels):
        gp = emission()
        for tick, assigned in enumerate(labels):
            gp.include_weighted_sample(tick, gp.direct_observation_grid,
                gp.direct_observation_grid, torch.tensor([[1.], [.3]]) * (cluster + 1),
                float(assigned == cluster))
        gp.backwards()
        models.append(gp)
    model._backend = SimpleNamespace(gpmodels=[models], T=len(labels), resp_assigned=[labels],
        cond_cuda=lambda value: value, cond_to_torch=lambda value: torch.as_tensor(value))
    model._has_data = True
    return model


class EarlyStopPlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_paths_start_at_zero_end_at_last_assignment_and_leave_histories_unchanged(self):
        model = fitted_histories([1, 1, 0, 0, 1, 1])
        gp = model._backend.gpmodels[0][0]
        before = torch.stack(_clock.trajectory(gp).covariances).clone()
        figure = plot_latent_trajectories(model, early_stop=True, time_label='Beat index')
        for row, endpoint in enumerate((3, 5)):
            axis = figure.axes[3 * row + 1]
            np.testing.assert_array_equal(axis.lines[0].get_xdata(), np.arange(endpoint + 1))
        # Cluster 0 first appears at tick 2: its unobserved initial ticks remain visible.
        self.assertEqual(figure.axes[1].lines[0].get_xdata()[0], 0)
        torch.testing.assert_close(torch.stack(_clock.trajectory(gp).covariances), before)
        self.assertEqual(len(_clock.trajectory(gp).means), 6)
        full = plot_latent_trajectories(model)
        np.testing.assert_array_equal(full.axes[1].lines[0].get_xdata(), np.arange(6))

    def test_gallery_reads_endpoint_distribution_and_inset_without_advancing(self):
        model = fitted_histories([0, 0, 1, 1, 1, 1])
        gp = model._backend.gpmodels[0][0]
        state = _clock.trajectory(gp)
        prediction = model.predictive(0, observation_index=1)
        expected_mean, expected_covariance = gp.fixed_observation_model.predictive_moments(
            state.means[1], state.covariances[1], noise=gp.Sigma[-1])
        np.testing.assert_allclose(prediction.mean[:, 0], expected_mean.numpy().reshape(-1))
        np.testing.assert_allclose(prediction.covariance[0], expected_covariance.numpy())
        self.assertEqual(model.posterior(0, observation_index=1)[0].observation_index, 1)
        np.testing.assert_allclose(model.posterior(0, observation_index=1)[0].smoothed_mean,
                                   state.means[1].numpy().reshape(-1))
        self.assertGreater(np.trace(model.predictive(0).covariance[0]),
                           np.trace(prediction.covariance[0]))
        figures = plot_clustering_summary(model, np.ones((6, 2, 1)), None, early_stop=True)
        gallery = next(axis for axis in figures[0].axes if axis.get_title().startswith('Cluster 0'))
        self.assertIn('last observation 1', gallery.get_title())
        np.testing.assert_allclose(gallery.lines[-1].get_ydata(), prediction.mean[:, 0])
        np.testing.assert_allclose(gallery.child_axes[0].lines[0].get_ydata(),
                                   state.means[1].numpy().reshape(-1))
        self.assertEqual(model.posterior(0)[0].observation_index, 5)
        self.assertEqual(len(state.means), 6)

    def test_zero_endpoint_and_invalid_or_nonphysical_queries(self):
        model = fitted_histories([0, 1, 1])
        figure = plot_latent_trajectories(model, early_stop=True)
        np.testing.assert_array_equal(figure.axes[1].lines[0].get_xdata(), [0])
        for index in (-1, 3, 1.5, True):
            for method in (model.posterior, model.predictive):
                with self.assertRaises(ValueError):
                    method(0, observation_index=index)
        legacy = HDPGPC(recipe('cluster-clock')).fit(np.array([[1., .3], [.9, .2]]))
        for plotter in (plot_latent_trajectories,
                        lambda m, **kw: plot_clustering_summary(m, np.ones((2, 2, 1)), None, **kw)):
            with self.assertRaisesRegex(ValueError, 'physical-clock'):
                plotter(legacy, early_stop=True)
        for method in (legacy.posterior, legacy.predictive):
            with self.assertRaisesRegex(ValueError, 'physical-clock'):
                method(0, observation_index=0)


if __name__ == '__main__':
    unittest.main()
