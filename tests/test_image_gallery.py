"""Physical image-size regressions for the single-page cluster gallery."""
from types import SimpleNamespace
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from hdpgpc.diagnostics._image_trajectories import plot_image_clustering_summary
from hdpgpc.emissions._fixed_observation import AffineGaussianObservation


class ImageGalleryLayoutTests(unittest.TestCase):
    def test_thumbnail_sizes_stay_constant_through_final_cluster(self):
        sizes = []
        for count in (1, 20):
            # Include both full and partially occupied galleries and long labels.
            labels = np.repeat(np.arange(count), [1 if k % 2 else 10 for k in range(count)])
            grid = np.arange(4.)[:, None]
            decoder = AffineGaussianObservation(np.ones((4, 1)), np.eye(4),
                                               grid, np.zeros(4), np.eye(1))
            gp = SimpleNamespace(fixed_observation_model=decoder, C=[decoder.C],
                A=[np.eye(1)], direct_observation_grid=decoder.grid,
                direct_observation_offset=decoder.offset)
            model = SimpleNamespace(labels_=labels,
                _backend=SimpleNamespace(gpmodels=[[gp] * count]))
            paths = {k: dict(indices=np.flatnonzero(labels == k),
                means=np.ones(((labels == k).sum(), 1)), prior_mean=np.zeros(1))
                for k in range(count)}
            figure, = plot_image_clustering_summary(model,
                np.ones((len(labels), 4, 1)), np.full(len(labels), 'descendingColon'),
                image_shape=(2, 2), trajectories=paths, representation_name='Learned affine MFA')
            try:
                from hdpgpc.experiments._baselines import equal_duration_baseline, add_baseline_header
                baseline = equal_duration_baseline(np.full(len(labels), 'descendingColon'), count)
                add_baseline_header(figure, baseline['metrics'], baseline)
                figure.set_dpi(20)
                figure.canvas.draw()
                width, height = figure.get_size_inches()
                image_sizes = [(axis.get_position().width * width,
                                axis.get_position().height * height)
                               for axis in figure.axes
                               if axis.images and axis.get_ylabel() != 'Current cluster']
                self.assertEqual(len(image_sizes), 4 * len(labels))
                np.testing.assert_allclose(image_sizes,
                    np.broadcast_to(image_sizes[0], (len(image_sizes), 2)),
                    rtol=1e-10, atol=1e-10)
                sizes.append(image_sizes[0])
                self.assertIsNone(figure.get_layout_engine())
            finally:
                plt.close(figure)
        np.testing.assert_allclose(sizes[0], sizes[1], rtol=1e-10, atol=1e-10)

    def test_state_overview_uses_time_and_preserves_membership_gaps(self):
        labels = np.array([0, 1, 0, 0])
        times = np.array([0., 2., 5., 9.])
        means = np.array([[1., 10.], [2., 20.], [3., 30.], [4., 40.]])
        grid = np.arange(4.)[:, None]
        decoder = AffineGaussianObservation(np.ones((4, 2)), np.eye(4),
                                           grid, np.zeros(4), np.eye(2))
        gp = SimpleNamespace(fixed_observation_model=decoder, C=[decoder.C],
            A=[np.eye(2)], direct_observation_grid=decoder.grid,
            direct_observation_offset=decoder.offset)
        model = SimpleNamespace(labels_=labels,
            _backend=SimpleNamespace(gpmodels=[[gp, gp]]))
        paths = {k: dict(indices=np.arange(4), means=means.copy(),
                         prior_mean=np.zeros(2)) for k in (0, 1)}
        for minimum in (1, 5):
            figure, = plot_image_clustering_summary(model,
                np.ones((4, 4, 1)), np.array(['cecum', 'rectum', 'cecum', 'rectum']), image_shape=(2, 2),
                times=times, trajectories=paths, min_cluster_samples=minimum)
            try:
                axis, = [a for a in figure.axes if a.name == '3d']
                self.assertEqual(axis.get_zlabel(), 'Video time (s)')
                self.assertTrue(axis.zaxis_inverted())
                truth = next(a for a in figure.axes if a.get_title() ==
                             'Reference anatomical regions (evaluation only)')
                regimes = next(a for a in figure.axes if a.get_title() == 'Inferred regimes')
                majority = next(a for a in figure.axes if a.get_title() ==
                                'Inferred regimes colored by majority label')
                reference_colors = truth.collections[0].get_facecolors()
                majority_colors = majority.collections[0].get_facecolors()
                # Cluster 0's majority is cecum, including its rectum frame.
                np.testing.assert_array_equal(majority_colors,
                    reference_colors[[0, 1, 0, 0]])
                for cluster_color in regimes.collections[0].get_facecolors():
                    self.assertFalse(any(np.allclose(cluster_color, c) for c in reference_colors))
                if minimum == 1:
                    self.assertEqual(len(axis.lines), 3)
                    # Cluster 0: do not bridge the intervening cluster-1 frame.
                    for line, indices in zip(axis.lines, ([0], [2, 3], [1])):
                        x, y, z = line.get_data_3d()
                        np.testing.assert_array_equal(x, means[indices, 0])
                        np.testing.assert_array_equal(y, means[indices, 1])
                        np.testing.assert_array_equal(z, times[indices])
                else:
                    self.assertFalse(axis.lines)
                    self.assertTrue(axis.texts)
                figure.set_dpi(20)
                figure.canvas.draw()
                np.testing.assert_array_equal(paths[0]['means'], means)
            finally:
                plt.close(figure)


if __name__ == '__main__':
    unittest.main()
