"""Projection arrows act on full native codes, without changing fitted models."""
from types import SimpleNamespace
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.quiver import Quiver
import numpy as np

from hdpgpc.diagnostics import model_embedding_dynamics, plot_embedding_dynamics
from hdpgpc.diagnostics._embedding_plots import _embedding_projection, plot_embedding_summary
from hdpgpc.diagnostics._embedding_dynamics import _operator_fields, _field_arrows


def model(assignments, matrices, inputs=None):
    groups = []
    for k,A in enumerate(matrices):
        gp = SimpleNamespace(A=[A.copy()], C=[np.eye(len(A))],Gamma=[np.eye(len(A))],
                             Sigma=[np.eye(len(A))],constant_input=inputs is not None)
        if inputs is not None:
            gp.b = [np.asarray(inputs[k]).reshape(-1,1).copy()]
        groups.append(gp)
    return SimpleNamespace(labels_=np.asarray(assignments),cluster_ids_=np.unique(assignments),
        _backend=SimpleNamespace(gpmodels=[groups]),
        config=SimpleNamespace(dynamics=SimpleNamespace(clock='physical-clock')))


class EmbeddingDynamicsTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_motion_projects_full_affine_transition_and_preserves_model(self):
        rng = np.random.default_rng(12)
        codes = rng.normal(size=(30,3))*[5.,2.,.1]
        A = np.array([[.8,0.,4.],[.2,.9,-2.],[0.,0.,.7]])
        b = np.array([.2,-.3,.1])
        fitted = model([0]*15+[1]*15,[A,np.eye(3)],[b,np.zeros(3)])
        data = model_embedding_dynamics(fitted,codes)
        future = codes.copy();future[:15] = codes[:15] @ A.T+b
        expected = (future-data['projection_center']) @ data['projection_components'].T
        np.testing.assert_allclose(data['changes'],expected-data['coordinates'],atol=1e-13)
        np.testing.assert_allclose(data['changes'][15:],0.,atol=0.)
        # A couples the third coordinate into visible motion: do not replace it
        # with the display-plane mean before applying the transition.
        lifted = data['coordinates'] @ data['projection_components']+data['projection_center']
        planar = (lifted @ (A-np.eye(3)).T+b) @ data['projection_components'].T
        self.assertGreater(np.max(np.abs(planar[:15]-data['changes'][:15])),1e-3)
        data['changes'][:] = 100
        np.testing.assert_array_equal(fitted._backend.gpmodels[0][0].A[0],A)
        np.testing.assert_array_equal(fitted._backend.gpmodels[0][0].b[0].reshape(-1),b)

    def test_figure_has_shared_projection_regions_arrows_and_no_composition_panel(self):
        codes = np.array([[-2.,0.],[-1.,.2],[1.,-.2],[2.,0.]])
        fitted = model([0,0,1,1],[.8*np.eye(2),1.2*np.eye(2)])
        figure = plot_embedding_dynamics(codes,fitted,['a','a','b','b'],region_resolution=30)
        axis = figure.axes[0]
        region = axis.images[0].get_array()
        self.assertEqual(set(np.unique(region.compressed())),{0,1})
        self.assertTrue(any(isinstance(artist,Quiver) for artist in axis.collections))
        self.assertFalse(any('composition' in artist.get_title().lower() for artist in figure.axes))
        old = plot_embedding_summary(codes,fitted.labels_,np.array(['a','a','b','b']),
            dict(purity=1.,adjusted_rand_index=1.,normalized_mutual_information=1.))
        np.testing.assert_allclose(axis.collections[0].get_offsets(),old.axes[0].collections[0].get_offsets())
        xy,_,_,_ = _embedding_projection(codes)
        np.testing.assert_allclose(axis.collections[1].get_offsets(),xy[2:])

    def test_constant_and_one_component_codes_stationary_dynamics_and_optional_labels(self):
        for codes in (np.ones((5,3)),np.arange(5.)[:,None]):
            with self.subTest(dimension=codes.shape[1]):
                fitted = model([0]*5,[np.eye(codes.shape[1])])
                figure = plot_embedding_dynamics(codes,fitted)
                self.assertFalse(any(isinstance(a,Quiver) for a in figure.axes[0].collections))
                self.assertTrue(np.isfinite(figure.axes[0].get_xlim()).all())
                figure.canvas.draw()

    def test_validation_rejects_wrong_coordinates_and_reference_lengths(self):
        fitted = model([0,0],[np.eye(2)])
        for codes in (np.zeros((2,3)),np.full((2,2),np.nan),np.zeros((1,2))):
            with self.assertRaises(ValueError):
                model_embedding_dynamics(fitted,codes)
        with self.assertRaises(ValueError):
            plot_embedding_dynamics(np.ones((2,2)),fitted,['a'])
        for kw in ({'max_arrows':0},{'region_resolution':1}):
            with self.assertRaises(ValueError):
                plot_embedding_dynamics(np.ones((2,2)),fitted,**kw)
        for coverage in (0.,1.1,np.nan,True):
            with self.assertRaises(ValueError):
                plot_embedding_dynamics(np.ones((2,2)),fitted,coverage=coverage)

    def test_operator_uses_all_members_and_matches_affine_least_squares(self):
        xy = np.array([[-2.,-1.],[-1.,1.],[0.,0.],[1.,-1.],[2.,1.]])
        changes = np.array([[0.,0.],[0.,0.],[0.,0.],[0.,0.],[4.,-3.]])
        data = dict(coordinates=xy,changes=changes,assignments=np.zeros(5,dtype=int))
        field = _operator_fields(data,.9)[0]
        expected = np.linalg.lstsq(np.column_stack((xy,np.ones(5))),changes,rcond=None)[0]
        np.testing.assert_allclose(field['jacobian'],expected[:2].T,atol=1e-14)
        np.testing.assert_allclose(field['intercept'],expected[2],atol=1e-14)
        self.assertEqual(field['members'],5)
        self.assertEqual(field['rank'],2)
        # The final observation influences the operator despite every earlier
        # vector being zero; no spatial-bin or sample quota may discard it.
        self.assertGreater(np.linalg.norm(field['jacobian']),.1)

    def test_rotation_is_preserved_even_when_average_motion_is_zero(self):
        xy = np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
        rotation = np.array([[0.,-1.],[1.,0.]])
        fields = _operator_fields(dict(coordinates=xy,changes=xy @ rotation.T,
            assignments=np.zeros(4,dtype=int)),.9)
        np.testing.assert_allclose(fields[0]['mean_motion'],0.)
        np.testing.assert_allclose(fields[0]['jacobian'],rotation,atol=1e-14)
        points,motion = _field_arrows(fields,np.array([-1.2,-1.2]),np.array([1.2,1.2]),40)
        np.testing.assert_allclose(motion,points @ rotation.T,atol=1e-14)
        np.testing.assert_allclose(np.sum(points*motion,axis=1),0.,atol=1e-14)
        self.assertTrue((points[:,0]*motion[:,1]-points[:,1]*motion[:,0] > 0).any())

    def test_empirical_core_covers_majority_and_sparse_fields_do_not_invent_slopes(self):
        xy = np.vstack((np.random.default_rng(9).normal(size=(19,2)),[100.,100.]))
        data = dict(coordinates=xy,changes=-xy,assignments=np.zeros(20,dtype=int))
        field = _operator_fields(data,.9)[0]
        centered = xy-field['center']
        distance = np.einsum('ni,ij,nj->n',centered,field['region_precision'],centered)
        self.assertGreaterEqual(np.sum(distance <= field['region_radius_squared']*(1+1e-12)),18)
        self.assertGreater(distance[-1],field['region_radius_squared'])
        self.assertEqual(field['members'],20)  # Outlier still enters the operator fit.
        sparse = _operator_fields(dict(coordinates=np.array([[3.,2.]]),changes=np.array([[1.,-.5]]),
            assignments=np.zeros(1,dtype=int)),.9)[0]
        self.assertEqual(sparse['rank'],0)
        np.testing.assert_array_equal(sparse['jacobian'],np.zeros((2,2)))
        np.testing.assert_array_equal(sparse['mean_motion'],[1.,-.5])

    def test_display_projection_is_reproducible_independently_of_global_rng(self):
        codes = np.random.default_rng(7).normal(size=(520,100))
        state = np.random.get_state()
        try:
            np.random.seed(1)
            first = _embedding_projection(codes)
            np.random.seed(99)
            second = _embedding_projection(codes)
        finally:
            np.random.set_state(state)
        for a,b in zip(first,second):
            np.testing.assert_array_equal(a,b)


if __name__ == '__main__':
    unittest.main()
