"""Physical-clock propagation, exact missing-data Gaussian checks and API isolation."""
from copy import deepcopy
from dataclasses import replace
import unittest

import numpy as np
import torch
from scipy.stats import multivariate_normal
from sklearn.gaussian_process.kernels import RBF

from hdpgpc import HDPGPC
from hdpgpc.config import (HDPGPCConfig,DynamicsConfig,GPEmissionConfig,HDPConfig,
    AlignmentConfig,InferenceConfig,NumericalConfig,MatrixPolicyConfig,NoisePolicyConfig,
    with_fixed_observation_model)
from hdpgpc.emissions._fixed_observation import AffineGaussianObservation
from hdpgpc.dynamics import _clock


def recipe(clock='physical-clock',*,schedule='offline',update='conjugate_batch',fixed=True,hard=False):
    decoder=AffineGaussianObservation(np.array([[1.],[.3]]),np.diag([.2,.4]),
        np.arange(2.)[:,None],np.zeros(2),np.eye(1))
    dynamics=DynamicsConfig(clock=clock,ini_gamma=.1,annealing=False,
        bayesian_params=True,lds_update=update,
        A_policy=MatrixPolicyConfig(mode='fixed',value=np.array([[.8]])) if fixed else None,
        Gamma_policy=NoisePolicyConfig(mode='fixed',value=np.array([[.1]])) if fixed else None)
    cfg=HDPGPCConfig(x_basis=np.array([[0.]]),verbose=False,
        emission=GPEmissionConfig(kernels=[RBF()],fit_gp=False,ini_sigma=.2),dynamics=dynamics,
        hdp=HDPConfig(max_models=2),alignment=AlignmentConfig(policy='none'),
        inference=InferenceConfig(schedule=schedule,it_limit=3 if schedule=='offline' else None,
            hard_assignments=hard,n_explore_steps=1),
        numerics=NumericalConfig(seed=7,num_threads=1,dtype='float64'))
    cfg=with_fixed_observation_model(cfg,decoder)
    return replace(cfg,dynamics=replace(cfg.dynamics,A_policy=dynamics.A_policy,
                                       Gamma_policy=dynamics.Gamma_policy))


def emission(cfg=None):
    return (cfg or recipe()).build_model().gpmodels[0][0]


class ClockMathTests(unittest.TestCase):
    def test_default_and_old_serialized_recipes(self):
        self.assertEqual(DynamicsConfig().clock,'cluster-clock')
        cfg=recipe('cluster-clock');raw=cfg.to_dict();raw['dynamics'].pop('clock',None)
        self.assertEqual(HDPGPCConfig.from_dict(raw).dynamics.clock,'cluster-clock')
        self.assertEqual(HDPGPCConfig.from_yaml(recipe().to_yaml()).dynamics.clock,'physical-clock')
        with self.assertRaises(ValueError):recipe('invalid')

    def test_missing_ticks_propagate_mean_covariance_without_observations(self):
        gp=emission();grid=gp.direct_observation_grid
        gp.include_weighted_sample(0,grid,grid,torch.tensor([[1.],[.3]]),1.)
        m=gp.f_star[-1].clone();P=gp.cov_f[-1].clone();params=deepcopy(gp.internal_params)
        gp.include_weighted_sample(3,grid,grid,torch.zeros(2,1),0.)
        state=_clock.trajectory(gp)
        torch.testing.assert_close(state.filtered_means[3],.8**3*m)
        torch.testing.assert_close(state.filtered_covariances[3],.8**6*P+.1*(1+.8**2+.8**4))
        self.assertEqual(gp.N,1);self.assertEqual(gp.indexes,[0])
        self.assertEqual(len(state.observations),1)
        torch.testing.assert_close(gp.internal_params.scale,params.scale)
        old=emission(recipe('cluster-clock'))
        old.include_weighted_sample(0,grid,grid,torch.tensor([[1.],[.3]]),1.)
        old.include_weighted_sample(3,grid,grid,torch.zeros(2,1),0.)
        torch.testing.assert_close(old.f_star[-1],m)

    def test_missing_data_smoothing_and_elbo_match_joint_gaussian(self):
        gp=emission();grid=gp.direct_observation_grid
        observations=np.array([[1.,.3],[.5,-.2]])
        gp.include_weighted_sample(0,grid,grid,torch.tensor(observations[0,:,None]),1.)
        gp.include_weighted_sample(3,grid,grid,torch.tensor(observations[1,:,None]),1.)
        gp.backwards();state=_clock.trajectory(gp)
        K=np.empty((4,4))
        variances=[1.]
        for _ in range(3):variances.append(.8**2*variances[-1]+.1)
        for i in range(4):
            for j in range(4):K[i,j]=variances[min(i,j)]*.8**abs(i-j)
        H=np.zeros((4,4));H[:2,0]=[1.,.3];H[2:,3]=[1.,.3]
        R=np.diag([.2,.4,.2,.4]);S=H @ K @ H.T+R
        y=observations.reshape(-1)
        expected_mean=K @ H.T @ np.linalg.solve(S,y)
        expected_covariance=K-K @ H.T @ np.linalg.solve(S,H @ K)
        np.testing.assert_allclose(np.array([m.item() for m in state.means]),expected_mean,rtol=1e-6,atol=1e-7)
        np.testing.assert_allclose(np.array([P.item() for P in state.covariances]),np.diag(expected_covariance),rtol=1e-6,atol=1e-7)
        score=sum(float(_clock.observation_score(gp,grid,torch.tensor(v[:,None]),t))
                  for t,v in zip([0,3],observations))+float(_clock.latent_scores(gp).sum())
        self.assertAlmostEqual(score,multivariate_normal.logpdf(y,cov=S),places=5)
        self.assertEqual(_clock.transition_statistics(gp)[0],3)
        self.assertEqual(_clock.observation_statistics(gp,[0,3])[0][0],2.)

    def test_physical_prior_entropy_is_counted_once_independently_of_assignments(self):
        cfg=recipe()
        cfg=replace(cfg,hdp=HDPConfig(M=2,max_models=2),
                    emission=replace(cfg.emission,kernels=[RBF(),RBF()]))
        state=cfg.build_model()
        grid=state.gpmodels[0][0].direct_observation_grid
        data=torch.tensor([[[1.],[.3]],[[.8],[.2]],[[-.8],[-.3]],[[-1.],[-.2]]])
        grids=grid[None].expand(4,-1,-1)
        resp=torch.tensor([[1.,0.],[0.,1.],[0.,1.],[1.,0.]])
        for k,gp in enumerate(state.gpmodels[0]):gp.full_pass_weighted(grids,data,resp[:,k])
        q=torch.stack([gp.compute_sq_err_all(grids,data) for gp in state.gpmodels[0]],1)
        qlat=torch.stack([gp.compute_q_lat_all(grids) for gp in state.gpmodels[0]],1)
        pair=torch.zeros(4,2,2);pair[1:]=resp[:-1,:,None]*resp[1:,None,:]
        _,_,components=state.compute_q_elbo(resp,pair,q,qlat,state.gpmodels,2,
            birth_score=False,return_components=True)
        expected=sum(_clock.latent_scores(gp).sum() for gp in state.gpmodels[0])
        torch.testing.assert_close(components['latent'],expected)
        changed=torch.full_like(resp,.5)
        _,_,again=state.compute_q_elbo(changed,pair,q,qlat,state.gpmodels,2,
            birth_score=False,return_components=True)
        torch.testing.assert_close(again['latent'],expected)

    def test_batch_learning_counts_missing_transitions_but_not_fake_measurements(self):
        gp=emission(recipe(fixed=False));grid=gp.direct_observation_grid
        weights=torch.tensor([1.,0.,0.,.5])
        data=torch.tensor([[[1.],[.3]],[[1000.],[1000.]],[[1000.],[1000.]],[[.5],[-.2]]])
        prior_int=float(gp.internal_params.n0);prior_obs=float(gp.observation_params.n0)
        gp.full_pass_weighted(grid[None].expand(4,-1,-1),data,weights)
        self.assertEqual(gp.indexes,[0,3]);self.assertEqual(gp.N,2)
        self.assertAlmostEqual(float(gp.internal_params.n0),prior_int+3.)
        # Fixed observation noise has no posterior data count; use raw observation statistics.
        self.assertEqual(_clock.observation_statistics(gp,[0,3])[0][0],1.5)
        self.assertTrue(torch.isfinite(gp.A[-1]).all())

    def test_online_learning_smooths_gap_and_ignores_rejected_measurement(self):
        gp=emission(recipe(update='conjugate_sequential',fixed=False))
        grid=gp.direct_observation_grid
        gp.include_weighted_sample(0,grid,grid,torch.tensor([[1.],[.3]]),1.)
        gp.include_weighted_sample(2,grid,grid,torch.zeros(2,1),0.)
        before=_clock.trajectory(gp).means[1].clone()
        degrees=float(gp.internal_params.n0)
        gp.include_weighted_sample(3,grid,grid,torch.tensor([[-1.],[-.3]]),1.)
        gp.bayesian_new_params(1.)
        self.assertFalse(torch.allclose(_clock.trajectory(gp).means[1],before))
        self.assertEqual(float(gp.internal_params.n0),degrees+3)
        degrees=float(gp.internal_params.n0)
        gp.include_weighted_sample(4,grid,grid,torch.ones(2,1),1.,snr=0.)
        gp.bayesian_new_params(1.)
        self.assertEqual(gp.indexes,[0,3])
        self.assertEqual(float(gp.internal_params.n0),degrees)

    def test_ml_minibatch_and_variance_bound_arguments(self):
        from unittest.mock import patch
        from tests.test_public_api import config
        cfg=config('dynamic')
        cfg=replace(cfg,verbose=False,alignment=AlignmentConfig(policy='none'),
            emission=replace(cfg.emission,fit_gp=False),
            dynamics=replace(cfg.dynamics,clock='physical-clock',bayesian_params=False),
            inference=replace(cfg.inference,hard_assignments=True))
        gp=emission(cfg);grid=gp.x_basis
        for t in [0,2,4]:
            gp.include_weighted_sample(t,grid,grid,torch.full((len(grid),1),float(t+1)),1.)
        bounded=torch.eye(len(grid))*.123
        with patch.object(_clock,'transition_statistics',wraps=_clock.transition_statistics) as transition, \
             patch.object(_clock,'observation_statistics',wraps=_clock.observation_statistics) as observation, \
             patch.object(gp,'check_bound_sigma',return_value=bounded) as bound:
            gp.new_params(batch=2,check_var=True)
        self.assertEqual(transition.call_args.kwargs,{'start':3,'end':5})
        self.assertEqual(observation.call_args.args[1],[2,4])
        bound.assert_called_once()
        torch.testing.assert_close(gp.Sigma[-1],bounded)


class ClockIntegrationTests(unittest.TestCase):
    def test_forced_recurring_assignments_advance_inactive_online_components(self):
        for schedule,hmm in [('online',True),('online_fast',True),('online',False),('online_fast',False)]:
            with self.subTest(schedule=schedule,hmm=hmm):
                cfg=recipe(schedule=schedule,update='conjugate_sequential',hard=True)
                cfg=replace(cfg,hdp=replace(cfg.hdp,hmm_switch=hmm))
                state=cfg.build_model()
                grid=np.arange(2.)[:,None]
                for tick,cluster in enumerate([0,1,1,0]):
                    method=state.online.update_fast if schedule=='online_fast' else state.online.update
                    method(grid,np.array([[1.],[.3]])*(1 if cluster==0 else -1),force_model=cluster)
                    for gp in state.gpmodels[0][:state.M]:
                        self.assertEqual(len(_clock.trajectory(gp).means),tick+1)
                self.assertEqual(state.gpmodels[0][0].indexes,[0,3])
                self.assertEqual(state.gpmodels[0][1].indexes,[1,2])

    def test_offline_online_and_update_policies(self):
        data=np.array([[1.,.3],[.8,.2],[-.8,-.3],[-1.,-.2],[.6,.2]])
        for schedule,updates in [('offline',['conjugate_batch','conjugate_sequential','natural_gradient','legacy']),
                                 ('online',['conjugate_sequential','natural_gradient','legacy']),
                                 ('online_fast',['conjugate_sequential','natural_gradient','legacy'])]:
            for update in updates:
                with self.subTest(schedule=schedule,update=update):
                    model=HDPGPC(recipe(schedule=schedule,update=update,fixed=False)).fit(data)
                    self.assertEqual(len(model.labels_),len(data))
                    self.assertTrue(np.isfinite(model.posterior(int(model.labels_[-1]))[0].filtered_mean).all())
                    for gp in model._backend.gpmodels[0]:
                        if gp.N:
                            state=_clock.trajectory(gp)
                            self.assertEqual(len(state.means),len(data))
                            self.assertEqual(len(gp.f_star_sm),len(gp.indexes)+1)

    def test_structural_search_topology_and_shared_observation(self):
        from hdpgpc.config import with_mfa_emission
        data=np.array([[1.,.3],[1.1,.2],[.9,.25],[-1.,-.2],[-1.1,-.3],[-.9,-.25]])
        for topology in ['ergodic','left_to_right']:
            cfg=recipe(fixed=False)
            cfg=replace(cfg,hdp=replace(cfg.hdp,transition_topology=topology),
                inference=replace(cfg.inference,offline_strategy='split_merge',
                                  split_merge_refine_steps=1,n_explore_steps=2))
            model=HDPGPC(cfg).fit(data)
            self.assertEqual(len(model.labels_),len(data))
            self.assertTrue(model.split_merge_history_)
        initial=AffineGaussianObservation(np.array([[1.],[.3]]),np.diag([.2,.4]),
                                         np.arange(2.)[:,None],np.zeros(2),np.eye(1))
        for dynamic in [False,True]:
            cfg=with_mfa_emission(recipe(fixed=False),initial,dynamic=dynamic,shared=True)
            model=HDPGPC(cfg).fit(data)
            self.assertEqual(len(model.labels_),len(data))
            self.assertTrue(np.isfinite(model.predictive(int(model.labels_[-1])).mean).all())

    def test_recurring_states_accept_split_and_keep_unobserved_states_advancing(self):
        from hdpgpc.diagnostics import model_latent_trajectories,plot_latent_trajectories
        import matplotlib.pyplot as plt
        cfg=recipe(fixed=True,hard=True)
        cfg=replace(cfg,dynamics=replace(cfg.dynamics,
                A_policy=MatrixPolicyConfig(mode='fixed',value=np.array([[.99]])),
                Gamma_policy=NoisePolicyConfig(mode='fixed',value=np.array([[.005]]))),
            observation=replace(cfg.observation,Sigma_policy=NoisePolicyConfig(mode='fixed',value=.01*np.eye(2))),
            hdp=replace(cfg.hdp,max_models=3),inference=replace(cfg.inference,
                offline_strategy='split_merge',it_limit=6,n_explore_steps=3,split_merge_refine_steps=1))
        data=np.array([[4.,1.2]]*3+[[-4.,-1.2]]*3+[[4.,1.2]]*3)
        model=HDPGPC(cfg).fit(data)
        self.assertEqual(model.n_clusters_,2)
        self.assertTrue(any(move['kind']=='split' and move['accepted'] for move in model.split_merge_history_))
        self.assertEqual(model.labels_[0],model.labels_[-1])
        self.assertNotEqual(model.labels_[0],model.labels_[4])
        model._backend.keep_last_all()
        paths=model_latent_trajectories(model)
        for cluster,path in paths.items():
            np.testing.assert_array_equal(path['indices'],np.arange(len(data)))
            self.assertEqual(path['clock'],'physical-clock')
            self.assertTrue((~path['observed']).any())
            self.assertEqual(model.posterior(cluster)[0].observation_index,len(data)-1)
        figure=plot_latent_trajectories(model,trajectories=paths,times=np.arange(len(data))*.1)
        plt.close(figure)
        gp=model._backend.gpmodels[0][int(model.labels_[0])]
        copied=model._backend.gpmodel_deepcopy(gp)
        saved=_clock.trajectory(gp).means[0].clone()
        _clock.trajectory(copied).means[0].add_(1)
        torch.testing.assert_close(_clock.trajectory(gp).means[0],saved)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_physical_clock_cuda_emission_propagation(self):
        from tests.test_public_api import config
        cfg=config('dynamic')
        cfg=replace(cfg,verbose=False,numerics=replace(cfg.numerics,cuda=True),
            alignment=AlignmentConfig(policy='none'),emission=replace(cfg.emission,fit_gp=False),
            dynamics=replace(cfg.dynamics,clock='physical-clock'))
        gp=cfg.build_model().gpmodels[0][0]
        grid=gp.x_basis
        gp.include_weighted_sample(0,grid,grid,torch.ones((len(grid),1),device='cuda'),1.)
        first=_clock.trajectory(gp).filtered_means[0].clone()
        gp.include_weighted_sample(3,grid,grid,torch.zeros((len(grid),1),device='cuda'),0.)
        state=_clock.trajectory(gp)
        self.assertEqual(state.means[-1].device.type,'cuda')
        torch.testing.assert_close(state.means[-1],torch.linalg.matrix_power(gp.A[-1],3) @ first)
        self.assertEqual(gp.N,1)

    def test_clock_state_survives_numpy_tensor_conversion(self):
        gp=emission();grid=gp.direct_observation_grid
        gp.include_weighted_sample(0,grid,grid,torch.tensor([[1.],[.3]]),1.)
        gp.include_weighted_sample(3,grid,grid,torch.zeros(2,1),0.)
        expected=_clock.trajectory(gp).filtered_means[-1].clone()
        gp.model_to_numpy();gp.model_to_torch()
        torch.testing.assert_close(_clock.trajectory(gp).filtered_means[-1],expected)
        gp.include_weighted_sample(4,grid,grid,torch.zeros(2,1),0.)
        torch.testing.assert_close(_clock.trajectory(gp).filtered_means[-1],.8*expected)

    def test_native_gp_static_ml_warps_and_shared_transitions(self):
        from tests.test_public_api import config,waves
        for kind in ['dynamic','static','ml','multi','shared','dtw','flow','historical','legacy']:
            with self.subTest(kind=kind):
                cfg=config('static' if kind=='static' else 'dynamic')
                cfg=replace(cfg,verbose=False,hdp=replace(cfg.hdp,max_models=2),
                    emission=replace(cfg.emission,fit_gp=False),
                    dynamics=replace(cfg.dynamics,clock='physical-clock'),
                    inference=replace(cfg.inference,schedule='offline',it_limit=2,n_explore_steps=1))
                if kind=='ml':
                    cfg=replace(cfg,dynamics=replace(cfg.dynamics,bayesian_params=False),
                                inference=replace(cfg.inference,hard_assignments=True))
                if kind=='multi':cfg=replace(cfg,multi_output=replace(cfg.multi_output,n_outputs=2))
                if kind=='shared':
                    cfg=replace(cfg,dynamics=replace(cfg.dynamics,
                        A_policy=MatrixPolicyConfig(mode='shared'),Gamma_policy=NoisePolicyConfig(mode='shared')))
                if kind in ('dtw','flow'):
                    cfg=replace(cfg,alignment=replace(cfg.alignment,
                        policy='dtw_gp' if kind=='dtw' else 'gp_flow',flow_steps=1,flow_samples=2,
                        flow_train_iter=0,flow_features=4,flow_inducing=3,dtw_points=5,dtw_inducing=3),
                        inference=replace(cfg.inference,warp=True))
                if kind=='historical':cfg=replace(cfg,dynamics=replace(cfg.dynamics,lds_scoring='historical'))
                if kind=='legacy':cfg=replace(cfg,dynamics=replace(cfg.dynamics,lds_update='legacy'))
                model=HDPGPC(cfg).fit(waves(2 if kind=='multi' else 1)[:4])
                self.assertEqual(len(model.labels_),4)
                self.assertTrue(np.isfinite(model.posterior(int(model.labels_[-1]))[0].filtered_mean).all())


if __name__=='__main__':unittest.main()
