"""Quota coverage, clock correctness and dynamics-only regime sensitivity."""
from copy import copy
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from hdpgpc import HDPGPC
from hdpgpc.config import HDPGPCConfig, InferenceConfig
from hdpgpc.inference import _dynamics_proposals as dynamics, _proposal_search as screen
from hdpgpc.inference import _proposal_quotas as allocation, _split_merge
from test_proposal_search import observations, configuration


def ar_series(opposite=True, count=600):
    rng = np.random.default_rng(91)
    values = np.zeros((count, 1))
    values[0] = rng.normal()
    for row in range(1,count):
        coefficient = -.9 if opposite and row >= count//2 else .9
        values[row] = coefficient * values[row-1] + np.sqrt(1-.9**2) * rng.normal()
    return torch.tensor(values, dtype=torch.float64)


def proxy_state(count, clock='cluster-clock', indexes=None, clusters=1):
    if indexes is None:
        indexes = [list(range(count))] if clusters == 1 else [list(range(count//2)),list(range(count//2,count))]
    return SimpleNamespace(gpmodels=[[SimpleNamespace(indexes=index) for index in indexes]],
        dynamic_factor=1., constant_input=True, dynamics_clock=clock)


class QuotaTests(unittest.TestCase):
    def test_equal_coverage_budget_and_rotation(self):
        for budget in range(25):
            for offset in range(4):
                quota = allocation.quotas(budget,(1.,1.,1.),offset)
                self.assertEqual(sum(quota),budget)
                self.assertLessEqual(max(quota)-min(quota),1)
        self.assertEqual([allocation.quotas(1,(1,1,1),i).index(1) for i in range(3)],[0,1,2])

    def test_weighted_zero_and_large_finite_weights(self):
        self.assertEqual(allocation.quotas(12,(1,1,2)),[3,3,6])
        self.assertEqual(allocation.quotas(9,(0,1,2)),[0,3,6])
        self.assertEqual(allocation.quotas(6,(1e308,1e308,1e308)),[2,2,2])

    def test_family_coverage_redistribution_and_global_deduplication(self):
        pools={family:[family+str(i) for i in range(10)] for family in allocation.FAMILIES}
        selected,diagnostics=allocation.select(pools,9,(1,1,1),0,lambda item:item)
        self.assertEqual(diagnostics['selected'],dict.fromkeys(allocation.FAMILIES,3))
        self.assertEqual(len(selected),9)
        pools['noise_aware']=pools['score_guided'][:1]
        selected,diagnostics=allocation.select(pools,9,(1,1,1),0,lambda item:item)
        self.assertEqual(len({item for _,item in selected}),9)
        self.assertEqual(diagnostics['unused'],0)
        self.assertEqual(diagnostics['selected']['noise_aware'],0)
        self.assertGreater(diagnostics['duplicate_candidates'],0)

    def test_disabled_family_never_consumes_slots(self):
        pools={family:[family+str(i) for i in range(8)] for family in allocation.FAMILIES}
        selected,_=allocation.select(pools,8,(0,1,1),0,lambda item:item)
        self.assertNotIn('score_guided',[family for family,_ in selected])


class DynamicsProposalTests(unittest.TestCase):
    def test_equal_marginals_opposite_operators_reveal_the_boundary(self):
        values=ar_series()
        resp=torch.ones((len(values),1),dtype=values.dtype)
        candidates=dynamics.candidates(proxy_state(len(values)),resp,values,ordered=True)
        self.assertTrue(candidates)
        best=candidates[0]
        self.assertGreater(best['details']['conditional_likelihood_gain'],100.)
        self.assertLess(abs(best['details']['window_start']-300),45)
        self.assertGreater(best['details']['proxy_gain'],0.)

    def test_same_operator_is_more_mergeable_than_opposite_operators(self):
        scores=[]
        for opposite in (False,True):
            values=ar_series(opposite)
            labels=torch.arange(len(values))//(len(values)//2)
            resp=torch.nn.functional.one_hot(labels,2).to(values)
            ranked=dynamics.merge_scores(proxy_state(len(values),clusters=2),resp,values,[(0,1)])
            scores.append(ranked[0]['details']['proxy_gain'])
        self.assertGreater(scores[0],scores[1]+100.)

    def test_physical_gaps_are_not_treated_as_one_step(self):
        values=ar_series(count=20)
        resp=torch.ones((20,1),dtype=values.dtype)
        indexes=[list(range(0,20,2))]
        cluster=dynamics.statistics(proxy_state(20,indexes=indexes),resp,0,values)
        physical=dynamics.statistics(proxy_state(20,'physical-clock',indexes=indexes),resp,0,values)
        self.assertEqual(len(cluster.rows),9)
        self.assertIsNone(physical)

    def test_prefix_intervals_match_direct_transition_statistics(self):
        values=ar_series(count=20)
        resp=torch.full((20,1),.7,dtype=values.dtype)
        state=proxy_state(20)
        stats=dynamics.statistics(state,resp,0,values)
        xx,xy,yy,mass=dynamics.interval(stats,3,10)
        x=torch.cat((values[3:10],torch.ones((7,1),dtype=values.dtype)),1)
        y=values[4:11]
        torch.testing.assert_close(xx,.49*x.T@x)
        torch.testing.assert_close(xy,.49*x.T@y)
        torch.testing.assert_close(yy,.49*y.T@y)
        self.assertAlmostEqual(float(mass),7*.49)

    def test_dynamics_projection_does_not_rely_on_variance_screen(self):
        values=ar_series()
        noise=torch.tensor(np.random.default_rng(3).normal(size=(600,10)),dtype=values.dtype)
        codes=torch.cat((values,noise),1)
        projected=dynamics.features(codes,components=2)
        self.assertEqual(projected.shape,(600,2))
        correlation=torch.abs(torch.corrcoef(torch.cat((values,projected),1).T)[0,1:]).max()
        self.assertGreater(float(correlation),.8)


class FamilyIntegrationTests(unittest.TestCase):
    def test_default_alias_roundtrip_and_weight_validation(self):
        self.assertEqual(InferenceConfig().offline_candidate_search,'combined')
        self.assertEqual(InferenceConfig(offline_candidate_search='legacy').offline_candidate_search,'score_guided')
        for mode in ('score_guided','noise_aware','dynamics','combined'):
            cfg=HDPGPCConfig(inference=InferenceConfig(offline_candidate_search=mode,
                proposal_family_weights=(1.,2.,3.)))
            self.assertEqual(HDPGPCConfig.from_yaml(cfg.to_yaml()),cfg)
        for weights in ((0,0,0),(1,-1,1),(1,2),(1,float('inf'),1),(True,1,1)):
            with self.subTest(weights=weights),self.assertRaises(ValueError):
                HDPGPCConfig(inference=InferenceConfig(proposal_family_weights=weights))

    def test_combined_modes_clocks_soft_assignments_and_budget(self):
        values,decoder=observations(points=16,dimension=2)
        values=values[:24]
        values[12:,:2]-=8.
        for strategy in ('legacy','split_merge'):
            for clock in ('cluster-clock','physical-clock'):
                for hard in (False,True):
                    with self.subTest(strategy=strategy,clock=clock,hard=hard):
                        cfg=configuration(decoder,strategy,clock,hard)
                        cfg=replace(cfg,inference=replace(cfg.inference,offline_candidate_search='combined',
                            n_explore_steps=6,it_limit=2))
                        model=HDPGPC(cfg).fit(values,grid=decoder.grid)
                        diagnostic=model.offline_search_diagnostics_
                        self.assertEqual(diagnostic['method'],'combined')
                        for event in diagnostic.get('allocations',[]):
                            used=sum(event['splits']['selected'].values())
                            used+=sum(event.get('merges',{}).get('selected',{}).values())
                            self.assertLessEqual(used,6)
                        self.assertTrue(torch.isfinite(model._backend.q_last).all())

    def test_screening_never_replaces_native_objective_or_data(self):
        values,decoder=observations(points=32,dimension=16)
        cfg=configuration(decoder)
        cfg=replace(cfg,inference=replace(cfg.inference,offline_candidate_search='combined'))
        state=HDPGPC(cfg).fit(values,grid=decoder.grid)._backend
        state.n_explore_steps=9
        _split_merge._score(state)
        before=(float(state._split_merge_score),float(state.elbo_last))
        original=state.q_last.clone()
        moves=list(_split_merge._moves(state,state._split_merge_y.reshape(state.T,-1)))
        self.assertTrue(moves)
        self.assertLessEqual(len(moves),9)
        self.assertTrue(all(description['candidate_family'] in allocation.FAMILIES
                            for description,*_ in moves))
        _split_merge._score(state)
        self.assertEqual(before,(float(state._split_merge_score),float(state.elbo_last)))
        torch.testing.assert_close(state.q_last,original,rtol=0,atol=0)

    def test_cli_normalizes_old_name_and_exposes_family_weights(self):
        from hdpgpc.experiments.records import build_parser,make_config
        _,decoder=observations()
        args=build_parser('mitbih','autoencoder').parse_args([
            '--offline-candidate-search','legacy','--proposal-family-weights','1','2','3'])
        cfg=make_config(decoder.grid,args,decoder=decoder)
        self.assertEqual(cfg.inference.offline_candidate_search,'score_guided')
        self.assertEqual(cfg.inference.proposal_family_weights,(1.,2.,3.))


if __name__=='__main__':
    unittest.main()
