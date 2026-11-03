"""Frozen legacy HDP calculations and focused component contracts."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from contextlib import redirect_stdout
import io
import pickle

from hdp_cases import CASES, encode, np, torch, owner, expected_logs, optimizer, ModelState
from hdpgpc.hdp import HDPVariational
from hdpgpc.hdp import _optimizer, _variational


REFERENCE = json.loads((Path(__file__).parent/'fixtures/hdp_reference.json').read_text())['cases']


def assert_snapshot(test, actual, expected, path='state'):
    if isinstance(expected, dict):
        test.assertEqual(set(actual),set(expected),path)
        if set(expected)=={'dtype','shape','values'}:
            test.assertEqual(actual['dtype'],expected['dtype'],path)
            test.assertEqual(actual['shape'],expected['shape'],path)
            if expected['dtype'].startswith(('int','bool')):
                np.testing.assert_array_equal(actual['values'],expected['values'],err_msg=path)
            else:
                np.testing.assert_allclose(actual['values'],expected['values'],rtol=1e-7,atol=1e-9,err_msg=path)
        else:
            for key in expected:
                assert_snapshot(test,actual[key],expected[key],path+'.'+key)
    elif isinstance(expected,list):
        test.assertEqual(len(actual),len(expected),path)
        for index,(a,e) in enumerate(zip(actual,expected)):
            assert_snapshot(test,a,e,f'{path}[{index}]')
    elif isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-9, err_msg=path)
    else:
        test.assertEqual(actual,expected,path)


class HDPRegressionTests(unittest.TestCase):
    def test_reference_inventory(self):
        self.assertEqual(set(CASES),set(REFERENCE))


def regression(name):
    def test(self):
        assert_snapshot(self,encode(CASES[name]()),REFERENCE[name],name)
    test.__name__='test_'+name
    return test


for name in CASES:
    setattr(HDPRegressionTests,'test_'+name,regression(name))


class HDPComponentTests(unittest.TestCase):
    def test_stateless_legacy_calls_still_accept_no_owner(self):
        pi=torch.tensor([-.5,-1.5])
        torch.testing.assert_close(ModelState.compute_trans_pi(None,3,pi),
                                   torch.tensor([-.5,-1.5,-np.inf]),rtol=0,atol=0)
        values=np.array([.4,.6,1.2])
        self.assertEqual(ModelState.c_Dir(None,values),HDPVariational.c_Dir(None,values))

    def test_live_state_and_legacy_model_serialization(self):
        model = owner()
        component = model.hdp
        self.assertIsInstance(component,HDPVariational)
        self.assertIs(component.rho,model.rho)
        replacement = model.rho.clone()
        component.rho = replacement
        self.assertIs(model.rho,replacement)
        model.startAlpha = .25
        self.assertEqual(component.startAlpha,.25)
        self.assertNotIn('hdp',vars(model))
        restored = pickle.loads(pickle.dumps(model))
        self.assertIs(restored.hdp.rho,restored.rho)
        torch.testing.assert_close(restored.rho,model.rho,rtol=0,atol=0)
        for name in ('gpmodels','wp_sys','include_sample'):
            with self.assertRaises(AttributeError):
                getattr(component,name)
        with self.assertRaises(AttributeError):
            component.M = 99  # Lifecycle belongs to its caller.

    def test_concentration_profiles_and_unknown_profile_noop(self):
        context = SimpleNamespace(hmm_switch=True)
        component = HDPVariational(context)
        component.set_hyperparameters('unrecognized')
        self.assertEqual(vars(context),{'hmm_switch': True})
        for name,expected in (('less',(1000.,1000.,1.,0.)),('balanced',(1000.,100.,1.,0.)),('more',(10.,100.,1.,0.))):
            component.set_hyperparameters(name)
            self.assertEqual(tuple(getattr(context,key) for key in ('gamma','transAlpha','startAlpha','kappa')),expected)
        component.set_hyperparameters('unrecognized')
        self.assertEqual(context.gamma,10.)

    def test_all_schedules_use_dirichlet_expected_log_probabilities(self):
        assert_snapshot(self,expected_logs(HDPVariational),REFERENCE['expected_logs'])
        values=expected_logs(HDPVariational)
        self.assertEqual(values['online_t']['values'],values['batch_t']['values'])
        self.assertEqual(values['online_t']['values'],values['classify_t']['values'])
        self.assertEqual(values['batch_t']['values'],values['stored_t']['values'])

    def test_growth_preserves_old_prefix_without_committing_or_creating_models(self):
        model=owner(2)
        component=model.hdp
        rho,omega,theta,start=model.rho,model.omega,model.transTheta,model.startTheta
        counts=torch.zeros(4,4)
        starts=torch.zeros(4)
        grown=component.temp_reinit_global_params(3,counts,starts)
        torch.testing.assert_close(grown[0][:2],rho,rtol=0,atol=0)
        torch.testing.assert_close(grown[1][:2],omega,rtol=0,atol=0)
        self.assertEqual(grown[2].shape,(4,4))
        self.assertEqual(grown[3].shape,(4,))
        for actual,expected in zip((model.rho,model.omega,model.transTheta,model.startTheta),(rho,omega,theta,start)):
            self.assertIs(actual,expected)
        self.assertEqual(model.M,2)
        self.assertFalse(hasattr(model,'gpmodels'))
        with self.assertRaises(RuntimeError):
            component.temp_reinit_global_params(1,counts,starts)

    def test_full_theta_ignores_last_counts_and_does_not_mutate_input(self):
        model=owner(3,sticky=.7)
        counts=torch.arange(16.).reshape(4,4)
        start=torch.arange(4.)
        initial=counts.clone()
        expected=model.hdp._calcThetaFull(counts,start,4)
        torch.testing.assert_close(counts,initial,rtol=0,atol=0)
        counts[-1,:],counts[:,-1],start[-1]=1234.,1234.,1234.
        actual=model.hdp._calcThetaFull(counts,start,4)
        for a,e in zip(actual,expected):
            torch.testing.assert_close(a,e,rtol=0,atol=0)

    def test_invalid_beta_length_and_shared_optimizer_caches(self):
        with self.assertRaisesRegex(ValueError,'length K or K\\+1'):
            owner().hdp.beta2rho(np.array([.3,.7]),3)
        self.assertIs(optimizer.find_optimum_multiple_tries,_optimizer.find_optimum_multiple_tries)
        self.assertIs(optimizer.kvecCache,_optimizer.kvecCache)
        self.assertIs(optimizer.kvec(5),_optimizer.kvec(5))
        self.assertIs(optimizer._get_diagIDs(4),_optimizer._get_diagIDs(4))
        self.assertIs(pickle.loads(pickle.dumps(optimizer.find_optimum)),optimizer.find_optimum)

    def test_optimizer_bounds_and_in_place_omega_contract(self):
        rho=np.array([-1.,.4,2.])
        actual=optimizer.forceRhoInBounds(rho)
        np.testing.assert_array_equal(actual,[1e-8,.4,1.-1e-8])
        np.testing.assert_array_equal(rho,[-1.,.4,2.])
        omega=np.array([-1.,0.,5.])
        self.assertIs(optimizer.forceOmegaInBounds(omega,maxOmegaVal=3.),omega)
        np.testing.assert_array_equal(omega,[1e-8,1e-8,3.])
        scales=np.array([.8,.9,2.,3.])
        values=np.array([.2,.3,1.2,2.1])
        unconstrained=optimizer.rhoomega2c(values,scales)
        np.testing.assert_allclose(optimizer.c2rhoomega(unconstrained,scales,True),values,rtol=1e-15,atol=0)

    def test_restart_factr_sequence_and_fallback_initialization(self):
        calls=[]
        def fake(*args,**kwargs):
            calls.append(kwargs)
            if len(calls)<=5:
                raise ValueError('FAILURE: test line search')
            return np.array([.2,.3,2.,3.]),7.,{'task':'OK','grad':np.zeros(4),'warnflag':0}
        with patch.object(_optimizer,'find_optimum',side_effect=fake):
            result=optimizer.find_optimum_multiple_tries(sumLogPi=np.array([-1.,-2.,-3.]),
                                                        initrho=np.array([.2,.3]),initomega=np.array([2.,3.]))
        self.assertEqual([call['factr'] for call in calls],[1e5,1e7,1e9,1e10,1e11,1e5])
        self.assertIsNone(calls[-1]['initrho'])
        self.assertIsNone(calls[-1]['initomega'])
        self.assertEqual(result[-1]['nRestarts'],0)  # The recursive trial restarts its counters.
        self.assertEqual(result[-1]['msg'],'OK')
        self.assertNotIn('grad',result[-1])
        with patch.object(_optimizer,'find_optimum',side_effect=ValueError('different failure')) as mocked:
            with self.assertRaisesRegex(ValueError,'different failure'):
                optimizer.find_optimum_multiple_tries(sumLogPi=np.array([-1.,-2.,-3.]))
        self.assertEqual(mocked.call_count,1)

    def test_scipy_warnflag_one_is_accepted_two_raises(self):
        def result(flag):
            return lambda fn,initial,**kwargs: (initial,1.,{'warnflag':flag,'task':'test status','grad':np.zeros_like(initial)})
        with patch.object(_optimizer.scipy.optimize,'fmin_l_bfgs_b',side_effect=result(1)) as solve:
            actual=optimizer.find_optimum(sumLogPi=np.array([-1.,-2.,-3.]),nDoc=3)
        self.assertEqual(actual[-1]['warnflag'],1)
        self.assertEqual(solve.call_args.kwargs['factr'],1e5)
        self.assertFalse(solve.call_args.kwargs['approx_grad'])
        self.assertNotIn('bounds',solve.call_args.kwargs)
        with patch.object(_optimizer.scipy.optimize,'fmin_l_bfgs_b',side_effect=result(2)):
            with self.assertRaisesRegex(ValueError,'FAILURE: test status'):
                optimizer.find_optimum(sumLogPi=np.array([-1.,-2.,-3.]),nDoc=3)

    def test_optimizer_model_fallback_keeps_values_or_uses_prior(self):
        model=owner()
        model.OptimizerInfo={'retained':True}
        with patch.object(_variational,'find_optimum_multiple_tries',side_effect=ValueError('FAILURE: fixture')), redirect_stdout(io.StringIO()):
            rho,omega=model.hdp.find_optimum_rhoOmega()
            self.assertIs(rho,model.rho)
            self.assertIs(omega,model.omega)
            model.M=4
            rho,omega=model.hdp.find_optimum_rhoOmega()
        torch.testing.assert_close(rho,torch.ones(4)/2,rtol=0,atol=0)
        torch.testing.assert_close(omega,torch.ones(4)*2,rtol=0,atol=0)
        self.assertEqual(model.OptimizerInfo,{'retained':True})
        with patch.object(_variational,'find_optimum_multiple_tries',side_effect=AssertionError('fixture')):
            with self.assertRaisesRegex(AssertionError,'fixture'):
                model.hdp.find_optimum_rhoOmega()


if __name__ == '__main__':
    unittest.main()
