"""Exact pre-extraction call-order and posterior regression gates."""

import unittest
import inspect
import pickle
from types import SimpleNamespace
from unittest.mock import patch

from inference_cases import NAMES, EDGE_NAMES, InferenceDivergence, reference, run_case, first_difference, trace_calls
from tests.characterization._support import ModelState, make_model
from hdpgpc.inference import InferenceEngine, OnlineInference, OfflineInference

REFERENCE = reference()['cases']
REFERENCE.update(reference(supplemental=True)['cases'])


class InferenceScheduleTests(unittest.TestCase):
    def test_fixture_inventory(self):
        self.assertEqual(set(NAMES + EDGE_NAMES), set(REFERENCE))


def regression(name):
    def test(self):
        try:
            run_case(name, REFERENCE[name])
        except InferenceDivergence as error:
            self.fail(str(error))
    return test


for name in NAMES + EDGE_NAMES:
    setattr(InferenceScheduleTests, 'test_' + name, regression(name))


class InferenceEngineTests(unittest.TestCase):
    def test_direct_engines_match_pre_extraction_boundaries(self):
        for name in ('online_dynamic', 'online_fast', 'offline_multi_snr'):
            with self.subTest(case=name):
                try:
                    run_case(name, REFERENCE[name], direct=True)
                except InferenceDivergence as error:
                    self.fail(str(error))

    def test_binding_does_not_initialize_or_copy_state(self):
        owner = SimpleNamespace(marker=[])
        for cls in (InferenceEngine, OnlineInference, OfflineInference):
            engine = cls(owner)
            self.assertIs(engine.model, owner)
            self.assertEqual(vars(owner), {'marker': []})
            owner.marker.append(cls.__name__)
            self.assertIs(engine.model.marker, owner.marker)
            owner.marker.clear()
            with self.assertRaises(AttributeError):
                engine.model = SimpleNamespace()

    def test_engine_signatures_match_schedule_implementations(self):
        from hdpgpc.inference import _online, _offline
        for cls, module, names in (
            (OnlineInference, _online, ('update', 'update_fast', '_init_resp_cache_cold_start',
                                       '_ensure_fast_cache_online', '_update_q_lat_tail', '_append_hard_step')),
            (OfflineInference, _offline, ('fit', 'variational_local_terms_batch', 'estimate_q_first',
                                          'estimate_q_all', 'classify_batch', 'fit_labels')),
        ):
            for name in names:
                self.assertEqual(inspect.signature(getattr(module,name)),
                                 inspect.signature(getattr(cls,name)))

    def test_engines_dispatch_once_and_return_same_object(self):
        owner, sentinel = object(), object()
        for cls, name, module, args in (
            (OnlineInference, 'update', '_online', (1, 2, False, 3, 4, True)),
            (OnlineInference, 'update_fast', '_online', (1, 2, False, 3, 4, True)),
            (OfflineInference, 'fit', '_offline', (1, 2, 3, True)),
            (OfflineInference, 'classify_batch', '_offline', (1, 2, False, 3, True)),
            (OfflineInference, 'fit_labels', '_offline', (1, 2, 3, 4, True)),
        ):
            with self.subTest(method=name), patch('hdpgpc.inference.'+module+'.'+name,return_value=sentinel) as target:
                self.assertIs(getattr(cls(owner),name)(*args),sentinel)
                target.assert_called_once_with(owner,*args)

    def test_existing_pickle_and_owner_fields_are_unchanged_by_binding(self):
        _, owner = make_model()
        before = pickle.dumps(owner)
        OnlineInference(owner)
        OfflineInference(owner)
        self.assertEqual(pickle.dumps(owner), before)
        restored = pickle.loads(before)
        self.assertIs(type(restored), ModelState)
        self.assertEqual(vars(restored).keys(), vars(owner).keys())

    def test_divergence_reports_first_field_and_bypasses_broad_fallbacks(self):
        self.assertEqual(first_difference({'rho': [0.5, 0.4]}, {'rho': [0.5, 0.3]}),
                         'boundary.rho[1]: 0.4 != 0.3 (rtol=1e-7, atol=1e-9)')
        self.assertFalse(issubclass(InferenceDivergence, Exception))
        with self.assertRaisesRegex(InferenceDivergence, 'extra event 0'):
            with trace_calls(expected=[]):
                _, model = make_model()
                model.create_gp_default()
