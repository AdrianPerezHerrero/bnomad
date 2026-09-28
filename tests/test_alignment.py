"""Active alignment regression values captured before extracting the component."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
import io
import pickle
import sys

from alignment_cases import CASES, make_model, waveforms, reset, torch, np
from test_dynamics import assert_snapshot
from hdpgpc.alignment import GPAlignment, NoAlignment, GaussianProcessWarp


REFERENCE = json.loads((Path(__file__).parent / "fixtures/alignment_reference.json").read_text())["cases"]


class AlignmentRegressionTests(unittest.TestCase):
    def test_reference_inventory(self):
        self.assertEqual(set(CASES), set(REFERENCE))


def regression(name):
    def test(self):
        assert_snapshot(self, CASES[name](), REFERENCE[name], name)
    test.__name__ = "test_" + name
    return test


for _name in CASES:
    setattr(AlignmentRegressionTests, "test_" + _name, regression(_name))


class AlignmentComponentTests(unittest.TestCase):
    def setUp(self):
        reset()

    def test_public_alias_and_legacy_pickle_identity(self):
        self.assertEqual(GaussianProcessWarp.__bases__, (object,))
        self.assertEqual(GaussianProcessWarp.__module__, "hdpgpc.alignment._amtgp")
        self.assertIs(pickle.loads(pickle.dumps(GaussianProcessWarp)), GaussianProcessWarp)
        x, template, targets = waveforms()
        warper = GaussianProcessWarp(x, recursive=True)
        warper.compute_warp_batch(x, targets, template, train_iter=3)
        restored = pickle.loads(pickle.dumps(warper))
        torch.testing.assert_close(restored._u_ctrl_prev, warper._u_ctrl_prev, rtol=0, atol=0)
        torch.testing.assert_close(restored.warp_gp._cache_L, warper.warp_gp._cache_L, rtol=0, atol=0)

    def test_identity_has_no_optimizer_and_preserves_values_storage_and_gradients(self):
        x, template, targets = waveforms()
        y = targets[0].clone().requires_grad_()
        identity = NoAlignment().create_warper(x)
        with patch.object(torch.optim, "Adam", side_effect=AssertionError("Identity must not optimize")):
            offset, result, score, trace = identity.compute_warp(x, y, template, train_iter=250)
            batch_offset, batch_result, batch_score, _ = identity.compute_warp_batch(x, targets, template)
        self.assertIs(result, y)
        self.assertIs(batch_result, targets)
        self.assertEqual(offset.dtype, x.dtype)
        self.assertEqual(offset.device, x.device)
        self.assertEqual(torch.count_nonzero(offset).item(), 0)
        self.assertEqual(torch.count_nonzero(batch_offset).item(), 0)
        self.assertEqual(score.item(), 0.)
        self.assertEqual(torch.count_nonzero(batch_score).item(), 0)
        self.assertTrue(all(not values for values in trace.values()))
        result.sum().backward()
        torch.testing.assert_close(y.grad, torch.ones_like(y), rtol=0, atol=0)

    def test_disabled_model_never_constructs_active_warper_and_survives_cluster_factory(self):
        with patch("hdpgpc.alignment._amtgp.GaussianProcessWarp", side_effect=AssertionError("No GP warper")):
            model, xs, ys, resp = make_model(alignment=NoAlignment())
            model.create_wp_sys_default()
            candidates, offsets, scores = model.compute_warp_y(xs[0], ys[0])
            self.assertTrue(all(torch.equal(y, ys[0]) for y in candidates))
            self.assertTrue(all(torch.count_nonzero(x) == 0 for x in offsets))
            self.assertEqual(torch.count_nonzero(scores).item(), 0)
            model.warp = True  # Direct helper requests are disabled too.
            _, _, scores = model.warp_batch_by_resp_amtgp_cached(xs, ys, resp)
            self.assertEqual(torch.count_nonzero(scores).item(), 0)

    def test_cached_identity_retains_legacy_absolute_grid_and_expanded_views(self):
        active, xs, ys, resp = make_model(outputs=2)
        active.warp = False
        disabled, _, _, _ = make_model(outputs=2, alignment=NoAlignment())
        disabled.warp = True
        expected = active.warp_batch_by_resp_amtgp_cached(xs, ys, resp)
        actual = disabled.warp_batch_by_resp_amtgp_cached(xs, ys, resp)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, rtol=0, atol=0)
            self.assertEqual(a.stride(), e.stride())
        self.assertEqual(actual[0].data_ptr(), ys.data_ptr())
        self.assertEqual(actual[1].data_ptr(), xs.data_ptr())
        self.assertEqual(disabled._warp_cache_full, {})

    def test_explicit_alignment_and_cache_reset_scope(self):
        model, xs, ys, resp = make_model()
        model.warp = True
        model.warp_batch_by_resp_amtgp_cached(xs, ys, resp, train_iter=2)
        cache = model._warp_cache_full
        self.assertTrue(cache)
        model.reset_warp_cache()
        self.assertIs(model._warp_cache_full, cache)
        self.assertTrue(cache)

    def test_monotone_output_step_count_and_warm_memory_reset(self):
        x, template, targets = waveforms()
        warper = GaussianProcessWarp(x, n_ctrl=4, recursive=True)
        offset, _, _, trace = warper.compute_warp_batch(x, targets, template, train_iter=16)
        grid = x[:, 0][None] + offset[..., 0]
        self.assertTrue(torch.all(torch.diff(grid, dim=1) > 0))
        self.assertEqual(len(trace["loss"]), 16)
        self.assertFalse(offset.requires_grad)
        L = warper.warp_gp._cache_L
        warper.reset()
        self.assertIsNone(warper._u_ctrl_prev)
        self.assertIs(warper.warp_gp._cache_L, L)


class DisabledAlignmentInferenceTests(unittest.TestCase):
    """Compare constructor-level disabling against the established no-warp goldens."""

    @classmethod
    def setUpClass(cls):
        workspace = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(workspace))
        from tests.characterization import _support
        cls.support = _support
        cls.baselines = workspace / "tests/characterization/baselines"

    def compare_case(self, name, actual):
        expected = json.loads((self.baselines / (name + ".json")).read_text())
        self.support.assert_snapshot(self, self.support.encode(actual), expected, path=name)

    def test_online_constructor_disabling_matches_disabled_dynamic_static_fast(self):
        s = self.support
        for name, dynamics, fast in (("online_dynamic", "dynamic", False),
                                     ("online_static", "static", False),
                                     ("online_fast", "dynamic", True)):
            with self.subTest(case=name), redirect_stdout(io.StringIO()):
                s.reset_runtime()
                x, model = s.make_model(dynamics=dynamics, alignment=NoAlignment())
                states = []
                with s.observe_inference(model) as trace:
                    for y in s.waveform_data()[:4]:
                        method = model.online.update_fast if fast else model.online.update
                        method(x, y)  # Default with_warp=True; constructor wins.
                        states.append(s.encode(s.model_state(model)))
                self.compare_case(name, {"steps": states, "inference_trace": trace})

    def test_offline_constructor_disabling_matches_disabled_single_multi_static(self):
        s = self.support
        for name, outputs, dynamics, use_snr in (
            ("offline_dynamic", 1, "dynamic", False),
            ("offline_static", 1, "static", False),
            ("offline_multi_snr", 2, "dynamic", True),
        ):
            with self.subTest(case=name), redirect_stdout(io.StringIO()):
                s.reset_runtime()
                x, model = s.make_model(outputs=outputs, dynamics=dynamics, use_snr=use_snr,
                                        alignment=NoAlignment())
                ys = s.waveform_data(outputs)
                xs = np.repeat(x[None], len(ys), axis=0)
                with s.observe_inference(model) as trace:
                    model.offline.fit(xs, ys, it_limit=3, warp=True)
                state = s.model_state(model)
                classification = model.classify_batch(xs, ys, learning=False, warp=True)
                self.assertFalse(model.warp)
                self.compare_case(name, {"final": state, "classification": classification, "inference_trace": trace})


if __name__ == "__main__":
    unittest.main()
