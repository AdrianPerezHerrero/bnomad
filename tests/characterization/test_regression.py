"""Golden comparisons of current behavior, including two known failure paths."""

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import unittest

from ._cases import CASES
from ._support import assert_snapshot, encode, reset_runtime, source_hashes

BASELINES = Path(__file__).with_name("baselines")


class CharacterizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_sources = source_hashes()
        cls.manifest = json.loads((BASELINES / "manifest.json").read_text(encoding="utf-8"))
        if set(cls.manifest["cases"]) != set(CASES):
            raise AssertionError("Every case must have an explicit, reviewed baseline")

    @classmethod
    def tearDownClass(cls):
        if source_hashes() != cls.original_sources:
            raise AssertionError("Characterization execution modified production source")

    def setUp(self):
        reset_runtime()

    def check_case(self, name):
        expected = json.loads((BASELINES / f"{name}.json").read_text(encoding="utf-8"))
        # Silence only progress printing. Warnings and exceptions remain visible.
        with redirect_stdout(io.StringIO()):
            actual = encode(CASES[name]())
        assert_snapshot(self, actual, expected, path=name)
        if name == "online_static":
            self.assertEqual([s["M"] for s in actual["steps"]], [1, 1, 2, 2])
            self.assertEqual([s["actual_state"] for s in actual["steps"]], [0, 0, 1, 1])
        if name.startswith("offline_"):
            self.assertEqual(actual["final"]["M"], 2)
            self.assertEqual(actual["final"]["occupied_clusters"], [2] * actual["final"]["n_outputs"])
        if name.startswith("warper_"):
            warp = actual["calls"][0]["result"][0]["values"]
            self.assertTrue(any(abs(v) > 1e-3 for v in _flatten(warp)), "Fixture must exercise nonidentity warping")


def _flatten(values):
    for value in values:
        if isinstance(value, list):
            yield from _flatten(value)
        else:
            yield value


def _case_test(name):
    def test(self):
        self.check_case(name)
    test.__name__ = f"test_{name}"
    test.__doc__ = f"Protect the recorded {name} path (see docs/characterization_tests.md)."
    return test


for _name in CASES:
    setattr(CharacterizationTests, f"test_{_name}", _case_test(_name))


class SnapshotPrecisionTests(unittest.TestCase):
    def test_determinism_file_paths_use_the_same_baseline_rules(self):
        from .verify_determinism import assert_baseline_snapshot
        for case in ('warper_scalar', 'warper_batch'):
            with self.subTest(case=case):
                expected = json.loads((BASELINES/(case+'.json')).read_text(encoding='utf-8'))
                actual = copy.deepcopy(expected)
                actual['calls'][1]['warm_start']['values'][0] += 1.012e-7
                # Exercise the filename-taking entry point used by the CLI.
                assert_baseline_snapshot(actual, expected, case+'.json')
                with self.assertRaises(AssertionError):
                    assert_snapshot(self, actual, expected, path=case, exact=True)
                actual['calls'][1]['warm_start']['values'][0] += 1e-5
                with self.assertRaises(AssertionError):
                    assert_baseline_snapshot(actual, expected, case+'.json')
                actual = copy.deepcopy(expected)
                # Scalar offsets are (T,1); batch offsets are (B,T,1).
                values = actual['calls'][1]['result'][0]['values']
                while isinstance(values[0], list):
                    values = values[0]
                values[0] += 1e-7
                with self.assertRaises(AssertionError):
                    assert_baseline_snapshot(actual, expected, case+'.json')

    def test_optimizer_control_bound_keeps_warp_outputs_and_exact_checks_strict(self):
        expected = json.loads((BASELINES/'warper_scalar.json').read_text(encoding='utf-8'))
        actual = copy.deepcopy(expected)
        # Observed Windows control drift is 1.012e-7; outputs are unchanged.
        actual['calls'][1]['warm_start']['values'][0] -= 1.012e-7
        assert_snapshot(self, actual, expected, path='warper_scalar')
        with self.assertRaises(AssertionError):
            assert_snapshot(self, actual, expected, path='warper_scalar', exact=True)
        actual['calls'][1]['warm_start']['values'][0] += 1e-5
        with self.assertRaises(AssertionError):
            assert_snapshot(self, actual, expected, path='warper_scalar')
        actual = copy.deepcopy(expected)
        actual['calls'][1]['result'][0]['values'][0][0] += 1e-7
        with self.assertRaises(AssertionError):
            assert_snapshot(self, actual, expected, path='warper_scalar')


if __name__ == "__main__":
    unittest.main()
