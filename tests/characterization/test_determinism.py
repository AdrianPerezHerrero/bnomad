"""Exercise the determinism CLI's independent repeat and baseline gates."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import verify_determinism


class DeterminismVerifierTests(unittest.TestCase):
    def run_verifier(self, mutate):
        baseline = {'calls': [{
            'warm_start': {'dtype': 'float64', 'shape': [1], 'values': [0.]},
            'result': [{'dtype': 'float64', 'shape': [1, 1], 'values': [[0.]]}],
        }]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module = root/'tests/characterization/verify_determinism.py'
            baselines = module.with_name('baselines')
            baselines.mkdir(parents=True)
            (baselines/'warper_batch.json').write_text(json.dumps(baseline))
            report_path = root/'report.json'
            orders = []

            def capture(command, **kwargs):
                output = Path(command[command.index('--output')+1])
                output.mkdir()
                actual = copy.deepcopy(baseline)
                mutate(actual, len(orders))
                (output/'warper_batch.json').write_text(json.dumps(actual, sort_keys=True))
                orders.append('--reverse' in command)

            console = io.StringIO()
            failure = None
            with patch.object(verify_determinism, '__file__', str(module)), \
                 patch.object(verify_determinism.sys, 'argv', [str(module), '--runs', '3',
                                                              '--report', str(report_path)]), \
                 patch.object(verify_determinism.subprocess, 'run', side_effect=capture), \
                 redirect_stdout(console):
                try:
                    verify_determinism.main()
                except SystemExit as error:
                    failure = str(error)
            self.assertEqual(orders, [False, True, False])
            return json.loads(report_path.read_text()), console.getvalue(), failure

    def test_identical_captures_accept_existing_baseline_control_roundoff(self):
        def mutate(actual, run):
            actual['calls'][0]['warm_start']['values'][0] += 1.012e-7
        report, console, failure = self.run_verifier(mutate)
        self.assertIsNone(failure)
        self.assertTrue(report['all_identical'])
        self.assertTrue(report['exact_repeats_identical'])
        self.assertTrue(report['baseline_matches'])
        self.assertEqual(console.count('0 baseline regressions'), 3)

    def test_even_allowed_control_roundoff_must_repeat_byte_for_byte(self):
        def mutate(actual, run):
            actual['calls'][0]['warm_start']['values'][0] += (run+1)*1e-8
        report, console, failure = self.run_verifier(mutate)
        self.assertIn('Repeated snapshots differ', failure)
        self.assertFalse(report['all_identical'])
        self.assertFalse(report['exact_repeats_identical'])
        self.assertTrue(report['baseline_matches'])
        self.assertEqual(report['runs'][1]['different_from_first_run'], ['warper_batch.json'])
        self.assertIn('Exact repeat difference: warper_batch.json', console)

    def test_identical_captures_with_changed_warp_output_still_fail(self):
        def mutate(actual, run):
            actual['calls'][0]['result'][0]['values'][0][0] += 1e-7
        report, console, failure = self.run_verifier(mutate)
        self.assertIn('Snapshots differ from reviewed baselines', failure)
        self.assertFalse(report['all_identical'])
        self.assertTrue(report['exact_repeats_identical'])
        self.assertFalse(report['baseline_matches'])
        for run in report['runs']:
            self.assertEqual(run['outside_baseline_tolerance'], ['warper_batch.json'])
            self.assertIn('warper_batch.calls[0].result[0].values[0][0]',
                          run['baseline_differences']['warper_batch.json'])
        self.assertIn('Baseline difference in warper_batch.json:', console)


if __name__ == '__main__':
    unittest.main()
