"""Bound known platform differences without accepting changed scientific results."""
import unittest

from test_dynamics import assert_snapshot as component_snapshot
from tests.characterization._support import assert_snapshot as scientific_snapshot


class SnapshotPortabilityTests(unittest.TestCase):
    def test_windows_warm_start_roundoff_keeps_outputs_and_exact_mode_strict(self):
        path = 'warper_scalar.calls[0].warm_start.values[0]'
        actual, expected = 0.00012091532587203341, 0.00012094887499892379
        scientific_snapshot(self, actual, expected, path)
        for checked_path, value, exact in (
            (path, actual, True),
            ('warper_scalar.calls[0].result[0].values[0]', actual, False),
            ('emission_dynamic.warm_start.values[0]', actual, False),
            (path, expected + 1e-5, False),
        ):
            with self.subTest(path=checked_path, exact=exact), self.assertRaises(AssertionError):
                scientific_snapshot(self, value, expected, checked_path, exact=exact)

    def test_alignment_loss_roundoff_keeps_other_scalar_parameters_exact(self):
        path = 'scalar_irregular[0].result[3].loss[10]'
        component_snapshot(self, 3.598612078575355, 3.5986120785753437,
                           path, trace_atol=1e-12)
        for checked_path, change in ((path, 1e-8), ('scalar_irregular[0].state.theta', 1e-14)):
            with self.subTest(path=checked_path), self.assertRaises(AssertionError):
                component_snapshot(self, 1. + change, 1., checked_path, trace_atol=1e-12)

    def test_python_error_wording_preserves_exception_type_and_variable(self):
        expected = dict(exception_type='UnboundLocalError',
                        message="cannot access local variable 'n_f' where it is not associated with a value")
        actual = dict(exception_type='UnboundLocalError',
                      message="local variable 'n_f' referenced before assignment")
        component_snapshot(self, actual, expected)
        for replacement in (
            dict(actual, message="local variable 'other' referenced before assignment"),
            dict(actual, exception_type='ValueError'),
            dict(actual, message='a different failure'),
        ):
            with self.subTest(replacement=replacement), self.assertRaises(AssertionError):
                component_snapshot(self, replacement, expected)


if __name__ == '__main__':
    unittest.main()
