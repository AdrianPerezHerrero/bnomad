"""Guard native ownership, absence of compatibility paths and frozen evidence."""
import ast
import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys
import unittest

from tests.characterization._support import reset_runtime

RETIRED = ('GPI_HDP', 'GPI', 'GPI_model', 'GPI_models_pytorch', 'OptimizerRhoOmega',
           'amtgp_warping_system', 'warping_system', 'util_plots', 'get_data',
           'extract_data', 'GPI_test')
ROOT = Path(__file__).resolve().parents[1]


class NativeArchitectureTests(unittest.TestCase):
    def test_reviewed_numerical_fixtures_match_integrity_manifest(self):
        hashes = json.loads((ROOT/'tests/fixtures/manifest.json').read_text())['sha256']
        self.assertGreater(len(hashes), 20)
        for name, expected in hashes.items():
            payload = (ROOT/name).read_bytes()
            if name.endswith('.json'):
                payload = payload.replace(b'\r\n', b'\n')
            self.assertEqual(hashlib.sha256(payload).hexdigest(), expected, name)

    def test_public_execution_cannot_import_retired_implementations(self):
        import hdpgpc
        from hdpgpc import HDPGPC
        from hdpgpc.examples import configuration, waveforms
        from hdpgpc.model._state import ModelState
        for name in RETIRED:
            self.assertIsNone(importlib.util.find_spec('hdpgpc.'+name), name)
        for name in ('include_sample', 'include_sample_fast', 'include_batch', 'from_config'):
            self.assertFalse(hasattr(ModelState, name), name)
        self.assertEqual(ModelState.__bases__, (object,))
        self.assertEqual(set(hdpgpc.__all__), {'HDPGPC', 'HDPGPCConfig', 'UpdateResult', 'Prediction', 'LatentState'})

        class RejectRetired(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.startswith('hdpgpc.') and fullname.split('.')[1] in RETIRED:
                    raise AssertionError('Attempted retired import: '+fullname)

        guard = RejectRetired()
        sys.meta_path.insert(0, guard)
        try:
            reset_runtime()
            model = HDPGPC(configuration('online'))
            result = model.update(waveforms()[0], include_prediction=True, include_posterior=True)
            self.assertEqual(result.assigned_cluster, 0)
            self.assertEqual(type(model._backend).__module__, 'hdpgpc.model._state')
            self.assertEqual(type(model._backend.gpmodels[0][0]).__module__, 'hdpgpc.emissions._gaussian_process')
        finally:
            sys.meta_path.remove(guard)
        self.assertFalse(set('hdpgpc.'+n for n in RETIRED) & set(sys.modules))

    def test_production_has_no_retired_import_or_class_definition(self):
        import hdpgpc
        root=Path(hdpgpc.__file__).parent
        for path in root.rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf8'))):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse(set((node.module or '').split('.')) & set(RETIRED), str(path))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertFalse(set(alias.name.split('.')) & set(RETIRED), str(path))
                elif isinstance(node, ast.ClassDef):
                    self.assertNotIn(node.name, RETIRED, str(path))

    def test_native_operations_have_scientific_module_owners(self):
        from hdpgpc.model._state import ModelState
        from hdpgpc.emissions._prior import GaussianProcessPrior
        for name, owner in (
            ('new_group', 'hdpgpc.model._clusters'),
            ('forward', 'hdpgpc.inference._messages'),
            ('compute_q_elbo', 'hdpgpc.inference._scoring'),
            ('weight_mean', 'hdpgpc.emissions._weights'),
        ):
            self.assertEqual(getattr(ModelState, name).__module__, owner)
        self.assertEqual(GaussianProcessPrior.fit_torch.__module__, 'hdpgpc.emissions._training')
