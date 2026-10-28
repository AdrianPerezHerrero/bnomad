"""Check additive import boundaries without running research experiment scripts."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "hdpgpc"


class PackageStructureTests(unittest.TestCase):
    def run_fresh(self, code):
        env = os.environ.copy()
        if env.get("HDPGPC_TEST_INSTALLED"):
            env.pop("PYTHONPATH", None)
        else:
            env["PYTHONPATH"] = str(SOURCE)
        env["MPLBACKEND"] = "Agg"
        result = subprocess.run(
            [sys.executable, "-B", "-c", textwrap.dedent(code)],
            env=env, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_namespace_imports_do_not_initialize_numerics(self):
        self.run_fresh("""
            import importlib
            import importlib.util
            import sys

            areas = ('model', 'config', 'hdp', 'emissions', 'dynamics',
                     'alignment', 'inference', 'numerics', 'datasets')
            import hdpgpc
            for area in areas:
                importlib.import_module('hdpgpc.' + area)
            from hdpgpc.emissions import EmissionModel
            assert EmissionModel.__name__ == 'EmissionModel'
            from hdpgpc.dynamics import DynamicsModel
            assert DynamicsModel.__name__ == 'DynamicsModel'
            from hdpgpc.config import HDPGPCConfig, InferenceConfig, NumericalConfig
            assert HDPGPCConfig.__name__ == 'HDPGPCConfig'
            from hdpgpc import HDPGPC, UpdateResult, Prediction, LatentState
            from hdpgpc.model import HDPGPC as ModelAlias
            assert HDPGPC is ModelAlias
            assert 'numpy' not in sys.modules
            assert 'torch' not in sys.modules
            assert 'hdpgpc.GPI_HDP' not in sys.modules
            assert 'hdpgpc.GPI_model' not in sys.modules
            assert 'hdpgpc.alignment._amtgp' not in sys.modules
            assert 'hdpgpc.get_data' not in sys.modules
            assert 'hdpgpc.extract_data' not in sys.modules

            # Retired implementation modules must not resolve.
            for name in ('GPI_HDP', 'GPI_model', 'GPI', 'GPI_models_pytorch',
                         'OptimizerRhoOmega', 'amtgp_warping_system',
                         'warping_system', 'get_data', 'extract_data',
                         'util_plots', 'GPI_test'):
                assert importlib.util.find_spec('hdpgpc.' + name) is None
        """)

    def test_canonical_components_own_classes_and_pickle_paths(self):
        self.run_fresh("""
            import pickle
            from hdpgpc.emissions import GaussianProcessEmission
            from hdpgpc.emissions._prior import GaussianProcessPrior
            from hdpgpc.alignment import GaussianProcessWarp
            for cls, path in ((GaussianProcessEmission, 'hdpgpc.emissions._gaussian_process'),
                              (GaussianProcessPrior, 'hdpgpc.emissions._prior'),
                              (GaussianProcessWarp, 'hdpgpc.alignment._amtgp')):
                assert cls.__module__ == path
                assert cls.__bases__ == (object,)
                assert pickle.loads(pickle.dumps(cls)) is cls
        """)

    def test_dynamics_components_do_not_import_model_or_emission(self):
        self.run_fresh("""
            import sys
            import torch
            before = torch.get_default_dtype()
            from hdpgpc.dynamics import DynamicsModel, StaticDynamics, LinearDynamics
            assert isinstance(StaticDynamics(), DynamicsModel)
            assert isinstance(LinearDynamics(), DynamicsModel)
            assert torch.get_default_dtype() == before
            for name in ('hdpgpc.GPI', 'hdpgpc.GPI_model', 'hdpgpc.GPI_HDP',
                         'hdpgpc.emissions._gaussian_process'):
                assert name not in sys.modules, name
        """)

    def test_inference_engines_do_not_initialize_components_or_numerics(self):
        self.run_fresh("""
            import sys
            from types import SimpleNamespace
            from hdpgpc.inference import InferenceEngine, OnlineInference, OfflineInference
            owner = SimpleNamespace()
            for cls in (OnlineInference, OfflineInference):
                engine = cls(owner)
                assert isinstance(engine, InferenceEngine)
                assert engine.model is owner
                assert vars(owner) == {}
            for name in ('numpy', 'torch', 'hdpgpc.GPI_HDP', 'hdpgpc.GPI_model',
                         'hdpgpc.inference._online', 'hdpgpc.inference._offline',
                         'hdpgpc.alignment._amtgp', 'hdpgpc.hdp._variational'):
                assert name not in sys.modules, name
        """)

    def test_hdp_component_does_not_import_emissions_or_model(self):
        self.run_fresh("""
            import sys
            import torch
            before = torch.get_default_dtype()
            from hdpgpc.hdp import HDPVariational
            from types import SimpleNamespace
            state = SimpleNamespace(hmm_switch=True)
            component = HDPVariational(state)
            component.set_hyperparameters('balanced')
            assert state.gamma == 1000.0
            assert torch.get_default_dtype() == before
            for name in ('hdpgpc.GPI_HDP', 'hdpgpc.GPI_model', 'hdpgpc.GPI',
                         'hdpgpc.alignment._amtgp', 'hdpgpc.dynamics._kalman'):
                assert name not in sys.modules, name
        """)

    def test_alignment_policies_do_not_construct_or_import_active_engine(self):
        self.run_fresh("""
            import sys
            import torch
            before = torch.get_default_dtype()
            from hdpgpc.alignment import NoAlignment, GPAlignment
            x = torch.arange(5.)[:, None]
            identity = NoAlignment().create_warper(x)
            assert identity.compute_warp(x, x, x)[2] == 0
            assert not NoAlignment().enabled(True)
            assert GPAlignment().enabled(True)
            assert torch.get_default_dtype() == before
            for name in ('hdpgpc.GPI_HDP', 'hdpgpc.GPI_model',
                         'hdpgpc.alignment._amtgp', 'hdpgpc.alignment._amtgp'):
                assert name not in sys.modules, name
        """)


if __name__ == "__main__":
    unittest.main()
