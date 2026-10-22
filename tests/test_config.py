"""Configuration defaults, data-only round trips and exact inference regressions."""

import ast
from contextlib import redirect_stdout
from dataclasses import replace
import inspect
import io
import json
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.characterization import _support
from tests.model_recipe import recipe_from_options
from hdpgpc.config._models import _initialization_options
from tests.characterization._support import (ModelState, make_model, model_state, encode,
                                             np, torch, reset_runtime, waveform_data)
from tests.model_recipe import build_state
from inference_cases import reference, run_case, InferenceDivergence
from hdpgpc.config import (HDPGPCConfig, HDPConfig, GPEmissionConfig, DynamicsConfig,
                           AlignmentConfig, InferenceConfig, MultiOutputConfig, NumericalConfig)
from hdpgpc.config._codec import encode as pack, decode as unpack

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = reference()['cases'] | reference(supplemental=True)['cases']


def recipe(**overrides):
    # Obtain exactly the existing fixture's constructor options without fitting.
    with patch.object(_support, 'build_state', side_effect=lambda *a, **kw: recipe_from_options(*a, **kw)):
        _, config = make_model(**overrides)
    return config


class ConfigurationTests(unittest.TestCase):
    def test_native_construction_requires_a_typed_recipe(self):
        self.assertEqual(tuple(inspect.signature(ModelState).parameters), ('config',))
        self.assertFalse(hasattr(HDPGPCConfig, 'from_legacy_args'))
        self.assertFalse(hasattr(HDPGPCConfig, 'from_legacy_kwargs'))
        self.assertFalse(hasattr(HDPGPCConfig, 'to_legacy_kwargs'))

    def test_dict_and_yaml_roundtrips(self):
        for config in (HDPGPCConfig(), recipe(), recipe(M=2, model_type=['static', 'dynamic'],
                           ini_gamma=[0.03, 0.05], bound_gamma=[(0.001, 0.1), (0.001, 0.2)])):
            with self.subTest(config=config.dynamics.model_type):
                self.assertEqual(HDPGPCConfig.from_dict(config.to_dict()), config)
                restored = HDPGPCConfig.from_yaml(config.to_yaml())
                self.assertEqual(restored, config)
                self.assertEqual(restored.to_yaml(), config.to_yaml())
                self.assertEqual(json.loads(config.to_yaml()), config.to_dict())

    def test_kernel_and_array_types_preserve_values_and_shapes(self):
        from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel, Matern
        kernel = ConstantKernel(1.2, (0.1, 3.)) * RBF(1.5, (0.5, 3.)) + WhiteKernel(.04, (.005, .08))
        config = replace(recipe(), emission=replace(recipe().emission, kernels=[kernel]))
        restored = HDPGPCConfig.from_yaml(config.to_yaml())
        np.testing.assert_array_equal(restored.emission.kernels[0].theta, kernel.theta)
        np.testing.assert_array_equal(restored.emission.kernels[0](config.x_basis), kernel(config.x_basis))
        for value in (np.array([True, False]), np.array(['static', 'dynamic']),
                      np.arange(4, dtype=np.float32).reshape(2, 2), np.float64(.4),
                      torch.tensor([.2, .3], dtype=torch.float32, requires_grad=True),
                      Matern(1.5, (0.5, 3.0)), (1., float('inf'))):
            self.assertEqual(pack(unpack(pack(value))), pack(value))
        self.assertIsInstance(restored.emission.bound_sigma, tuple)
        self.assertEqual(restored.x_basis.dtype, config.x_basis.dtype)

    def test_shared_parameter_objects_survive_serialization(self):
        from sklearn.gaussian_process.kernels import RBF
        grid = np.arange(5.)[:, None]
        kernel = RBF(1.5)
        config = recipe(M=2, x_basis_warp=[grid, grid], kernels=[kernel, kernel])
        restored = HDPGPCConfig.from_yaml(config.to_yaml())
        self.assertIs(restored.emission.kernels[0], restored.emission.kernels[1])
        self.assertIs(restored.alignment.x_basis_warp[0], restored.alignment.x_basis_warp[1])
        legacy = _initialization_options(restored)
        self.assertIs(legacy['kernels'][0], legacy['kernels'][1])
        self.assertIsNot(legacy['kernels'][0], restored.emission.kernels[0])
        shared = [.1, .2]
        self.assertEqual(pack(unpack(pack([shared, shared]))), pack([shared, shared]))

    def test_examples_roundtrip_and_run(self):
        paths = sorted((ROOT/'examples/configurations').glob('*.yaml'))
        self.assertEqual(len(paths), 6)
        for path in paths:
            with self.subTest(example=path.stem):
                config = HDPGPCConfig.from_yaml(path.read_text())
                self.assertEqual(HDPGPCConfig.from_dict(config.to_dict()), config)
                outputs = []
                for restored in (False, True):
                    reset_runtime()
                    model = HDPGPCConfig.from_yaml(config.to_yaml()).build_model() if restored else config.build_model()
                    x, y = config.x_basis, waveform_data(config.multi_output.n_outputs)
                    with redirect_stdout(io.StringIO()):
                        if config.inference.schedule == 'offline':
                            config.inference.run(model, np.repeat(x[None], len(y), axis=0), y)
                        else:
                            for sample in y[:4]:
                                config.inference.run(model, x, sample)
                    outputs.append(encode(model_state(model)))
                self.assertEqual(*outputs)

    def test_policy_roundtrip_and_no_alignment_override(self):
        from hdpgpc.alignment import NoAlignment, GPAlignment
        for policy in (None, GPAlignment(), NoAlignment()):
            config = recipe(alignment=policy)
            restored = HDPGPCConfig.from_yaml(config.to_yaml())
            self.assertEqual(restored, config)
            model = restored.build_model()
            expected = 'none' if type(policy) is NoAlignment else 'gp'
            self.assertEqual('none' if type(model.alignment) is NoAlignment else 'gp', expected)

    def test_factory_preserves_initial_state_and_does_not_mutate_recipe(self):
        reset_runtime()
        config = recipe()
        text = config.to_yaml()
        expected = encode(model_state(config.build_model()))
        reset_runtime()
        actual = HDPGPCConfig.from_yaml(text).build_model()
        self.assertEqual(encode(model_state(actual)), expected)
        self.assertEqual(config.to_yaml(), text)
        self.assertNotIn('config', vars(actual))

    def test_defaults_do_not_reset_runtime(self):
        reset_runtime()
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        before = (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone())
        recipe().numerics.apply()
        self.assertEqual(random.getstate(), before[0])
        np.testing.assert_array_equal(np.random.get_state()[1], before[1][1])
        torch.testing.assert_close(torch.random.get_rng_state(), before[2], rtol=0, atol=0)

    def test_explicit_runtime_settings(self):
        settings = NumericalConfig(seed=19, num_threads=1, dtype='float64', deterministic=True)
        settings.apply()
        first = (random.random(), np.random.rand(), torch.rand(2))
        settings.apply()
        self.assertEqual(random.random(), first[0])
        self.assertEqual(np.random.rand(), first[1])
        torch.testing.assert_close(torch.rand(2), first[2], rtol=0, atol=0)
        reset_runtime()

    def test_validation_rejects_inconsistent_options(self):
        invalid = (
            {'hdp': HDPConfig(M=0)}, {'hdp': HDPConfig(M=3, max_models=2)},
            {'hdp': HDPConfig(hdp_hyp='unknown')}, {'dynamics': DynamicsConfig(model_type='other')},
            {'multi_output': MultiOutputConfig(n_outputs=0)},
            {'multi_output': MultiOutputConfig(n_outputs=2, reduce_outputs=True, reduce_outputs_ratio=.01)},
            {'inference': InferenceConfig(schedule='offline', classify=True)},
            {'inference': InferenceConfig(schedule='online', it_limit=3)},
            {'inference': InferenceConfig(it_limit=0)},
            {'numerics': NumericalConfig(dtype='float32')}, {'numerics': NumericalConfig(seed=-1)},
            {'alignment': AlignmentConfig(policy='unknown')},
            {'emission': GPEmissionConfig(bound_sigma=(1., .5))},
            {'emission': GPEmissionConfig(bound_sigma=[.001, .1])},
            {'hdp': HDPConfig(M=2), 'dynamics': DynamicsConfig(model_type=['static'])},
        )
        for options in invalid:
            with self.subTest(options=options), self.assertRaises((ValueError, TypeError)):
                HDPGPCConfig(**options)
        with self.assertRaisesRegex(ValueError, 'x_basis'):
            HDPGPCConfig().build_model()
        with self.assertRaisesRegex(ValueError, 'ini_sigma'):
            HDPGPCConfig(x_basis=np.arange(5.)[:, None]).build_model()

    def test_unsafe_or_unknown_serialized_values_rejected(self):
        for data in ({'schema_version': 2}, {'schema_version': 1, 'typo': 2},
                     {'schema_version': 1, 'hdp': {'typo': 1}},
                     {'schema_version': 1, 'x_basis': {'__type__': 'pickle', 'value': 'bad'}}):
            with self.assertRaises(ValueError):
                HDPGPCConfig.from_dict(data)
        with self.assertRaises(TypeError):
            recipe_from_options(typo=1)

    def test_all_public_recipes_use_typed_sections(self):
        from hdpgpc.examples import MODES, configuration
        for mode in MODES:
            config=configuration(mode)
            self.assertEqual(config, HDPGPCConfig.from_dict(config.to_dict()))

    def test_configured_schedule_matches_legacy_entry_point(self):
        for name in ('online', 'online_fast', 'offline'):
            config = replace(recipe(), inference=InferenceConfig(schedule=name, warp=False,
                                  it_limit=1 if name == 'offline' else None, n_explore_steps=2))
            actual_config = HDPGPCConfig.from_yaml(config.to_yaml())
            states = []
            for configured in (False, True):
                reset_runtime()
                model = actual_config.build_model()
                x, y = config.x_basis, waveform_data()
                with redirect_stdout(io.StringIO()):
                    if name == 'offline':
                        xs = np.repeat(x[None], len(y), axis=0)
                        if configured: actual_config.inference.run(model, xs, y)
                        else: model.offline.fit(xs, y, it_limit=1)
                    else:
                        for sample in y[:2]:
                            if configured: actual_config.inference.run(model, x, sample)
                            else: getattr(model.online, 'update_fast' if name == 'online_fast' else 'update')(x, sample, warp=False)
                states.append(encode(model_state(model)))
            self.assertEqual(*states, name)


class SerializedConfigurationCharacterizationTests(unittest.TestCase):
    """Real serialized construction, compared to all frozen inference boundaries."""


def characterization(name):
    def test(self):
        def construct(*args, **kwargs):
            config = recipe_from_options(*args, **kwargs)
            return HDPGPCConfig.from_yaml(config.to_yaml()).build_model()
        with patch.object(_support, 'build_state', side_effect=construct):
            try:
                run_case(name, REFERENCES[name])
            except InferenceDivergence as error:
                self.fail(str(error))
    return test


for name in REFERENCES:
    setattr(SerializedConfigurationCharacterizationTests, 'test_' + name, characterization(name))
