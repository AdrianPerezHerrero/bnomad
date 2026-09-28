"""Private integration audits against frozen legacy inference trajectories.

Public-only usage lives separately in test_public_api.py. This file deliberately
inspects the backend to establish that input/result adaptation does not change
scientific state or update ordering.
"""

from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.characterization import _support
from tests.model_recipe import recipe_from_options
from hdpgpc.config._models import _initialization_options
from tests.characterization._support import make_model, waveform_data, model_state, encode, reset_runtime, np
from inference_cases import reference, trace_calls, InferenceDivergence, first_difference
from hdpgpc import HDPGPC, HDPGPCConfig

REFERENCE = reference()['cases'] | reference(supplemental=True)['cases']
NAMES = tuple(n for n in REFERENCE if n not in ('reload_labels', 'known_failure_forced_reuse'))


def run_public(name):
    reset_runtime()
    expected = REFERENCE[name]
    outputs = 2 if 'multi' in name else 1
    with patch.object(_support, 'build_state', side_effect=lambda *a, **kw: recipe_from_options(*a, **kw)):
        x, cfg = make_model(outputs=outputs, dynamics='static' if name.endswith('_static') else 'dynamic',
                            use_snr=name == 'offline_multi_snr')
    online = 'online' in name
    cfg = replace(cfg, inference=replace(cfg.inference,
        schedule=('online_fast' if name == 'online_fast' else 'online') if online else 'offline',
        with_warp=name == 'known_failure_online_warp', warp=name == 'offline_warp',
        it_limit=None if online else (1 if name in ('offline_single', 'offline_limit') else 3)))
    data = waveform_data(outputs)
    with redirect_stdout(io.StringIO()), trace_calls(expected['events']):
        public = HDPGPC(cfg)
        if name.startswith('known_failure_'):
            kind = IndexError if outputs == 2 else RuntimeError
            for index, sample in enumerate(data[:2]):
                try:
                    public.update(sample, grid=x)
                except kind as error:
                    result = dict(failed_observation=index, exception_type=type(error).__name__,
                                  message=str(error), partial_state=model_state(public._backend))
                    break
            else:
                raise AssertionError('Legacy failure no longer occurs')
        elif online:
            states = []
            for sample in data[:4]:
                # Additional result reads must neither change the next step nor
                # cause calls to filtering/smoothing/variational update routines.
                public.update(sample, grid=x, include_prediction=True, include_posterior=True)
                states.append(encode(model_state(public._backend)))
            result = {'steps': states}
        else:
            selected = data[:1] if name == 'offline_single' else data
            public.fit(selected, grid=x)
            state = model_state(public._backend)
            if name in ('offline_single', 'offline_limit'):
                result = state
            else:
                # Classification-learning is intentionally still a legacy API;
                # retain this final read-only call to compare the full old trace.
                xs = np.repeat(x[None], len(data), axis=0)
                classification = public._backend.classify_batch(xs, data, learning=False)
                result = {'final': state, 'classification': classification}
        actual = encode(result)
    wanted = {k: v for k, v in expected['result'].items() if k != 'inference_trace'}
    diff = first_difference(actual, wanted, name)
    if diff:
        raise InferenceDivergence(diff)


class PublicFacadeRegressionTests(unittest.TestCase):
    def test_observation_predictions_and_latent_snapshots_copy_legacy_quantities(self):
        reset_runtime()
        with patch.object(_support, 'build_state', side_effect=lambda *a, **kw: recipe_from_options(*a, **kw)):
            x, cfg = make_model(outputs=2)
        cfg = replace(cfg, inference=replace(cfg.inference, it_limit=1))
        with redirect_stdout(io.StringIO()):
            public = HDPGPC(cfg).fit(waveform_data(2))
        before = encode(model_state(public._backend))
        cluster = int(public.labels_[0])
        query = np.linspace(0., 4., 7)[:, None]
        prediction = public.predictive(cluster, grid=query)
        posterior = public.posterior(cluster)
        for ld, output in enumerate(public._backend.gpmodels):
            gp = output[cluster]
            mean, covariance = gp.observe_last(public._backend.cond_to_torch(query))
            np.testing.assert_array_equal(prediction.mean[:, ld], mean.detach().cpu().numpy().reshape(-1))
            np.testing.assert_array_equal(prediction.covariance[ld], covariance.detach().cpu().numpy())
            np.testing.assert_array_equal(posterior[ld].filtered_mean, gp.f_star[-1].detach().cpu().numpy().reshape(-1))
            np.testing.assert_array_equal(posterior[ld].smoothed_covariance, gp.cov_f_sm[-1].detach().cpu().numpy())
        self.assertEqual(encode(model_state(public._backend)), before)


def regression(name):
    def test(self):
        try:
            run_public(name)
        except InferenceDivergence as error:
            self.fail(str(error))
    return test


for name in NAMES:
    setattr(PublicFacadeRegressionTests, 'test_' + name, regression(name))
