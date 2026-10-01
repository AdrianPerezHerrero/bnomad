"""Public input/result adaptation over the native online and offline engines."""

from __future__ import annotations

from tqdm.auto import tqdm
from copy import deepcopy
from numbers import Integral

from ..config import HDPGPCConfig
from ._results import Prediction, LatentState, UpdateResult, WarpPosterior, _snapshot


def _data(value, name):
    """Normalize the new array-like API to independent finite CPU float64 data."""
    import numpy as np
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype.kind not in 'biuf' or not np.all(np.isfinite(array)):
        raise ValueError(name + ' must contain finite real numeric values')
    return np.array(array, dtype=np.float64, copy=True)


class HDPGPC:
    """Fit waveform batches or stream observations with a typed configuration.

    fit(X, grid=None): X is (samples, points[, outputs]); returns self.
    update(x, grid=None): x is (points[, outputs]); returns UpdateResult.
    Default coordinates come from config.x_basis when it is one shared grid.
    An explicit schedule is honored; None chooses offline for fit and standard
    online for update. fit always starts a fresh model. Offline-to-online mixing
    is rejected on this facade; advanced operations use explicit inference engines.
    """

    def __init__(self, config: HDPGPCConfig):
        if not isinstance(config, HDPGPCConfig):
            raise TypeError('config must be an HDPGPCConfig')
        self._config = deepcopy(config)
        self._config.validate(for_build=True)
        self._backend = None
        self._mode = None
        self._has_data = False
        self._failed = False

    @property
    def config(self) -> HDPGPCConfig:
        """An independent copy of the construction recipe."""
        return deepcopy(self._config)

    def _require_fitted(self):
        if self._failed:
            raise RuntimeError('The last inference call failed; call fit() for a fresh model')
        if not self._has_data:
            raise RuntimeError('Call fit() or update() before requesting fitted results')

    def _default_grid(self):
        import numpy as np
        basis = self._config.x_basis
        if isinstance(basis, list):
            if not all(np.array_equal(basis[0], other) for other in basis[1:]):
                raise ValueError('Different per-cluster bases require an explicit observation grid')
            basis = basis[0]
        return basis

    def _inputs(self, values, grid, *, batch):
        import numpy as np
        data = _data(values, 'X' if batch else 'x')
        if batch and data.ndim == 2:
            data = data[:, :, None]
        elif not batch and data.ndim == 1:
            data = data[:, None]
        expected_ndim = 3 if batch else 2
        if data.ndim != expected_ndim or any(size == 0 for size in data.shape):
            raise ValueError('Expected nonempty (samples, points[, outputs])' if batch
                             else 'Expected nonempty (points[, outputs])')
        if data.shape[-1] != self._config.multi_output.n_outputs:
            raise ValueError('Observation output count differs from config.multi_output.n_outputs')
        points = data.shape[-2]
        coordinates = _data(self._default_grid() if grid is None else grid, 'grid')
        if coordinates.ndim == 1:
            coordinates = coordinates[:, None]
        if batch and coordinates.ndim == 2 and coordinates.shape == (points, 1):
            coordinates = np.repeat(coordinates[None], len(data), axis=0)
        expected = (len(data), points, 1) if batch else (points, 1)
        if coordinates.shape != expected:
            raise ValueError(f'grid must have shape {expected} (or a shared (points, 1) grid for fit)')
        return data, coordinates

    def fit(self, X, *, grid=None) -> HDPGPC:
        """Fit a fresh model; honor offline or explicitly configured online order."""
        values, coordinates = self._inputs(X, grid, batch=True)
        inference = self._config.inference
        if inference.classify:
            raise ValueError('fit/update require classify=False; use OfflineInference for classification')
        backend = self._config.build_model()
        self._backend = backend
        self._has_data = False
        self._failed = False
        self._mode = inference.schedule or 'offline'
        try:
            if self._mode == 'offline':
                backend.offline.fit(coordinates, values, it_limit=inference.it_limit, warp=inference.warp)
            else:
                samples = zip(coordinates, values)
                if not backend.verbose:
                    samples = tqdm(samples, total=len(values), desc='Online fit', unit='sample')
                for x, y in samples:
                    self._include(x, y)
        except BaseException:
            # The original implementation may already have mutated its state.
            # Preserve the exception; never present stale successful summaries.
            self._failed = True
            raise
        self._has_data = True
        return self

    def _include(self, grid, values):
        inference = self._config.inference
        method = (self._backend.online.update_fast if self._mode == 'online_fast'
                  else self._backend.online.update)
        return method(grid, values, with_warp=inference.with_warp,
                      force_model=inference.force_model, minibatch=inference.minibatch,
                      classify=inference.classify)

    def update(self, x, *, grid=None, include_prediction=False, include_posterior=False) -> UpdateResult:
        """Commit one observation; optional summaries are computed after the commit."""
        import numpy as np
        if type(include_prediction) is not bool or type(include_posterior) is not bool:
            raise TypeError('include_prediction/include_posterior must be boolean')
        if self._failed:
            self._require_fitted()
        inference = self._config.inference
        if inference.schedule == 'offline' or self._mode == 'offline':
            raise ValueError('update cannot continue an offline fit; select online/online_fast on a fresh model')
        if inference.classify:
            raise ValueError('fit/update require classify=False; use OfflineInference for classification')
        values, coordinates = self._inputs(x, grid, batch=False)
        if self._backend is None:
            self._backend = self._config.build_model()
            self._mode = inference.schedule or 'online'
        before = self._backend.M
        try:
            self._include(coordinates, values)
        except BaseException:
            self._failed = True
            raise
        self._has_data = True
        # Do not read resp_last: standard online does not refresh that cache.
        assigned = int(self._backend.resp_assigned[-1][-1].item())
        if not 0 <= assigned < self._backend.M:
            raise RuntimeError('Legacy inference returned an out-of-range assignment')
        hard = np.zeros(self._backend.M, dtype=np.float64)
        hard[assigned] = 1.0
        return UpdateResult(
            observation_index=int(self._backend.T - 1), assigned_cluster=assigned,
            cluster_ids=_snapshot(np.arange(self._backend.M, dtype=np.int64)),
            responsibilities=_snapshot(hard), new_cluster=bool(self._backend.M > before),
            prediction=self.predictive(assigned, grid=coordinates) if include_prediction else None,
            posterior=self.posterior(assigned) if include_posterior else None)

    @property
    def labels_(self):
        """Latest committed hard labels, in current cluster numbering, as a copy."""
        import numpy as np
        self._require_fitted()
        backend = self._backend
        if backend.resp_assigned:
            labels = _snapshot(backend.resp_assigned[-1])
            if labels.shape == (backend.T,):
                return labels
            raise RuntimeError('Legacy inference did not retain one label per observation')
        # A one-sample offline fit exits before its history append. Report the
        # unique component that actually included it, without a new inference pass.
        if backend.T == 1:
            assigned = [m for m, gp in enumerate(backend.gpmodels[0]) if 0 in gp.indexes]
            if len(assigned) == 1:
                return _snapshot(np.array(assigned, dtype=np.int64))
        raise RuntimeError('Legacy inference did not retain unambiguous fitted labels')

    @property
    def cluster_ids_(self):
        """Cluster indices represented in labels_; empty reserve slots are omitted."""
        import numpy as np
        return _snapshot(np.unique(self.labels_))

    @property
    def n_clusters_(self) -> int:
        """Number of represented clusters, excluding unassigned reserve slots."""
        return len(self.cluster_ids_)

    def _cluster(self, cluster):
        self._require_fitted()
        if isinstance(cluster, bool) or not isinstance(cluster, Integral) or cluster < 0:
            raise ValueError('cluster must be a nonnegative integer from cluster_ids_')
        cluster = int(cluster)
        if cluster not in self.cluster_ids_:
            raise ValueError('cluster is not represented in the current labels')
        if any(cluster >= len(output) for output in self._backend.gpmodels):
            raise RuntimeError('Legacy inference did not retain the requested component')
        return cluster

    def predictive(self, cluster, *, grid=None) -> Prediction:
        """Read the current observation distribution, not a forecast/label predictor.

        Default coordinates are this cluster's first-output GP basis. Supplied
        coordinates are model/aligned coordinates; no inverse warp is performed.
        """
        import numpy as np
        cluster = self._cluster(cluster)
        models = [output[cluster] for output in self._backend.gpmodels]
        coordinates = _data(models[0].x_basis if grid is None else grid, 'grid')
        if coordinates.ndim == 1:
            coordinates = coordinates[:, None]
        if coordinates.ndim != 2 or coordinates.shape[1] != 1 or not len(coordinates):
            raise ValueError('Prediction grid must be nonempty (points, 1)')
        x = self._backend.cond_cuda(self._backend.cond_to_torch(coordinates))
        predictions = [gp.observe_last(x) for gp in models]
        means = np.column_stack([_snapshot(mean).reshape(-1) for mean, _ in predictions])
        covariances = np.stack([_snapshot(cov) for _, cov in predictions])
        return Prediction(cluster, _snapshot(coordinates), _snapshot(means), _snapshot(covariances))

    def posterior(self, cluster) -> tuple[LatentState, ...]:
        """Copy the latest native-basis filtered/smoothed state for each output."""
        cluster = self._cluster(cluster)
        result = []
        for ld, output in enumerate(self._backend.gpmodels):
            gp = output[cluster]
            result.append(LatentState(
                output=ld, observation_index=int(gp.indexes[-1]) if gp.indexes else -1,
                grid=_snapshot(gp.x_basis), filtered_mean=_snapshot(gp.f_star[-1]).reshape(-1),
                filtered_covariance=_snapshot(gp.cov_f[-1]),
                smoothed_mean=_snapshot(gp.f_star_sm[-1]).reshape(-1),
                smoothed_covariance=_snapshot(gp.cov_f_sm[-1])))
        return tuple(result)

    def warp_posterior(self, x, *, grid=None, cluster=None) -> dict[int, WarpPosterior]:
        """Infer a new beat's warp against the current cluster GP representers.

        This is read-only: it neither assigns the beat nor changes the learned
        emission state.  The candidate-conditioned posterior is inferred anew
        for each current cluster.  ``cluster`` selects just one candidate.
        Requires alignment.policy='gp_flow' and an already fitted model.
        """
        self._require_fitted()
        if not getattr(self._backend.alignment, 'uses_posterior_flow', False):
            raise ValueError("warp_posterior requires alignment.policy='gp_flow'")
        values, coordinates = self._inputs(x, grid, batch=False)
        if cluster is not None:
            if (isinstance(cluster, bool) or not isinstance(cluster, Integral)
                    or not 0 <= cluster < len(self._backend.gpmodels[0])):
                raise ValueError('cluster must be a current candidate index')
            cluster = int(cluster)
        import torch
        devices = [torch.cuda.current_device()] if self._backend.cuda else []
        with torch.random.fork_rng(devices=devices):
            results, scores = self._backend.alignment.posterior_next(
                self._backend, coordinates, values, candidate=cluster)
        return {m: WarpPosterior(
            cluster=m, grid_samples=_snapshot(result.grid_samples),
            mean_displacement=_snapshot(result.mean_displacement),
            std_displacement=_snapshot(result.std_displacement),
            mean_whitened=_snapshot(result.mean_whitened),
            chol_whitened=_snapshot(result.chol_whitened),
            score_by_output=_snapshot(scores[m]),
            expected_log_likelihood=float(result.expected_log_likelihood),
            kl=float(result.kl), elbo=float(result.elbo))
            for m, result in results.items()}
