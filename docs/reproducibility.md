# Reproducible experiments

The reference is a small, explicitly configured scientific regression workload.
Versioned snapshots record intentional inference changes; normal tests never
regenerate their expectations.

The bundled reference uses six five-point waveforms, CPU float64, seed 42,
one PyTorch/BLAS thread and deterministic PyTorch algorithms. It sets Python,
NumPy, PyTorch and available CUDA seeds through `NumericalConfig` at fresh model
construction. The reference runner also limits BLAS thread pools and records the
serialized configuration, dataset SHA-256 and software versions. Some legacy
modules set seeds and the default torch dtype on import; import order and fresh
processes therefore matter. Do not assume a library import preserves global RNG
state. See [implementation specification](implementation_specification.md).

```sh
hdpgpc-reference --check --output run-1.json
hdpgpc-reference --check --output run-2.json
python -m pip freeze > environment.txt
```

Compare `scientific_result` and `data_sha256`, not platform metadata. On the
validated CPU stack repeated runs are exactly equal. The original packaged snapshot came from performance checkpoint `4db9bf1`.
The current version 2 snapshot records the corrected development implementation
and its environment in `hdpgpc/examples/reference_offline.json`.
It protects labels, represented cluster count, observation means/covariances and
latest filtered/smoothed means through public APIs. It supplements the much
larger internal characterization suite; it does not replace it.

The command compares discrete values exactly, and floating values at
`rtol=1e-7, atol=1e-9`, matching the existing characterization tolerance to permit
small floating-point platform differences. Call-order traces retain exact event order, shapes and discrete state; their
finite floats use the same tolerance. Fresh repeated captures must still match
one another byte for byte, including a run with reversed case order. Never regenerate a
reference merely because a test fails. Investigate the first changed quantity.

For maintainers, from the repository or unpacked source archive:

```sh
python -B -m unittest discover -s tests -v
python -B -m tests.characterization.verify_determinism --runs 3 --report determinism.json
```

Set `HDPGPC_TEST_INSTALLED=1` to require tests to use the installed package instead
of inserting the development source directory. The install gate asserts that
`hdpgpc.__file__` lies in `site-packages`. Reviewed baselines live in `tests/characterization/baselines`, making the
source archive self-contained. Their manifest records the source fingerprints. Known online warping/multi-output failure paths remain tested
as failures; packaging does not repair or silently bypass them.

CPU determinism does not establish CUDA determinism or bitwise equality between
different BLAS libraries, hardware or dependency patches. CUDA-specific unit
tests skip when no device is available. Record OS, Python, CPU/GPU, drivers,
thread settings, full dependency versions and the exact configuration for a
publication. The small runner records the principal scientific versions;
`pip freeze` supplies the remaining transitive versions.

In clean-install validation, changing only torch 2.9.1 back to 2.9.0 restored the
first failing exact trace. The Windows wheels report MKL 2025.3 and 2025.2,
respectively. See [release validation](release_validation.md) for the final full
suite and environment evidence. This is dependency/build variability, not
stochastic variation hidden by a newly widened tolerance.

The lightweight performance check uses alternating paired scalar warp calls,
one warmup, seven repetitions and a same-process preserved calculation as its
control. It requires exact outputs and a median runtime ratio at most 1.25.
This intentionally broad CI gate catches large regressions; it is not evidence
for accepting a 25% slowdown or a replacement for the measured studies under
`docs/performance_*.md`. Run it on an idle CPU without concurrent test jobs.


## Reference version 2

The original reference inherited the named `balanced` HDP preset. Subsequent
research changes altered that preset, dimension-scaled proposal scores, Gaussian
likelihood normalization and conjugate LDS updates. Its frozen two-cluster
expectation therefore disagreed with the development model. The installed CI
stopped at `console` because that command runs the same `--check` comparison.

Version 2 pins `gamma=1`, `trans_alpha=1`, `start_alpha=0.1`, `kappa=0`, hard
assignments, unscaled birth KL, the legacy offline search strategy, final LDS
scoring and the corrected `conjugate_sequential` update. Research defaults and
named presets are unchanged. Both waveform families remain separated as
`[0, 0, 1, 1, 0, 1]`; current means, covariances and latent states replace the
historical numerical snapshot. Recipe/data fingerprints now fail explicitly
when a workload changes, and the original float tolerances remain unchanged.
Internal snapshots were reviewed alongside analytic Gaussian and HMM checks;
forced online reuse now has successful-path coverage rather than an obsolete
expected exception.

Reproduce the README illustration using the same checked public experiment:

```sh
python tools/plot_synthetic_reference.py --output synthetic_reference.svg --report synthetic_reference.json
```

The SVG omits timestamps and fixes element IDs. The JSON records the complete
recipe and environment. Existing platform evidence in `docs/release_evidence`
remains historical evidence for its recorded revision, not a claim that the
current development tree has passed every hosted CI platform.
