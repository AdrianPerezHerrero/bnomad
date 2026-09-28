# Reproducible experiments

The implementation is the executable specification. No inference mathematics,
optimizer step counts, jitter, update order or model defaults changed in packaging.

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
validated CPU stack repeated runs are exactly equal. The packaged reference
was captured from the unchanged implementation after performance checkpoint
`4db9bf1`, on Python 3.11.0, NumPy 1.26.4, SciPy 1.14.1, torch 2.9.0,
GPyTorch 1.13, linear-operator 0.5.3, Pyro 1.8.6 and scikit-learn 1.5.2.
It protects labels, represented cluster count, observation means/covariances and
latest filtered/smoothed means through public APIs. It supplements the much
larger internal characterization suite; it does not replace it.

The command compares discrete values exactly, and floating values at
`rtol=1e-7, atol=1e-9`, matching the existing characterization tolerance to permit
small floating-point platform differences. Exact call-order regressions in the
full suite retain their stronger existing comparisons. Never regenerate a
reference merely because a test fails. Investigate the first changed quantity.

For maintainers, from the repository or unpacked source archive:

```sh
python -B -m unittest discover -s tests -v
python -B -m tests.characterization.verify_determinism --runs 3 --report determinism.json
```

Set `HDPGPC_TEST_INSTALLED=1` to require tests to use the installed package instead
of inserting the development source directory. The install gate asserts that
`hdpgpc.__file__` lies in `site-packages`. Original baselines were copied into
`tests/characterization/baselines` byte-for-byte, making the source archive
self-contained. Known online warping/multi-output failure paths remain tested
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
