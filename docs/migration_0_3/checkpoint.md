# 0.3 migration checkpoint

Work ONLY in npdc-0.3.0. npdc, npdc.zip and HDP-GPC remain untouched references.

Completed:
- Inventory and ownership map before production edits (inventory.md/json).
- Baseline: 221 tests, 217 pass/4 skips; all eight notebooks twice, exact repeat reports.
- Stage 1 committed 7d02e40: native emission, prior/training and AMTGP components. 99 numerical + 57 boundary tests pass (one hardware skip).
- Stage 2 committed 5bd27ed: typed native state, cluster lifecycle, HMM messages, objective scoring and output weights. All scientific traces pass. Eight old API assertion errors in its first full run were replaced with six passing engine-contract tests.
- Legacy root modules and old study scripts/notebooks deleted. Removed APIs/codecs/aliases; explicit native alignment policy. Native whole-model engines update/update_fast/fit/classify_batch/fit_labels.
- Full deleted-file source suite: 221 tests in 303.146s, 217 pass/4 skips (native-tests.log).
- Four new architecture tests pass (native-architecture-tests.log); fixture bytes untouched.
- 127 extracted function ASTs match reference after documented renames/docstring exclusion (source_equivalence.json).
- Six workload hashes exactly match; runtime -1.48% to +3.66%, peak RSS difference <3 MB (benchmark_comparison.json).
- 0.3 wheel/sdist build and twine/artifact checks passed. A first sandboxed build failed only because network was denied; approved isolated build succeeded.

Current:
- Installing pinned dependencies plus the unpacked SOURCE archive in a new .release-030 environment (no system-site-packages). See install.log/install-report.json. Source archive unpacked outside checkout at ../.release-030-source/hdpgpc-0.3.0.

Remaining:
1. Finish install; run tools/verify_installed.py from the extracted source using .release-030 Python, output an external fresh evidence directory. This runs console, isolated examples, full installed tests, 3 exact determinism captures and performance smoke sequentially. New suite should be 225 tests, 222 pass/3 CUDA skips when installed.
2. Execute tools/validate_tutorials.py from release checkout with .release-030 Python, output docs/migration_0_3/final-notebooks. Helper now requires hdpgpc 0.3.0. Compare scientific results with baseline notebooks, not just repeat equality.
3. Complete docs/migration_0_3/report.md, release checklist, remaining-reference audit, compact notebook/installed evidence copies. Rebuild final sdist for updated docs (wheel scientific sources unchanged), check artifacts, build Colab bundle, checksum artifacts.
4. Commit final deletion/verification stage and produce standalone npdc-0.3.0.zip without Git/env/build/cache/scratch notebooks; keep old npdc.zip untouched.

Do NOT re-run workspace migration_stage1.py, migration_stage2.py, migration_tests2.py or migration_finalize.py: all have already been applied. They are one-shot audit helpers, not runtime dependencies.
No frozen fixtures/tolerances may change. No inference optimization is part of this release. Preserve known failure cases. Do not claim clean-install/tutorial verification until it finishes.
