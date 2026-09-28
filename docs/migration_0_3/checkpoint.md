# Final migration checkpoint

Status: implementation migration finished in npdc-0.3.0. User explicitly stopped
further testing on 2026-09-28. Do not run tests, benchmarks or notebooks unless
the user later authorizes them. See report.md for exact completed/deferred evidence.

Completed: inventory; baseline; native GP/alignment; native state and responsibility
modules; legacy deletion; API/configuration migration; source 221-test pass plus
four architecture tests; 127-function AST equivalence; six exact benchmark hashes;
wheel/sdist build/metadata checks; fresh source install in ../.release-030.

Original npdc/npdc.zip/HDP-GPC are untouched. Local stage commits:
ca8b8cf baseline inventory; 7d02e40 emission/alignment ownership;
5bd27ed native state/orchestration; final deletion/documentation commit follows.

The installed source copy is ../.release-030-source/hdpgpc-0.3.0.
Large baseline executed notebooks are outside the package in
../migration-0.3-evidence/baseline-notebooks; compact summary is baseline_notebooks.json.

Deferred by user: installed execution/regression verification, final 0.3 tutorial
executions and repeat captures. If explicitly authorized in a future task:
1. Use .release-030 Python and tools/verify_installed.py from the unpacked source,
   with a new external evidence directory (no PYTHONPATH, HDPGPC_TEST_INSTALLED=1).
2. Use that same installed Python with tools/validate_tutorials.py, output a new
   folder outside the minimal release; compare scientific reports with baseline.
3. Record results without modifying fixture numbers or tolerances.
No remote release/CI/tag/push was performed. Review declared Python/OS matrix before
publication. Keep the wheel minimal; test/evidence/tutorial assets are source-only.

Do NOT re-run migration_stage1.py, migration_stage2.py, migration_tests2.py or
migration_finalize.py: all one-shot transformations have already been applied.

Final delivery: ../npdc-0.3.0.zip and its .sha256; wheel/source archive in
 dist/; optional tutorials/bundles/hdpgpc-colab.zip. Archives are construction
 artifacts, not additional verification evidence. Final packaging used a fresh
 build directory. Source ZIP excludes environments, Git internals, build output,
 caches and optional datasets; the Colab bundle carries only record 100.
