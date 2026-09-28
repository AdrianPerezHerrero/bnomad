# Independent 0.3 repository

`npdc-0.3.0` is the separate native release workspace. The package/distribution
name remains `hdpgpc`. The original `npdc` and its ZIP are preserved as 0.2 references.
This working folder has local migration commits. No remote was configured or pushed.

Install with Python 3.10-3.12 (3.11 is the locally validated reference):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\hdpgpc-reference.exe --check --output reference.json
```

For notebook infrastructure, install `.[tutorials]`; see tutorials/README.md.
Build artifacts use `python -m build`. `python tools/build_colab_bundle.py` creates
an uploadable wheel/helper/record-100 bundle after building one wheel under dist.
No manual Python import-path changes are required.

Before publishing, review repository URLs while retaining original attribution,
licenses and citation. Add an empty repository URL with `git remote add origin URL`
and push the reviewed branch yourself. No publication was performed by this task.
See docs/release_checklist.md and docs/migration_0_3/report.md for release status.
