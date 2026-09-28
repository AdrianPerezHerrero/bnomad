"""Bundle one current wheel, tutorial helpers and only the 3.3 MB record 100.

Build a wheel first with python -m build. No inference code is transformed.
The archive is an uploadable Colab transport, not a published package release.
"""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    wheels = list((ROOT / 'dist').glob('*.whl'))
    assert len(wheels) == 1, 'Build one current wheel under dist first'
    files = {wheels[0].name: wheels[0]}
    for path in (ROOT / 'tutorials').glob('*.ipynb'):
        files[path.relative_to(ROOT).as_posix()] = path
    for path in (ROOT / 'tutorials').glob('*.md'):
        files[path.relative_to(ROOT).as_posix()] = path
    for path in (ROOT / 'tutorials').glob('*.json'):
        files[path.relative_to(ROOT).as_posix()] = path
    for rel in ('examples/colab_setup.py', 'examples/tutorial_utils.py',
                'hdpgpc/data/mitbih/100.npy', 'hdpgpc/data/mitbih/100_labels.npy',
                'LICENSE', 'NOTICE.md'):
        files[rel] = ROOT / rel
    manifest = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    target = ROOT / 'tutorials/bundles/hdpgpc-colab.zip'
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items(): archive.write(path, name)
        archive.writestr('manifest.json', json.dumps(manifest, indent=2) + '\n')
    print(target, target.stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
