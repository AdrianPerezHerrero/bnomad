"""Validate wheel/sdist contents without importing research scripts."""
import argparse
from pathlib import Path
import tarfile
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, nargs='?', default=Path('dist'))
    args = parser.parse_args()
    wheels = list(args.directory.glob('*.whl'))
    sources = list(args.directory.glob('*.tar.gz'))
    assert len(wheels) == len(sources) == 1, 'Use a clean artifact directory'
    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
        for required in ('hdpgpc/__init__.py', 'hdpgpc/examples/reference_offline.json',
                         'hdpgpc/examples/configurations/online.yaml', 'hdpgpc/model/_state.py'):
            assert required in names, required
        for suffix in ('/licenses/LICENSE', '/licenses/NOTICE.md', '/licenses/licenses/bnpy.txt'):
            assert any(name.endswith(suffix) for name in names), suffix
        assert not any('/data/' in n or '/results/' in n or n.endswith('.pyc') for n in names)
    with tarfile.open(sources[0]) as archive:
        names = [n.split('/', 1)[-1] for n in archive.getnames()]
        for required in ('pyproject.toml', 'CITATION.cff', 'tests/test_release.py',
                         'tests/characterization/baselines/manifest.json',
                         'docs/release_checklist.md', 'tools/performance_smoke.py',
                         'tools/verify_installed.py', 'tools/plot_synthetic_reference.py',
                         'docs/assets/synthetic_reference.svg', 'docs/assets/ecg_quickstart.svg',
                         'examples/ecg_quickstart.py', 'docs/ecg_quickstart.md',
                         *(f'examples/minimal_{mode}.py' for mode in
                           ('online', 'offline', 'static', 'warping', 'multi_output'))):
            assert required in names, required
        assert not any(n.startswith('hdpgpc/data/') or n.endswith('.pyc') for n in names)
    print('Wheel and source archive contents passed')


if __name__ == '__main__':
    main()
