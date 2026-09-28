"""Run release gates against a noneditable install from an unpacked sdist.

Run outside the development checkout. Logs/results go into a new directory.
This script neither installs dependencies nor regenerates approved baselines.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env['HDPGPC_TEST_INSTALLED'] = '1'
    env['MPLBACKEND'] = 'Agg'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '1'
    steps = [
        ('provenance', ['-I', '-c',
         'import hdpgpc,sys; from pathlib import Path; '
         'p=Path(hdpgpc.__file__).resolve(); print(sys.version); print(p); '
         'assert "site-packages" in p.parts; assert Path(sys.prefix).resolve() in p.parents']),
        ('dependencies', ['-m', 'pip', 'check']),
        ('environment', ['-m', 'pip', 'freeze']),
        ('reference', ['-I', '-m', 'hdpgpc.examples.reference', '--check', '--output', str(output/'reference.json')]),
        *[(f'example_{mode}', ['-I', f'examples/minimal_{mode}.py'])
          for mode in ('online', 'offline', 'static', 'warping', 'multi_output')],
        ('tests', ['-B', '-m', 'unittest', 'discover', '-s', 'tests', '-v']),
        ('determinism', ['-B', '-m', 'tests.characterization.verify_determinism', '--runs', '3', '--report', str(output/'determinism.json')]),
        ('performance', ['tools/performance_smoke.py', '--output', str(output/'performance.json')]),
    ]
    console = Path(sys.executable).parent / ('hdpgpc-reference.exe' if os.name == 'nt' else 'hdpgpc-reference')
    commands = [('console', [str(console), '--check', '--output', str(output/'console_reference.json')])]
    commands.extend((name, [sys.executable, *arguments]) for name, arguments in steps)
    status = {}
    for name, command in commands:
        print('Running '+name, flush=True)
        with (output/(name+'.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=root, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        status[name] = result.returncode
        (output/'status.json').write_text(json.dumps(status, indent=2)+'\n')
        if result.returncode:
            raise SystemExit(f'{name} failed; inspect {output/(name+".log")}')
    print('All installed-package release gates passed', flush=True)


if __name__ == '__main__':
    main()
