"""Execute all notebook cells twice in fresh kernels; audit public imports/results.

Run with an installed hdpgpc and the tutorials extra. No baseline regeneration,
cell skipping, source rewriting, private model reads or mocked model calls.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = {'hdpgpc', 'hdpgpc.config', 'hdpgpc.alignment'}


def audit(notebook):
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        count += 1
        source = cell.source.removeprefix('%%hdpgpc\n')
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and (node.module or '').startswith('hdpgpc'):
                assert node.module in PUBLIC, node.module
                assert not any(n.name.startswith('_') or n.name == 'GPI_HDP' for n in node.names)
            if isinstance(node, ast.Import):
                assert all(not n.name.startswith('hdpgpc.') or n.name in PUBLIC for n in node.names)
            if isinstance(node, ast.Attribute):
                assert not node.attr.startswith('_'), node.attr
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'tutorials/.validation')
    parser.add_argument('--pattern', default='*.ipynb')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ['IPYTHONDIR'] = str(output / '.ipython-host')
    specs = output / 'kernels' / 'validation'
    specs.mkdir(parents=True)
    (specs / 'kernel.json').write_text(json.dumps(dict(
        argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        display_name='Tutorial validation', language='python')), encoding='utf8')
    results = []
    for path in sorted((ROOT / 'tutorials').glob(args.pattern)):
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        source = nbformat.read(path, as_version=4)
        nbformat.validate(source)
        cells = audit(source)
        runs = []
        for repeat in range(2):
            assert hashlib.sha256(path.read_bytes()).hexdigest() == source_hash, 'Notebook changed during validation'
            folder = output / path.stem / str(repeat + 1)
            folder.mkdir(parents=True)
            km = KernelManager(kernel_name='validation', kernel_spec_manager=KernelSpecManager(
                kernel_dirs=[str(specs.parent)], ensure_native_kernel=False))
            notebook = nbformat.read(path, as_version=4)
            client = NotebookClient(notebook, km=km, timeout=600, allow_errors=False,
                                    resources={'metadata': {'path': str(folder)}})
            start = time.perf_counter()
            try:
                client.execute(cwd=str(folder))
            finally:
                if km.has_kernel:
                    km.shutdown_kernel(now=True)
            seconds = time.perf_counter() - start
            nbformat.write(notebook, folder / path.name)
            report = json.loads((folder / 'tutorial-results' / (path.stem + '.json')).read_text())
            scientific = dict(input_sha256=report['input_sha256'], config=report['config'],
                              scientific_result=report['scientific_result'], extra=report['extra'])
            digest = hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()
            runs.append(dict(seconds=seconds, inference_seconds=report['inference_seconds'],
                             scientific_sha256=digest, environment=report['environment']))
            print(path.name, 'run', repeat + 1, round(seconds, 3), 'seconds', flush=True)
        assert runs[0]['scientific_sha256'] == runs[1]['scientific_sha256'], path.name + ': nondeterministic results'
        results.append(dict(notebook=path.name, notebook_sha256=source_hash, code_cells=cells, runs=runs, exact_repeats=True,
                            exceeds_five_minutes=any(r['seconds'] > 300 for r in runs)))
        (output / 'validation.json').write_text(json.dumps(dict(
            platform_scope='Local CPU notebook execution; hosted Colab not exercised',
            public_import_audit=True, results=results), indent=2) + '\n', encoding='utf8')
    assert results, 'No notebooks selected'
    print('All notebooks executed twice; scientific reports identical', flush=True)


if __name__ == '__main__':
    main()
