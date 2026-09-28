"""Isolated, persistent notebook worker for HDP-GPC; no inference code.

Register %%hdpgpc so Colab's own NumPy/PyTorch need not be replaced or restarted.
Local supported installations are reused in a fresh kernel. Colab always uses a
private Python 3.11 venv provisioned by uv, independently of host ensurepip support.
"""
import atexit
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def start(root):
    from IPython import get_ipython
    from IPython.display import display, clear_output
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager

    root = Path(root).resolve()
    shell = get_ipython()
    previous = shell.user_ns.get('hdpgpc_worker')
    if previous is not None:
        previous.shutdown_kernel(now=True)
    colab = importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
    worker = Path(sys.executable)
    supported = (3, 10) <= sys.version_info[:2] < (3, 13)
    reusable = supported and not colab and os.environ.get('HDPGPC_TUTORIAL_FORCE_ISOLATED') != '1' and subprocess.run(
        [str(worker), '-c', 'from importlib.metadata import version; assert version("hdpgpc") == "0.3.0"; from hdpgpc import HDPGPC; import ipykernel'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if not reusable:
        env_path = root / '.hdpgpc-tutorial-env'
        worker = env_path / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        if not worker.exists():
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'uv>=0.8,<1'], check=True)
            subprocess.run([sys.executable, '-m', 'uv', 'venv', '--python', '3.11',
                            '--seed', str(env_path)], check=True)
        if subprocess.run([str(worker), '-c', 'from importlib.metadata import version; assert version("hdpgpc") == "0.3.0"; from hdpgpc import HDPGPC; import ipykernel'],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
            wheels = list(root.glob('*.whl'))
            package = str(wheels[0]) if wheels else str(root)
            # Small CPU wheel; leave Colab's GPU/PyTorch installation untouched.
            torch_command = [str(worker), '-m', 'pip', 'install', 'torch==2.9.0']
            if sys.platform == 'linux':
                torch_command += ['--index-url', 'https://download.pytorch.org/whl/cpu']
            subprocess.run(torch_command, check=True)
            subprocess.run([str(worker), '-m', 'pip', 'install', package, 'ipykernel>=6.29,<7'], check=True)
    specs = tempfile.TemporaryDirectory(prefix='hdpgpc-kernel-')
    spec_dir = Path(specs.name) / 'hdpgpc'
    spec_dir.mkdir()
    (spec_dir / 'kernel.json').write_text(json.dumps(dict(
        argv=[str(worker), '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        display_name='HDP-GPC CPU', language='python')), encoding='utf8')
    km = KernelManager(kernel_name='hdpgpc', kernel_spec_manager=KernelSpecManager(
        kernel_dirs=[specs.name], ensure_native_kernel=False))
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env['IPYTHONDIR'] = str(Path.cwd() / '.ipython-worker')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    km.start_kernel(cwd=str(Path.cwd()), env=env)
    client = km.client()
    client.start_channels()
    client.wait_for_ready(timeout=120)

    def execute(line, cell):
        msg_id = client.execute(cell, stop_on_error=True)
        deadline = time.monotonic() + 600
        failure = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                km.interrupt_kernel()
                raise TimeoutError('Tutorial cell exceeded 10 minutes')
            message = client.get_iopub_msg(timeout=remaining)
            if message.get('parent_header', {}).get('msg_id') != msg_id:
                continue
            kind, content = message['header']['msg_type'], message['content']
            if kind == 'stream':
                print(content['text'], end='', file=sys.stderr if content['name'] == 'stderr' else sys.stdout)
            elif kind in ('display_data', 'execute_result'):
                display(content['data'], raw=True, metadata=content.get('metadata', {}))
            elif kind == 'clear_output':
                clear_output(wait=content.get('wait', False))
            elif kind == 'error':
                failure = '\n'.join(content['traceback'])
            elif kind == 'status' and content['execution_state'] == 'idle':
                break
        if failure:
            raise RuntimeError(failure)

    shell.register_magic_function(execute, magic_kind='cell', magic_name='hdpgpc')
    shell.user_ns['hdpgpc_worker'] = km
    def close_worker():
        client.stop_channels()
        if km.has_kernel:
            km.shutdown_kernel(now=True)
        specs.cleanup()
    atexit.register(close_worker)
    helper = root / 'examples/tutorial_utils.py'
    execute('', "import runpy\n" + f"utils = runpy.run_path({str(helper)!r})\n" +
            "%matplotlib inline\n" + "import matplotlib.pyplot as plt\nplt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 300, 'axes.spines.top': False, 'axes.spines.right': False})")
    print('HDP-GPC CPU worker:', worker)
    print('Run the remaining %%hdpgpc cells in order; variables persist between cells.')
