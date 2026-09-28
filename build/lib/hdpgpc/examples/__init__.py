"""Small synthetic research recipes, shipped with the wheel.

These explicit tutorial parameters are not model defaults. Importing examples
does not construct a model or change random seeds.
"""
from importlib.resources import files

MODES = ('online', 'offline', 'static', 'warping', 'multi_output')


def configuration(mode='offline'):
    """Load a packaged, deterministic five-point tutorial recipe."""
    from hdpgpc import HDPGPCConfig
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}')
    name = {'warping': 'dynamic_warping', 'multi_output': 'multi_output'}.get(mode, mode)
    text = files(__package__).joinpath('configurations', name+'.yaml').read_text(encoding='utf-8')
    return HDPGPCConfig.from_yaml(text)


def waveforms(outputs=1):
    """Six deterministic synthetic observations; shape (samples, points, outputs)."""
    import numpy as np
    a = np.array([0., .4, 1., .4, 0.])
    b = np.array([0., -.6, -1.2, -.6, 0.])
    data = np.stack([a, a+.01, b, b-.01, a-.015, b+.015])
    if outputs == 1:
        return data[:, :, None]
    if outputs == 2:
        return np.stack([data, .7*data[:, ::-1]+.02], axis=-1)
    raise ValueError('The tutorial fixture supports one or two outputs')
