"""Versioned, data-only values; no pickle or arbitrary class imports.

Tuple/list and ndarray distinctions are intentional: the legacy constructor
uses them to distinguish shared values from per-cluster options.
"""

import math

KERNELS = frozenset(('ConstantKernel', 'WhiteKernel', 'RBF', 'Matern',
                     'RationalQuadratic', 'ExpSineSquared', 'DotProduct',
                     'Sum', 'Product', 'Exponentiation', 'PairwiseKernel'))


def _encode(value, memo):
    if value is None or type(value) in (str, bool, int):
        return value
    from ._models import MatrixPolicyConfig, NoisePolicyConfig
    from dataclasses import fields
    if type(value) in (MatrixPolicyConfig, NoisePolicyConfig):
        return {'__type__': type(value).__name__,
                'fields': {f.name: encode(getattr(value, f.name), memo) for f in fields(value)}}
    if type(value) is float:
        if not math.isfinite(value):
            return {'__type__': 'float', 'value': repr(value)}
        return value
    if isinstance(value, tuple):
        return {'__type__': 'tuple', 'items': [encode(v, memo) for v in value]}
    if isinstance(value, list):
        return [encode(v, memo) for v in value]
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value) or '__type__' in value:
            raise ValueError('Configuration mappings require string keys; __type__ is reserved')
        return {k: encode(value[k], memo) for k in sorted(value)}
    import numpy as np
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in 'biufU':
            raise TypeError('Unsupported configuration array dtype: ' + str(value.dtype))
        return {'__type__': 'ndarray', 'dtype': value.dtype.str,
                'shape': list(value.shape), 'values': encode(value.tolist(), memo)}
    if isinstance(value, np.generic):
        return {'__type__': 'numpy_scalar', 'dtype': value.dtype.str, 'value': encode(value.item(), memo)}
    from sklearn.gaussian_process.kernels import Kernel
    if isinstance(value, Kernel):
        if type(value).__module__ != 'sklearn.gaussian_process.kernels' or type(value).__name__ not in KERNELS:
            raise TypeError('Custom kernels need an explicit configuration codec')
        return {'__type__': 'kernel', 'class': type(value).__name__,
                'parameters': encode(value.get_params(deep=False), memo)}
    # Tensor options are supported without converting dtype/device to defaults.
    import torch
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.is_complex():
            raise TypeError('Only real dense configuration tensors are supported')
        return {'__type__': 'tensor', 'dtype': str(value.dtype).split('.')[-1],
                'device': str(value.device), 'requires_grad': value.requires_grad,
                'shape': list(value.shape), 'values': encode(value.detach().cpu().tolist(), memo)}
    raise TypeError('Unsupported configuration value: ' + type(value).__name__)


def _decode(value, memo):
    if isinstance(value, list):
        return [decode(v, memo) for v in value]
    if not isinstance(value, dict):
        if value is None or type(value) in (str, bool, int, float):
            return value
        raise TypeError('Expected a data-only configuration value')
    tag = value.get('__type__')
    if tag is None:
        return {k: decode(v, memo) for k, v in value.items()}
    keys = {
        'MatrixPolicyConfig': {'fields'}, 'NoisePolicyConfig': {'fields'},
        'tuple': {'items'}, 'float': {'value'},
        'numpy_scalar': {'dtype', 'value'}, 'ndarray': {'dtype', 'shape', 'values'},
        'kernel': {'class', 'parameters'},
        'tensor': {'dtype', 'device', 'requires_grad', 'shape', 'values'},
    }
    if tag not in keys or set(value) != keys[tag] | {'__type__'}:
        raise ValueError('Unknown or malformed configuration tag: ' + str(tag))
    if tag in ('MatrixPolicyConfig', 'NoisePolicyConfig'):
        from ._models import MatrixPolicyConfig, NoisePolicyConfig
        from dataclasses import fields
        typ = {'MatrixPolicyConfig': MatrixPolicyConfig,
               'NoisePolicyConfig': NoisePolicyConfig}[tag]
        definitions = value['fields']
        if not isinstance(definitions, dict) or set(definitions) != {f.name for f in fields(typ)}:
            raise ValueError('Malformed LDS parameter policy')
        return typ(**{key: decode(item, memo) for key, item in definitions.items()})
    if tag == 'tuple':
        return tuple(decode(v, memo) for v in value['items'])
    if tag == 'float':
        if value['value'] not in ('inf', '-inf', 'nan'):
            raise ValueError('Invalid special float')
        return float(value['value'])
    if tag in ('ndarray', 'numpy_scalar'):
        import numpy as np
        dtype = np.dtype(value['dtype'])
        if dtype.kind not in 'biufU':
            raise ValueError('Unsupported configuration dtype')
        if tag == 'numpy_scalar':
            return dtype.type(decode(value['value'], memo))
        return np.asarray(decode(value['values'], memo), dtype=dtype).reshape(value['shape'])
    if tag == 'kernel':
        if value['class'] not in KERNELS:
            raise ValueError('Unknown kernel class: ' + str(value['class']))
        from sklearn.gaussian_process import kernels
        return getattr(kernels, value['class'])(**decode(value['parameters'], memo))
    import torch
    dtypes = {str(t).split('.')[-1]: t for t in
              (torch.float16, torch.bfloat16, torch.float32, torch.float64,
               torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool)}
    if value['dtype'] not in dtypes:
        raise ValueError('Unsupported tensor dtype')
    return torch.tensor(decode(value['values'], memo), dtype=dtypes[value['dtype']],
                        device=value['device'], requires_grad=value['requires_grad']).reshape(value['shape'])


def encode(value, memo=None):
    """Preserve repeated mutable parameters by stable traversal-local references."""
    if value is None or type(value) in (str, bool, int, float):
        return _encode(value, {} if memo is None else memo)
    if memo is None:
        memo = {}
    key = id(value)
    if key in memo:
        _, ref, active = memo[key]
        if active:
            raise ValueError('Cyclic configuration values are not supported')
        return {'__type__': 'ref', 'id': ref}
    ref = len(memo)
    # Retain objects, including temporary kernel parameter dictionaries, so
    # Python cannot recycle an id during this traversal.
    memo[key] = (value, ref, True)
    result = _encode(value, memo)
    if isinstance(result, list):
        result = {'__type__': 'list', 'items': result}
    elif '__type__' not in result:
        result = {'__type__': 'mapping', 'items': result}
    result['__id__'] = ref
    memo[key] = (value, ref, False)
    return result


def decode(value, memo=None):
    """Decode only known data tags and previously defined shared references."""
    if memo is None:
        memo = {}
    if not isinstance(value, dict):
        return _decode(value, memo)
    if value.get('__type__') == 'ref':
        if set(value) != {'__type__', 'id'} or type(value['id']) is not int or value['id'] not in memo:
            raise ValueError('Invalid configuration reference')
        return memo[value['id']]
    payload = dict(value)
    ref = payload.pop('__id__', None)
    if ref is not None and (type(ref) is not int or ref < 0 or ref in memo):
        raise ValueError('Invalid or duplicate configuration object id')
    if payload.get('__type__') in ('list', 'mapping'):
        if set(payload) != {'__type__', 'items'}:
            raise ValueError('Malformed configuration container')
        if payload['__type__'] == 'list':
            result = [decode(v, memo) for v in payload['items']]
        else:
            result = {k: decode(v, memo) for k, v in sorted(payload['items'].items())}
    else:
        result = _decode(payload, memo)
    if ref is not None:
        memo[ref] = result
    return result
