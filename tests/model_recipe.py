"""Test/benchmark fixture construction through typed configuration only."""
from copy import deepcopy
from dataclasses import fields
from hdpgpc.config import HDPGPCConfig
from hdpgpc.config._models import SECTIONS, NON_CONSTRUCTOR

def recipe_from_options(x_basis=None, **kwargs):
    remaining=dict(kwargs)
    for key, value in dict(gamma=1., trans_alpha=1., start_alpha=.1, kappa=0.,
                           lds_update='conjugate_sequential', lds_scoring='final',
                           birth_lds_kl_per_dimension=False).items():
        remaining.setdefault(key, value)
    root={'x_basis':x_basis, 'verbose':remaining.pop('verbose',False)}
    policy=remaining.pop('alignment',None)
    from hdpgpc.alignment import NoAlignment
    for name,typ in SECTIONS.items():
        options={f.name:remaining.pop(f.name) for f in fields(typ)
                 if f.name not in NON_CONSTRUCTOR.get(name,()) and f.name in remaining}
        if name=='alignment': options['policy']='none' if isinstance(policy,NoAlignment) else 'gp'
        root[name]=typ(**options)
    if remaining: raise TypeError('Unknown fixture options: '+str(sorted(remaining)))
    return HDPGPCConfig(**deepcopy(root))

def build_state(x_basis=None, **kwargs):
    return recipe_from_options(x_basis,**kwargs).build_model()
