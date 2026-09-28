"""GP emissions; implementation modules are private."""
__all__ = ('EmissionModel', 'GaussianProcessEmission')

def __getattr__(name):
    if name == 'EmissionModel':
        from ._interface import EmissionModel
        return EmissionModel
    if name == 'GaussianProcessEmission':
        from ._gaussian_process import GaussianProcessEmission
        return GaussianProcessEmission
    raise AttributeError(name)
