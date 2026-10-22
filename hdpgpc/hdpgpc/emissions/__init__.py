"""GP/FA emissions and fixed observation encoders/decoders; implementations are private."""
__all__ = (
    'EmissionModel', 'GaussianProcessEmission',
    'FactorAnalysisEmission', 'FactorAnalysisPosterior',
    'FactorAnalysisStatistics', 'FactorAnalysisMoments',
    'FixedObservationModel', 'AffineGaussianObservation',
    'PCAObservationEstimate', 'estimate_pca_observation',
    'SFAObservationEstimate', 'estimate_sfa_observation',
    'LinearPredictionObservationEstimate', 'estimate_linear_prediction_observation',
    'real_fourier_basis', 'make_fourier_observation',
    'real_wavelet_basis', 'make_wavelet_observation',
    'TiedLinearAutoencoder', 'AutoencoderObservationEstimate', 'estimate_autoencoder_observation',
)

def __getattr__(name):
    if name in ('FactorAnalysisEmission', 'FactorAnalysisPosterior',
                'FactorAnalysisStatistics', 'FactorAnalysisMoments'):
        from . import _factor_analysis
        return getattr(_factor_analysis, name)
    if name in ('LinearPredictionObservationEstimate', 'estimate_linear_prediction_observation'):
        from . import _linear_prediction
        return getattr(_linear_prediction, name)
    if name in ('SFAObservationEstimate', 'estimate_sfa_observation'):
        from . import _sfa
        return getattr(_sfa, name)
    if name in ('TiedLinearAutoencoder', 'AutoencoderObservationEstimate', 'estimate_autoencoder_observation'):
        from . import _autoencoder
        return getattr(_autoencoder, name)
    if name in ('FixedObservationModel', 'AffineGaussianObservation'):
        from . import _fixed_observation
        return getattr(_fixed_observation, name)
    if name in ('PCAObservationEstimate', 'estimate_pca_observation'):
        from . import _pca
        return getattr(_pca, name)
    if name in ('real_fourier_basis', 'make_fourier_observation'):
        from . import _fourier
        return getattr(_fourier, name)
    if name in ('real_wavelet_basis', 'make_wavelet_observation'):
        from . import _wavelet
        return getattr(_wavelet, name)
    if name == 'EmissionModel':
        from ._interface import EmissionModel
        return EmissionModel
    if name == 'GaussianProcessEmission':
        from ._gaussian_process import GaussianProcessEmission
        return GaussianProcessEmission
    raise AttributeError(name)
