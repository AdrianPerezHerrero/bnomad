"""Dataset utilities. Path-dependent research readers remain private."""

__all__ = ('MITBIHRecord', 'align_legacy_annotations',
           'extract_annotated_beats', 'extract_wfdb_record',
           'load_mitbih_record', 'DAPHNET_CHANNELS', 'DaphnetRecord', 'DaphnetWindows',
           'daphnet_records', 'download_daphnet', 'load_daphnet_record',
           'window_daphnet_record', 'PreparedDataset', 'archive_datasets',
           'prepare_dataset', 'download_archive', 'CASColonFrames',
           'cas_colon_annotations', 'load_cas_colon_frames', 'CAS_COLON_REGIONS',
           'SwitchingLDSImages', 'generate_switching_lds_images', 'load_switching_lds_images',
           'ZebrafishExpression', 'load_zebrafish_expression', 'download_zebrafish', 'ZEBRAFISH_STAGES',
           'load_xenopus_expression', 'download_xenopus', 'FlutterRecord', 'FlutterWindows',
           'FLUTTER_RECORDS', 'download_flutter', 'load_flutter_record', 'window_flutter_record',
           'XJTUBearingWindows', 'XJTU_RECORDS', 'download_xjtu_sy', 'load_xjtu_sy_windows')


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    if name in ('XJTUBearingWindows', 'XJTU_RECORDS', 'download_xjtu_sy', 'load_xjtu_sy_windows'):
        from . import xjtu_sy
        return getattr(xjtu_sy, name)
    if name in ('FlutterRecord', 'FlutterWindows', 'FLUTTER_RECORDS',
                'download_flutter', 'load_flutter_record', 'window_flutter_record'):
        from . import flutter
        return getattr(flutter, name)
    if name in ('load_xenopus_expression', 'download_xenopus'):
        from . import xenopus
        return getattr(xenopus, name)
    if name in ('ZebrafishExpression', 'load_zebrafish_expression', 'download_zebrafish', 'ZEBRAFISH_STAGES'):
        from . import zebrafish
        return getattr(zebrafish, name)
    if name in ('SwitchingLDSImages', 'generate_switching_lds_images', 'load_switching_lds_images'):
        from . import synthetic_images
        return getattr(synthetic_images, name)
    if name in ('CASColonFrames', 'cas_colon_annotations', 'load_cas_colon_frames', 'CAS_COLON_REGIONS'):
        from . import cas_colon
        return getattr(cas_colon, name)
    if name in ('PreparedDataset', 'archive_datasets', 'prepare_dataset', 'download_archive'):
        from . import ucr
        return getattr(ucr, name)
    if name in ('MITBIHRecord', 'align_legacy_annotations', 'extract_annotated_beats',
                'extract_wfdb_record', 'load_mitbih_record'):
        from . import mitbih
        return getattr(mitbih, name)
    from . import daphnet
    return getattr(daphnet, name)
