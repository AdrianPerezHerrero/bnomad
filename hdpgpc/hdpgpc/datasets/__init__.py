"""Dataset utilities. Path-dependent research readers remain private."""

__all__ = ('MITBIHRecord', 'align_legacy_annotations',
           'extract_annotated_beats', 'extract_wfdb_record',
           'load_mitbih_record', 'DAPHNET_CHANNELS', 'DaphnetRecord', 'DaphnetWindows',
           'daphnet_records', 'download_daphnet', 'load_daphnet_record',
           'window_daphnet_record', 'PreparedDataset', 'archive_datasets',
           'prepare_dataset', 'download_archive')


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    if name in ('PreparedDataset', 'archive_datasets', 'prepare_dataset', 'download_archive'):
        from . import ucr
        return getattr(ucr, name)
    if name in ('MITBIHRecord', 'align_legacy_annotations', 'extract_annotated_beats',
                'extract_wfdb_record', 'load_mitbih_record'):
        from . import mitbih
        return getattr(mitbih, name)
    from . import daphnet
    return getattr(daphnet, name)
