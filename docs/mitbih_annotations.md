# MIT-BIH beat and annotation alignment

The bundled `hdpgpc/data/mitbih` arrays were made by the original
`extract_data.py` and `get_data.py`. They read the **filtered** WFDB records,
retain beat symbols `N L R a A J S e j V E F / f Q ! n`, and extract 90
signal samples per annotation. With the legacy `samples=[60, 150]` setting,
the window is `[annotation_sample - 27, annotation_sample + 63)`. Each
channel has its own mean removed from each beat.

The old reader removed an early annotation and its label together when its
window would begin before the signal. It did **not** remove a label when a
window would end at or beyond the signal length. The saved arrays have no
annotation sample positions. Of the 48 bundled records, eight have one more
label than waveform: `100`, `113`, `115`, `116`, `210`, `212`, `213`, and
`233`. For these files, the extra label is the **last** one. An initial label
must not be dropped to repair this mismatch.

Use the package loader for the archived arrays:

```python
from hdpgpc.datasets import load_mitbih_record

record = load_mitbih_record("hdpgpc/data/mitbih", "100")
waveforms = record.waveforms
labels = record.annotations
assert len(waveforms) == len(labels)
```

`record.omitted_trailing_annotations` reports whether a label was excluded.
The complete-record and autoencoder examples use this loader before fitting;
the notebook helper applies the same rule after downloading its archived
arrays. The original `.npy` files remain untouched.

For new extraction, use `extract_wfdb_record(record_base)` on the **filtered**
WFDB record used by the legacy workflow, or pass an already loaded filtered
signal and its paired annotation samples and symbols to
`extract_annotated_beats`. This keeps sample positions and labels under one
validity mask and returns both in the same order. The WFDB helper requires
the optional `hdpgpc[datasets]` dependency.

Equal-length arrays cannot prove that an annotation is attached to the
correct waveform internally: the saved labels contain no WFDB sample
positions. To investigate a suspected internal shift, re-extract from the
original filtered signal and its `.atr` annotations, then compare the
returned `annotation_samples` with the desired beat positions. The loader
does not guess a shift from waveform shape.
