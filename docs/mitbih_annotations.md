# MIT-BIH beat and annotation alignment

The bundled `hdpgpc/data/mitbih` arrays were made by the original
`extract_data.py` and `get_data.py`. They read the **filtered** WFDB records,
retain beat symbols `N L R a A J S e j V E F / f Q ! n`, and extract 90
signal samples per annotation. With the legacy `samples=[60, 150]` setting,
the window is `[annotation_sample - 27, annotation_sample + 63)`. Each
channel has its own mean removed from each beat.

The old reader removed an early annotation and its label together when its
window would begin before the signal. It did **not** remove a label when a
window would end at or beyond the signal length. Most legacy arrays have no
annotation sample positions; corrected record 203 includes them as described
below. Of the 48 bundled records, eight have one more
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
arrays. The loader does not rewrite input files. Record 203 has the explicit
bundled-data correction described below.

For new extraction, use `extract_wfdb_record(record_base)` on the **filtered**
WFDB record used by the legacy workflow, or pass an already loaded filtered
signal and its paired annotation samples and symbols to
`extract_annotated_beats`. This keeps sample positions and labels under one
validity mask and returns both in the same order. The WFDB helper requires
the optional `hdpgpc[datasets]` dependency.

Equal-length arrays alone cannot prove that an annotation is attached to the
correct waveform internally. Legacy files without an annotation-sample sidecar
contain no WFDB sample positions. To investigate a suspected internal shift, re-extract from the
original filtered signal and its `.atr` annotations, then compare the
returned `annotation_samples` with the desired beat positions. The loader
does not guess a shift from waveform shape.

## Corrected record 203

The bundled record 203 now retains **2,979 paired windows and labels** from the
current PhysioNet reference. The legacy extraction used pre-June-2010
annotations. The official [correction log](https://archive.physionet.org/cd-updates/mitdb-corrections-since-1987.txt)
records an added normal beat at sample 318755 (14:45.431), and a deleted normal
beat at sample 418550 (19:22.639), now marked as a QRS-like artifact.

The obsolete window at legacy row 1965 (zero-based) and its label were removed
together. Every retained waveform is unchanged and keeps the current reference
label at its original sample position. The filtered waveform at sample 318755
is not available in the legacy array; it was not synthesized or substituted
with an unfiltered window. This is therefore the 2,979-event common subset,
not a complete re-extraction of all 2,980 current reference beats. Class counts
are N=2528, V=444, a=2, F=1, Q=4.

`203_annotation_samples.npy` stores one original WFDB sample index for each
retained row. `203_annotation_provenance.json` records the reference checksum,
removed and unavailable events, and checksums before and after correction.
The loader reads the optional `<record>_annotation_samples.npy` sidecar and
validates its length and strictly increasing integer positions:

```python
record = load_mitbih_record("hdpgpc/data/mitbih", "203")
assert len(record.waveforms) == len(record.annotations) == len(record.annotation_samples) == 2979
```

New record experiments preserve these positions in `assignments.csv` as
`annotation_sample` and include the sidecar in their data fingerprint. Existing
saved runs retain their old assignments and scores; rerunning against the
corrected inputs creates a different run identity. The loader never shifts
labels by row number or interpolates a missing beat. Records without the
sidecar retain their original behavior.
