# License and attribution

HDP-GPC is distributed under GNU GPL version 3; the existing full license is in
`LICENSE`. Package metadata uses `GPL-3.0-only`; this preparation does not relicense
the code or add an "or later" permission. Copyright remains with the authors.

`hdpgpc/hdpgpc/hdp/_optimizer.py` (formerly `OptimizerRhoOmega.py`) states that code
was extracted and adapted from bnpy. Its attribution comments are retained.
The upstream BSD license is included in `licenses/bnpy.txt`; upstream source:
https://github.com/bnpy/bnpy and https://bnpy.readthedocs.io/.

Dependencies are separate projects with their own licenses. Dataset licenses are
separate from this software license. No MIT-BIH records or other research datasets
are included in built distributions. Download and cite datasets according to their
providers' terms. Synthetic tutorial data are generated locally.

`GPI_test.py` and `extract_data.py` are historical experiment scripts, not safe
import entry points: they can change directories or request local data. They are
retained for compatibility. Use the public API and packaged examples instead.
