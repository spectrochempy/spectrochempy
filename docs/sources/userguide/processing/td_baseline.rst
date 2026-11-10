Time-domain offset correction
=============================

For NMR FIDs, follow the plugin's
`processing guide <https://spectrochempy-nmr.readthedocs.io/latest/userguide/processing_1d.html>`_
and `FID preparation API <https://spectrochempy-nmr.readthedocs.io/latest/reference/generated/spectrochempy_nmr.Experiment.html#spectrochempy_nmr.Experiment.prepare_fid>`_.

The core :func:`~spectrochempy.dc` operation subtracts a constant estimated
from the end of a time-domain signal.
