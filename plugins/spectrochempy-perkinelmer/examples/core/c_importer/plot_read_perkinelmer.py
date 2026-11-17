# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415

"""
Read and plot a PerkinElmer IR spectrum
=======================================

Use the optional ``spectrochempy-perkinelmer`` plugin to read a single-spectrum
``.sp`` file. Install it with ``pip install spectrochempy-perkinelmer``.

See :ref:`import-perkinelmer-ir` for supported files and metadata.
"""

import spectrochempy as scp

spectrum = scp.perkinelmer.read("irdata/perkinelmer/spectra.sp")
_ = spectrum.plot()
