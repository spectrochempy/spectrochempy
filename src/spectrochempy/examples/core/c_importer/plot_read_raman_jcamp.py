# ruff: noqa: D205, D400, D415
"""
Read an experimental Raman spectrum from JCAMP-DX
==================================================

Read and plot the Raman spectrum of high-density polyethylene (HDPE).
Horiba Scientific supplied the measurements, distributed by OpenSpecy under
`CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`_.
The `OpenSpecy CSV
<https://github.com/wincowgerDEV/OpenSpecy-package/blob/eb563f48b451126fa7ae69a9b42fb62da724dfe0/inst/extdata/raman_hdpe.csv>`_
was converted to JCAMP-DX without changing its measured pairs.
"""

from importlib.resources import as_file
from importlib.resources import files

import spectrochempy as scp

# %%
# Read the explicit Raman shifts and intensities on the measured grid.
reference = files("spectrochempy.data").joinpath("gallery", "raman_hdpe.dx")
with as_file(reference) as filename:
    spectrum = scp.read_jcamp(filename)

_ = spectrum.plot()
