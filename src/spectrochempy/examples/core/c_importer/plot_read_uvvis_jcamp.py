# ruff: noqa: D205, D400, D415
"""
Read a UV-visible spectrum from JCAMP-DX
========================================

Read an explicit wavelength and absorbance table from a synthetic UV-visible
spectrum. The wavelength grid has variable spacing around the absorption band.
"""

import spectrochempy as scp

# %%
# Pass a local .jdx path to read_jcamp() for an instrument export.
# This example supplies the same format in memory.
uvvis = b"""##TITLE=Synthetic UV-visible absorption band
##JCAMP-DX=5.01
##DATA TYPE=UV/VIS SPECTRUM
##XUNITS=NANOMETERS
##YUNITS=ABSORBANCE
##FIRSTX=235
##LASTX=310
##NPOINTS=12
##XYPOINTS=(XY..XY)
235,0.03
245,0.08
253,0.22
258,0.45
262,0.69
265,0.79
268,0.74
273,0.53
280,0.24
289,0.10
300,0.04
310,0.02
##END=
"""

spectrum = scp.read_jcamp({"uvvis.jdx": uvvis})
_ = spectrum.plot()
