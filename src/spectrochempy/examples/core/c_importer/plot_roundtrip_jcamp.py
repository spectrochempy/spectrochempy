# ruff: noqa: D205, D400, D415
"""
Write and read an infrared spectrum in JCAMP-DX
===============================================

Use JCAMP-DX to exchange an infrared spectrum with other software. This
example writes a real absorbance spectrum on an evenly spaced wavenumber axis.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import spectrochempy as scp

# %%
# The infrared writer expects a y coordinate, even for a single spectrum.
x = scp.Coord.linspace(1800.0, 1400.0, 101, title="wavenumber", units="cm^-1")
y = scp.Coord([0.0])
absorbance = scp.gaussian(x.data, pos=1650.0, width=35.0, normalized=False)
spectrum = scp.NDDataset(
    absorbance.reshape(1, -1), coordset=[y, x], title="absorbance", units="absorbance"
)

# %%
# Write the .jdx file and read it back before removing the temporary directory.
with TemporaryDirectory() as directory:
    filename = spectrum.write_jcamp(Path(directory) / "spectrum.jdx")
    restored = scp.read_jcamp(filename)

_ = restored.plot()
