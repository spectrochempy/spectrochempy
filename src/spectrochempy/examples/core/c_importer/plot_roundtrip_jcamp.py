# ruff: noqa: D205, D400, D415
"""
Write and read an infrared spectrum in JCAMP-DX
===============================================

Use JCAMP-DX to exchange an infrared spectrum with other software. This
example writes a real absorbance spectrum on an evenly spaced wavenumber axis.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from packaging.version import Version

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

# %%
# Read a processed NMR spectrum with the same reader
# -------------------------------------------------
# With the optional plugin installed (``pip install 'spectrochempy-nmr>=0.3'``),
# ``scp.read_jcamp`` recognizes the NMR data type declared in the file and
# delegates the read to the plugin. Here is a small synthetic proton spectrum
# with a chemical-shift axis in ppm.

try:
    nmr_version = Version(version("spectrochempy-nmr"))
except PackageNotFoundError:
    nmr_version = None

if nmr_version is not None and nmr_version >= Version("0.3"):
    nmr_jcamp = """##TITLE=Synthetic proton spectrum
##JCAMP-DX=5.01
##DATA TYPE=NMR SPECTRUM
##DATA CLASS=XYDATA
##.OBSERVE FREQUENCY=400
##.OBSERVE NUCLEUS=^1H
##XUNITS=PPM
##YUNITS=ARBITRARY UNITS
##FIRSTX=8
##LASTX=0
##NPOINTS=9
##XFACTOR=1
##YFACTOR=1
##XYDATA=(X++(Y..Y))
8 0 1 2 10 2 1 6 1 0
##END=
"""
    with TemporaryDirectory() as directory:
        filename = Path(directory) / "proton.jdx"
        filename.write_text(nmr_jcamp, encoding="ascii")
        proton = scp.read_jcamp(filename)

    _ = proton.plot()
