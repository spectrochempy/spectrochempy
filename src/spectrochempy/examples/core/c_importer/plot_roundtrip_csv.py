# ruff: noqa: D205, D400, D415
"""
Exchange a spectrum with Excel using CSV
========================================

A CSV file can carry the coordinates and values of a single spectrum. In Excel,
use **Data > From Text/CSV** and select the comma delimiter. To bring edited
values back into SpectroChemPy, save the sheet as CSV and keep the two columns
in the same order. To keep the exported file for Excel, pass a path in your
working directory to ``write_csv``.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import spectrochempy as scp

# %%
# Create a spectrum with one coordinate and one value per row in the CSV file.
x = scp.Coord.linspace(1800.0, 1400.0, 101, title="wavenumber", units="cm^-1")
spectrum = scp.NDDataset(
    scp.gaussian(x.data, pos=1650.0, width=35.0, normalized=False),
    coordset=[x],
    title="absorbance",
    units="absorbance",
)

# %%
# Use an explicit delimiter for both export and import.
with TemporaryDirectory() as directory:
    filename = spectrum.write_csv(Path(directory) / "spectrum.csv", delimiter=",")
    restored = scp.read_csv(filename, csv_delimiter=",")

# %%
# Restore the axis and value units explicitly for the plot.
restored.x.title = "wavenumber"
restored.x.units = "cm^-1"
restored.title = "absorbance"
restored.units = "absorbance"
_ = restored.plot()
