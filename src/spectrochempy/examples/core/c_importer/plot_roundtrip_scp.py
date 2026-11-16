# ruff: noqa: D205, D400, D415
"""
Save and reload a native SpectroChemPy dataset
==============================================

Use the native SpectroChemPy format (``.scp``) to save an NDDataset with its
coordinates, units and metadata. Galactic SPC files use the ``.spc`` extension
and have a separate reader.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import spectrochempy as scp

# %%
# Create a small spectrum with a calibrated wavenumber axis and a description.
x = scp.Coord.linspace(1800.0, 1400.0, 101, title="wavenumber", units="cm^-1")
spectrum = scp.NDDataset(
    scp.gaussian(x.data, pos=1650.0, width=35.0, normalized=False),
    coordset=[x],
    title="absorbance",
    units="absorbance",
    description="Simulated infrared absorption band",
)

# %%
# Save and reload the dataset. The temporary directory is removed on exit.
with TemporaryDirectory() as directory:
    filename = spectrum.save_as(Path(directory) / "spectrum.scp")
    restored = scp.NDDataset.load(filename)

_ = restored.plot()
