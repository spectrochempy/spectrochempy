# ruff: noqa: D205, D400, D415
"""
Show complex components and masks in 1D
=======================================

Use ``plot_1D`` to display the real and imaginary components of a spectrum
or highlight a masked interval.
"""

import spectrochempy as scp

x = scp.Coord.linspace(1800.0, 1400.0, 201, title="wavenumber", units="cm^-1")
real = scp.gaussian(x.data, pos=1650.0, width=45.0, normalized=False)
imaginary = scp.gaussian(x.data, pos=1620.0, width=45.0, normalized=False)
spectrum = scp.NDDataset(real + 0.5j * imaginary, coordset=[x], title="intensity")

# %%
# Display both components and zoom in on the absorption bands.
_ = spectrum.plot_1D(show_complex=True, xlim=(1730.0, 1550.0))

# %%
# Mask a wavenumber interval on a copy of the real component. The plot leaves
# a gap at the masked points and shades the excluded interval.
masked = spectrum.real.copy()
masked[1590.0:1550.0] = scp.MASKED
_ = masked.plot_1D(show_mask=True)
