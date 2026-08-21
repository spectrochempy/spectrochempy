# %%
# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
NMR: reading TopSpin files (plugin)
=====================================

This example shows how to read Bruker TopSpin NMR files using the
optional ``spectrochempy-nmr`` plugin.

It also compares a simple public SpectroChemPy reconstruction of the raw
TopSpin FID against the bundled TopSpin processed reference spectrum.

Requires the official ``spectrochempy-nmr`` plugin.
Install with: ``pip install spectrochempy[nmr]``.
"""

# %%
import numpy as np

import spectrochempy as scp

# %%
# The recommended public reader lives under the ``scp.nmr`` plugin namespace.
# The longer ``scp.nmr.read_topspin`` and top-level ``scp.read_topspin`` forms
# remain available for compatibility.

fid = scp.nmr.read(
    scp.preferences.datadir / "nmrdata" / "bruker" / "tests" / "nmr" / "topspin_1d",
    expno=1,
    remove_digital_filter=True,
)
print(f"Loaded raw FID: {fid}")
print(f"Shape: {fid.shape}")

# %%
# Plot the raw FID:

_ = fid.plot()

# %%
# Read the processed TopSpin reference spectrum.
topspin = scp.nmr.read(
    scp.preferences.datadir / "nmrdata" / "bruker" / "tests" / "nmr" / "topspin_1d",
    expno=1,
    procno=1,
)
print(f"Loaded processed TopSpin spectrum: {topspin}")
print(f"Shape: {topspin.shape}")

# %%
# Reconstruct the spectrum from the raw FID with the public API.
#
# For this bundled TopSpin example, the processed reference uses:
# - digital-filter correction on the FID;
# - no exponential/sine-bell apodization (`LB=0`, `GB=0`, `SSB=0`);
# - zero filling to `SI=16384`.
#
# The FID metadata also carries TopSpin's stored phasing parameters, and the
# public complex FFT path reuses that convention automatically for this direct
# 1D `QSIM` dataset.
spectrum = fid.fft(size=int(topspin.meta.si[0]))

# %%
# Compare the reconstructed spectrum and the TopSpin processed reference on the
# same ppm grid.
axis = np.asarray(topspin.x.data, dtype=float)
ref = np.asarray(topspin.data).squeeze()
calc = np.asarray(spectrum.data).squeeze()
if axis[0] > axis[-1]:
    axis = axis[::-1]
    ref = ref[::-1]
    calc = calc[::-1]

amplitude_scale = np.vdot(calc, ref) / np.vdot(calc, calc)
maxabs_ratio = np.max(np.abs(ref)) / np.max(np.abs(calc))

ref_norm = ref / np.max(np.abs(ref))
calc_norm = calc / np.max(np.abs(calc))
complex_overlap = np.abs(np.vdot(calc_norm, ref_norm)) / (
    np.linalg.norm(calc_norm) * np.linalg.norm(ref_norm)
)
real_corr = np.corrcoef(calc_norm.real, ref_norm.real)[0, 1]

print(f"Complex overlap with TopSpin reference: {complex_overlap:0.6f}")
print(f"Real-part correlation with TopSpin reference: {real_corr:0.6f}")
print(f"Amplitude scale (calc -> TopSpin): {amplitude_scale.real:0.6f}")
print(f"Amplitude ratio max|TopSpin|/max|calc|: {maxabs_ratio:0.6f}")

# %%
# Overlay the real absorptive part against the TopSpin `1r` reference.
ax = topspin.real.plot(color="black")
_ = spectrum.real.plot(ax=ax, clear=False, color="red")

# %%
# The imaginary parts (`1i`-like channel) can also be compared directly.
ax = topspin.imag.plot(color="black")
_ = spectrum.imag.plot(ax=ax, clear=False, color="red")

# %%
# If the plugin is not installed, the function or method raises a
# :class:`~spectrochempy.plugins.deps.MissingPluginError` with installation
# instructions.

# scp.show()
