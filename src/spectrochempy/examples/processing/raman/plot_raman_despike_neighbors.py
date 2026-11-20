# ruff: noqa: D205
"""
Remove transient spikes from time-resolved Raman spectra.
=========================================================

Use adjacent acquisitions to identify isolated positive spikes in operando
Raman spectra on Cu/CeO₂ from the
`Hoyer and Hess dataset <https://doi.org/10.5281/zenodo.13929684>`_.
Documentation builds use the bundled data; standalone runs download the
original archive if the local input file is unavailable.
"""

# %%
# Read the time-resolved spectra
# ------------------------------------------------
# The authors distribute the data under
# `CC BY-SA 4.0 <https://creativecommons.org/licenses/by-sa/4.0/>`_.
# The first row contains Raman shifts. Each subsequent row starts with
# the acquisition time as a fraction of a day.

from io import BytesIO
from zipfile import ZipFile

import numpy as np
import requests

import spectrochempy as scp

# %%
# .. raw:: html
#
#    <details><summary>Read the Raman data</summary>

filename = scp.preferences.datadir / "ramandata/hoyer_hess_2024/Fig_2-a-c_raw-data.txt"
if filename.is_file():
    payload = filename.read_bytes()
else:
    response = requests.get(
        "https://zenodo.org/records/13929684/files/data.zip", timeout=60
    )
    response.raise_for_status()
    with ZipFile(BytesIO(response.content)) as archive:
        payload = archive.read("data/Fig_2-a-c_raw-data.txt")
with BytesIO(payload) as stream:
    shifts = np.fromstring(stream.readline().decode(), sep=" ")
    table = np.loadtxt(stream)

# %%
# .. raw:: html
#
#    </details>

# %%
# Detect transient spikes along the acquisition axis
# -------------------------------------------------------
# Compare spectra along the acquisition axis at each Raman shift.
# Keep a point only as a despiking candidate if it exceeds both immediate
# neighbours by eight robust noise estimates for that channel. Replace
# candidates by the mean of their two neighbours. A narrow Raman band
# that persists over successive acquisitions is retained by this test.
# The threshold is illustrative and may miss spikes spanning several
# acquisitions or flag a real, unusually fast transient.

intensities = table[:, 1:]
raw = scp.NDDataset(intensities, title="Raman intensity")
raw.set_coordset(
    x=scp.Coord(shifts, title="Raman shift", units="cm^-1"),
    y=scp.Coord(
        (table[:, 0] - table[0, 0]) * 86400,
        title="acquisition time",
        units="s",
        rounding=False,
        sigdigits=15,
        linearize_below=0,
    ),
)
cleaned = raw.despike(method="neighbors", dim="y", delta=8)

# %%
# Inspect a corrected spectrum
# ----------------------------
# Locate the acquisition with the largest correction and compare its raw
# and corrected spectra in a window around the affected Raman shift.
# The adjacent acquisitions show the intensities used to estimate the correction.

corrections = abs((raw - cleaned).data)
row, column = np.unravel_index(corrections.argmax(), corrections.shape)
center = float(raw.x.data[column])
window = slice(center - 35.0, center + 35.0)
ax = raw[row - 1, window].plot(label="Previous acquisition", alpha=0.6, show=False)
raw[row + 1, window].plot(ax=ax, label="Next acquisition", alpha=0.6, show=False)
raw[row, window].plot(ax=ax, label="Raw", show=False)
cleaned[row, window].plot(ax=ax, label="Despiked", show=False)
ax.legend()
ax.set_title("Largest correction in the Raman series")
