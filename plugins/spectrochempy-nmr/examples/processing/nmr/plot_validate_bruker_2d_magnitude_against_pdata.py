# %%
# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Validate 2D Bruker magnitude-mode reconstruction against TopSpin pdata
======================================================================

This example compares a 2D Bruker raw ``ser`` dataset processed in
SpectroChemPy with the corresponding TopSpin processed reference stored in
``pdata/1``.

It uses the bundled ``exam2d_HH`` dataset, which is a Bruker magnitude-mode
case already packaged in ``spectrochempy_data``.  The goal is to validate the
standard reconstruction workflow against the TopSpin reference on a real
dataset that is distinct from the current hypercomplex example family.

Requires the official ``spectrochempy-nmr`` plugin.
Install with: ``pip install spectrochempy[nmr]``.
"""

# %%
# Import API
# ----------
import numpy as np

import spectrochempy as scp
from spectrochempy_nmr import Experiment


DATASET = {
    "label": "Bundled Bruker magnitude-mode validation dataset",
    "base": scp.preferences.datadir / "nmrdata" / "bruker" / "tests" / "nmr" / "exam2d_HH",
    "expno": 1,
    "lb_f2": 2.0,
    "lb_f1": 2.0,
}


def _magnitude(dataset):
    """Return magnitude for complex or quaternion datasets."""
    if dataset.dtype.kind == "V":
        import quaternion  # noqa: PLC0415

        return np.sqrt(np.sum(quaternion.as_float_array(dataset.data) ** 2, axis=-1))
    return np.abs(dataset.data)


def _peak_coords(dataset):
    """Return the coordinates of the strongest peak."""
    mag = _magnitude(dataset)
    idx = np.unravel_index(np.argmax(mag), mag.shape)
    return idx, float(dataset.y.data[idx[0]]), float(dataset.x.data[idx[1]])


# %%
# Locate the bundled dataset
# --------------------------
expdir = DATASET["base"] / str(DATASET["expno"])
if not (expdir / "ser").exists() or not (expdir / "pdata" / "1").exists():
    msg = "The bundled magnitude-mode validation dataset was not found."
    raise FileNotFoundError(msg)

print(DATASET["label"])
print(expdir)

# %%
# Read the raw SER and the TopSpin processed reference
# ----------------------------------------------------
ser = scp.read_topspin(DATASET["base"], expno=DATASET["expno"], remove_digital_filter=True)
reference = scp.read_topspin(DATASET["base"], expno=DATASET["expno"], procno=1)

# %%
# Print a short summary
ser

# %%
reference

# %%
# Process the raw SER in two stages
# ---------------------------------
f2_processed = Experiment(ser).process(
    apodization="em",
    lb=DATASET["lb_f2"],
    size=reference.shape[1],
)
reconstructed = (
    f2_processed.zf_size(size=reference.shape[0], dim="y")
    .em(lb=DATASET["lb_f1"], dim="y")
    .fft(dim="y")
)

# %%
# Compare strongest-peak positions
# --------------------------------
scp_peak, scp_y, scp_x = _peak_coords(reconstructed)
ref_peak, ref_y, ref_x = _peak_coords(reference)
print(f"SpectroChemPy peak: index={scp_peak}, y={scp_y:.3f}, x={scp_x:.3f}")
print(f"TopSpin peak:       index={ref_peak}, y={ref_y:.3f}, x={ref_x:.3f}")

# %%
# Plot the reconstructed spectrum
_ = reconstructed.plot_map()

# %%
# Plot the TopSpin processed reference
_ = reference.plot_map()

# %%
# Compare a normalized F2 slice through the strongest peak
# --------------------------------------------------------
slice_y = ref_y
scp_slice = reconstructed[slice_y, :].normalize(method="max", dim="x")
ref_slice = reference[slice_y, :].normalize(method="max", dim="x")
scp_slice.title = "normalized intensity"
ref_slice.title = "normalized intensity"

_ = scp_slice.plot(color="black", ylabel="normalized intensity")
_ = ref_slice.plot(clear=False, color="red", linestyle="--")
slice_xlim = _.axes.get_xlim()

# %%
# Compare a normalized F1 slice through the strongest peak
# --------------------------------------------------------
scp_column = reconstructed[:, ref_x].squeeze().normalize(method="max", dim="y")
ref_column = reference[:, ref_x].squeeze().normalize(method="max", dim="y")
scp_column.title = "normalized intensity"
ref_column.title = "normalized intensity"

_ = scp_column.plot(color="black", ylabel="normalized intensity")
_ = ref_column.plot(clear=False, color="red", linestyle="--")

# %%
# Optional: compare a lightly phased reconstruction
# -------------------------------------------------
reconstructed_phased = reconstructed.pk(phc0=10.0, phc1=0.0, dim="x", rel=True)

# %%
# Compare the normalized F2 slice again after the manual phase touch-up
scp_slice_phased = reconstructed_phased[slice_y, :].normalize(method="max", dim="x")
scp_slice_phased.title = "normalized intensity"

_ = scp_slice_phased.plot(color="blue", ylabel="normalized intensity")
_ = ref_slice.plot(clear=False, color="red", linestyle="--", xlim=slice_xlim)
