# %%
# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Validate 2D Bruker echo-antiecho reconstruction against TopSpin pdata
=====================================================================

This example compares a 2D Bruker raw ``ser`` dataset processed in
SpectroChemPy with the corresponding TopSpin processed reference stored in
``pdata/1``.

It uses the bundled ``exam2d_HC`` dataset, which is an Echo-Antiecho Bruker
case already packaged in ``spectrochempy_data``.  The goal is to validate the
standard reconstruction workflow against the TopSpin reference on a real
dataset that differs from the existing STATES-TPPI example.

Requires the official ``spectrochempy-nmr`` plugin.
Install with: ``pip install spectrochempy[nmr]``.
"""

# %%
# Import API
# ----------
import numpy as np

import spectrochempy as scp
from spectrochempy_nmr import Experiment
from spectrochempy_nmr.processing.hypercomplex import _extract_quaternion_components


DATASET = {
    "label": "Bundled Bruker Echo-Antiecho validation dataset",
    "base": scp.preferences.datadir / "nmrdata" / "bruker" / "tests" / "nmr" / "exam2d_HC",
    "expno": 3,
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


def _normalize_trace(dataset, dim):
    """Return a normalized 1D trace for visual comparisons."""
    trace = dataset.squeeze().copy()
    return trace.normalize(method="max", dim=dim)


# %%
# Locate the bundled dataset
# --------------------------
expdir = DATASET["base"] / str(DATASET["expno"])
if not (expdir / "ser").exists() or not (expdir / "pdata" / "1").exists():
    msg = "The bundled Echo-Antiecho validation dataset was not found."
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
_ = ser.plot_map()
_ = ser[:,0].plot()
ser

# %%
reference

# %%
# Process the raw SER step by step
# --------------------------------
# First, perform only the direct-dimension (F2) processing so we can inspect
# the intermediate spectrum before and after the Echo-Antiecho F2 phase fix.
f2_unphased = ser.em(lb=DATASET["lb_f2"]).fft(size=reference.shape[1])
f2_phased = f2_unphased.pk(phc0=-90.0, phc1=0.0, dim="x", rel=True)
f2_phased[0,:].plot()
# Then continue with the indirect-dimension (F1) processing.
f1_ready = f2_phased.zf_size(size=reference.shape[0], dim="y").em(
    lb=DATASET["lb_f1"], dim="y"
)
reconstructed = f1_ready.fft(dim="y")

# %%
# Inspect the intermediate spectrum after the first F2 transform
# --------------------------------------------------------------
f2_unphased

# %%
f2_phased

# %%
# Plot the first-pass spectrum before the F2 phase correction
_ = f2_unphased.plot_map()

# %%
# Plot the first-pass spectrum after the F2 phase correction
_ = f2_phased.plot_map()

# %%
# Compare the first F2 slice before and after the intermediate phase correction
# ---------------------------------------------------------------------------
first_f2_unphased = f2_unphased[0, :].squeeze().normalize(method="max", dim="x")
first_f2_phased = f2_phased[0, :].squeeze().normalize(method="max", dim="x")
first_f2_unphased.title = "normalized intensity"
first_f2_phased.title = "normalized intensity"

_ = first_f2_unphased.plot(color="black", ylabel="normalized intensity")
_ = first_f2_phased.plot(clear=False, color="blue", linestyle="--")

# %%
# Compare strongest-peak positions
# --------------------------------
scp_peak, scp_y, scp_x = _peak_coords(reconstructed)
ref_peak, ref_y, ref_x = _peak_coords(reference)
print(f"SpectroChemPy peak: index={scp_peak}, y={scp_y:.3f}, x={scp_x:.3f}")
print(f"TopSpin peak:       index={ref_peak}, y={ref_y:.3f}, x={ref_x:.3f}")

# %%
# Inspect the two complex subspectra right before the F1 reconstruction
# ---------------------------------------------------------------------
f1_swapped = f1_ready.swapdims("x", "y")
RR, RI, IR, II = _extract_quaternion_components(f1_swapped.data)
fr = RR + 1j * RI
fi = IR + 1j * II

Fr = np.fft.fftshift(np.fft.fft(np.conjugate(fr)), -1)
Fi = np.fft.fftshift(np.fft.fft(np.conjugate(fi)), -1)

Fr_real = scp.NDDataset(
    np.transpose(Fr.real, (1, 0)),
    coordset=[reference.y.copy(), reference.x.copy()],
    units=reference.units,
    title="Fr.real",
)
Fr_imag = scp.NDDataset(
    np.transpose(Fr.imag, (1, 0)),
    coordset=[reference.y.copy(), reference.x.copy()],
    units=reference.units,
    title="Fr.imag",
)
Fi_real = scp.NDDataset(
    np.transpose(Fi.real, (1, 0)),
    coordset=[reference.y.copy(), reference.x.copy()],
    units=reference.units,
    title="Fi.real",
)
Fi_imag = scp.NDDataset(
    np.transpose(Fi.imag, (1, 0)),
    coordset=[reference.y.copy(), reference.x.copy()],
    units=reference.units,
    title="Fi.imag",
)

# %%
# Plot the intermediate F1 subspectra components around the main peak region
_ = Fr_real.plot_map()

# %%
_ = Fr_imag.plot_map()

# %%
_ = Fi_real.plot_map()

# %%
_ = Fi_imag.plot_map()

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
# Compare the intermediate F1 columns carried by Fr and Fi
# --------------------------------------------------------
fr_real_column = _normalize_trace(Fr_real[:, ref_x], dim="y")
fr_imag_column = _normalize_trace(Fr_imag[:, ref_x], dim="y")
fi_real_column = _normalize_trace(Fi_real[:, ref_x], dim="y")
fi_imag_column = _normalize_trace(Fi_imag[:, ref_x], dim="y")
fr_real_column.title = "normalized intensity"
fr_imag_column.title = "normalized intensity"
fi_real_column.title = "normalized intensity"
fi_imag_column.title = "normalized intensity"

_ = fr_real_column.plot(color="black", ylabel="normalized intensity")
_ = ref_column.plot(clear=False, color="red", linestyle="--")

# %%
_ = fr_imag_column.plot(color="blue", ylabel="normalized intensity")
_ = ref_column.plot(clear=False, color="red", linestyle="--")

# %%
_ = fi_real_column.plot(color="green", ylabel="normalized intensity")
_ = ref_column.plot(clear=False, color="red", linestyle="--")

# %%
_ = fi_imag_column.plot(color="purple", ylabel="normalized intensity")
_ = ref_column.plot(clear=False, color="red", linestyle="--")

# %%
# Optional: simple manual phase touch-up for a closer visual overlay
# ------------------------------------------------------------------
reconstructed_phased = reconstructed.pk(phc0=25.0, phc1=0.0, dim="x", rel=True)

# %%
# Compare the normalized F2 slice again after the manual phase touch-up
scp_slice_phased = reconstructed_phased[slice_y, :].normalize(method="max", dim="x")
scp_slice_phased.title = "normalized intensity"

_ = scp_slice_phased.plot(color="blue", ylabel="normalized intensity")
_ = ref_slice.plot(clear=False, color="red", linestyle="--", xlim=slice_xlim)
