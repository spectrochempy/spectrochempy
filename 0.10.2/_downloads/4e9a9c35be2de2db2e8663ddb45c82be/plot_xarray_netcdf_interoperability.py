# %%
# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Interoperability with xarray and NetCDF
=======================================

This example converts a SpectroChemPy dataset to xarray, uses it as a normal
xarray object, saves it as NetCDF, and restores it in SpectroChemPy.
"""

# %%
# Create a small dataset
# ----------------------
# A synthetic dataset keeps this example fast and self-contained. It represents
# three infrared spectra measured on a short wavenumber axis.
from datetime import UTC
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import spectrochempy as scp


wavenumbers = scp.Coord(
    [1800.0, 1760.0, 1720.0, 1680.0, 1640.0, 1600.0],
    name="x",
    title="wavenumber",
    units="cm^-1",
)
samples = scp.Coord(
    [0.0, 1.0, 2.0],
    labels=["reference", "sample A", "sample B"],
    name="y",
    title="sample index",
)

data = np.array(
    [
        [0.10, 0.14, 0.22, 0.18, 0.12, 0.08],
        [0.12, 0.18, 0.31, 0.24, 0.15, 0.10],
        [0.09, 0.16, 0.28, 0.21, 0.13, 0.07],
    ]
)
mask = np.zeros_like(data, dtype=bool)
mask[1, 2] = True

dataset = scp.NDDataset(
    data,
    dims=["y", "x"],
    coordset=[samples, wavenumbers],
    mask=mask,
    units="absorbance",
    name="demo_spectra",
    title="Infrared spectra",
    description="Synthetic spectra for the xarray and NetCDF tutorial.",
    author="SpectroChemPy documentation team",
    origin="synthetic example",
    meta={
        "sample_set": "tutorial",
        "conditions": {"temperature_K": 298.15, "atmosphere": "nitrogen"},
        "replicates": 3,
    },
)
dataset.acquisition_date = datetime(2025, 2, 3, 9, 30, tzinfo=UTC)
dataset.history = "Created for the xarray and NetCDF interoperability tutorial"

# %%
# Inspect the SpectroChemPy object
# --------------------------------
# Coordinates, units, scientific identity, and provenance are all attached to
# the dataset before conversion.
print(dataset)
print("Dimensions:", tuple(dataset.dims))
print("Coordinates:", dataset.coordtitles)
print("Data units:", dataset.units)
print(
    "Selected metadata:",
    {
        "name": dataset.name,
        "title": dataset.title,
        "author": dataset.author,
        "origin": dataset.origin,
        "acquisition_date": dataset.acquisition_date,
    },
)

# %%
# Convert to xarray
# -----------------
# :meth:`~spectrochempy.NDDataset.to_xarray` returns a regular
# :class:`xarray.Dataset`. This makes the data available to tools throughout the
# scientific Python ecosystem while retaining the portable SpectroChemPy
# metadata needed for restoration.
xds = dataset.to_xarray()

print("xarray type:", type(xds))
print("xarray dimensions:", dict(xds.sizes))
print("xarray coordinates:", tuple(xds.coords))
print(
    "Selected portable attrs:",
    {
        key: xds.attrs[key]
        for key in ("scpy_name", "scpy_title", "scpy_author", "scpy_origin")
    },
)

# %%
# Use xarray normally
# -------------------
# For example, xarray can compute a mean spectrum along the sample dimension.
# We keep this derived object separate and use the original exported dataset for
# the round-trip below.
xds_mean = xds.mean(dim="y")
print("Mean spectrum shape:", xds_mean["demo_spectra"].shape)

# %%
# Export to NetCDF and restore
# ----------------------------
# The SpectroChemPy NetCDF method uses the same xarray-backed portable mapping.
# A temporary directory avoids leaving generated files in the source tree.
with TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "spectra.nc"
    dataset.to_netcdf(path)
    restored = scp.NDDataset.from_netcdf(path)

# %%
# Check the portable round-trip
# -----------------------------
# The numerical data, support information, and selected scientific metadata all
# survive. ``created`` and ``modified`` are generated automatically when an
# ``NDDataset`` is created; they do not need to be set manually.
np.testing.assert_allclose(restored.data, dataset.data)
np.testing.assert_allclose(restored.x.data, dataset.x.data)
np.testing.assert_allclose(restored.y.data, dataset.y.data)
np.testing.assert_array_equal(restored.mask, dataset.mask)

assert tuple(restored.dims) == tuple(dataset.dims)
assert str(restored.units) == str(dataset.units)
assert str(restored.x.units) == str(dataset.x.units)
assert restored.x.title == dataset.x.title
assert list(restored.y.labels) == list(dataset.y.labels)
assert restored.name == dataset.name
assert restored.title == dataset.title
assert restored.description == dataset.description
assert restored.author == dataset.author
assert restored.origin == dataset.origin
assert restored.created == dataset.created
assert restored.modified == dataset.modified
assert restored.acquisition_date == dataset.acquisition_date
assert restored.history == dataset.history
assert restored.meta["sample_set"] == dataset.meta["sample_set"]
assert restored.meta["conditions"] == dataset.meta["conditions"]

print(
    "Restored successfully:",
    {
        "dimensions": tuple(restored.dims),
        "units": str(restored.units),
        "mask": bool(restored.is_masked),
        "identity": restored.name,
        "history entries": len(restored.history),
        "metadata sample set": restored.meta["sample_set"],
    },
)

# %%
# What is preserved?
# ------------------
# Portable persistence keeps the scientific subset that is useful across tools
# and languages. It is intentionally narrower than SpectroChemPy's native
# persistence.
#
# .. list-table::
#    :header-rows: 1
#    :widths: 1 2
#
#    * - Portable round-trip
#      - Examples
#    * - Preserved
#      - Dimensions and coordinates, units, masks, name/title/description,
#        author/origin, timestamps, history, JSON-compatible metadata, and
#        supported textual coordinate labels
#    * - Not full native fidelity
#      - Arbitrary Python metadata objects, full ``CoordSet`` topology,
#        result containers, and richer non-portable label structures
#
# Use native SpectroChemPy persistence when exact runtime fidelity is required;
# use xarray and NetCDF when interoperability is the goal.
