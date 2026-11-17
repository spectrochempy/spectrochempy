# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Resolve overlapping chromatographic profiles
============================================

Resolve four components in an HPLC-DAD run using reference spectra as
the initial estimate. The data accompany :cite:t:`jaumot:2005,jaumot:2015`.
See the :doc:`MCR-ALS tutorial </userguide/analysis/mcr_als>` for
initialization, constraints, and reconstruction diagnostics.
"""

import spectrochempy as scp

datasets = scp.read("matlabdata/als2004dataset.MAT")
by_name = {dataset.name: dataset for dataset in datasets}
chromatogram = by_name["m1"]
chromatogram.title = "absorbance"

# %%
# Each row contains the spectrum at one elution step. The map shows
# where the detector signal varies over the run.

_ = chromatogram.plot_image(
    title="Measured HPLC-DAD run",
    xlabel="Spectral channel",
    ylabel="Elution index",
    zlabel="Absorbance",
    colorbar=True,
)

# %%
# Estimate the component spectra from the measured matrix.

model = scp.MCRALS()
_ = model.fit(chromatogram, by_name["spure"])
result = model.result
profiles = result.C.T
spectra = result.components
profiles.set_coordset(None, scp.Coord.arange(profiles.shape[-1], title="elution index"))
spectra.set_coordset(
    None, scp.Coord.arange(spectra.shape[-1], title="spectral channel")
)
profiles.title = "relative concentration"
spectra.title = "relative intensity"
_ = scp.multiplot(
    [profiles, spectra],
    labels=["Elution profiles", "Resolved spectra"],
    ncol=2,
    method="pen",
    legend=["1", "2", "3", "4"],
    figsize=(10, 4),
)
