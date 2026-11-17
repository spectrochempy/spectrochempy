# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Resolve a sequential reaction with MCR-ALS
==========================================

Fit the UV–visible spectra of a sequential reaction using a kinetic
constraint on the concentration profiles. The data accompany
:cite:t:`bijlsma:2001`.
See the :doc:`MCR-ALS tutorial </userguide/analysis/mcr_als>` for the
rate model and a comparison with a fit without the kinetic constraint.
"""

import spectrochempy as scp
from spectrochempy.analysis import constraints as ct

time_data, spectral_data = scp.read("matlabdata/METING9.MAT")
time = scp.Coord(time_data.data.squeeze(), title="elapsed time", units="s")
wavelength = scp.Coord(
    spectral_data[:, 0].data.squeeze(), title="wavelength", units="nm"
)
spectra = scp.NDDataset(
    spectral_data[:, 1:].data.T,
    coordset=[time.to("min"), wavelength],
    title="absorbance",
)
reaction_spectra = spectra[::4, 300.0:500.0]

# %%
# Absorbance changes as the reaction progresses. Plot a subset of the
# spectra across the wavelength window used for fitting.

_ = reaction_spectra[::5].plot(
    method="lines", title="UV–visible spectra during the reaction"
)

# %%
# Fit concentration profiles constrained by a sequential reaction model.

rates = {"k[0]": 0.5, "k[1]": 0.05}
kinetics = scp.ActionMassKinetics(
    ("A -> B", "B -> C"), {"A": 5.0, "B": 0.0, "C": 0.0}, list(rates.values())
)
initial = kinetics.integrate(reaction_spectra.y.data)
kinetic_mcr = scp.MCRALS(
    constraints=[
        ct.ModelProfile(
            "C",
            components=[0, 1, 2],
            model=kinetics.fit_to_concentrations,
            model_args=([0, 1, 2], [0, 1, 2], rates),
            model_kwargs={
                "ivp_solver_kwargs": {"return_NDDataset": False},
                "optimizer_kwargs": {"options": {"disp": False}},
            },
        )
    ]
)
_ = kinetic_mcr.fit(reaction_spectra, initial)
_ = kinetic_mcr.C.T.plot(
    legend=["A", "B", "C"], title="Kinetically constrained concentration profiles"
)
