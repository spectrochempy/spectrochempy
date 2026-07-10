# %%
# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
SIMPLISMA example
-----------------
In this example, we perform the PCA dimensionality reduction of a spectra
dataset

"""

# %%
# Import the package
import spectrochempy as scp

# %%
# Load and annotate the dataset
# ------------------------------
print("Dataset (Jaumot et al., Chemometr. Intell. Lab. 76 (2005) 101-110)):")
lnd = scp.read_matlab("matlabdata/als2004dataset.MAT", merge=False)
for mat in lnd:
    print("    " + mat.name, str(mat.shape))

ds = lnd[-1]
_ = ds.plot()

# %%
# Add metadata for a nicer display:
ds.title = "absorbance"
ds.units = "absorbance"
ds.set_coordset(None, None)
ds.y.title = "elution time"
ds.x.title = "wavelength"
ds.y.units = "hours"
ds.x.units = "nm"

# %%
# Fit the SIMPLISMA model
# ------------------------
print("Fit SIMPLISMA on {}\n".format(ds.name))
simpl = scp.SIMPLISMA(n_components=20, tol=0.2, noise=3, log_level="INFO")
_ = simpl.fit(ds)

# %%
# Visualize the results
# ----------------------
# Concentration profiles:
_ = simpl.C.T.plot(title="Concentration")

# %%
# Pure component spectra:
_ = simpl.components.plot(title="Pure profiles")

# %%
# Merit plot after reconstruction:
_ = simpl.plotmerit(offset=0, nb_traces=5)

# %%
# This ends the example ! The following line can be uncommented if no plot shows when
# running the .py script with python

# scp.show()
