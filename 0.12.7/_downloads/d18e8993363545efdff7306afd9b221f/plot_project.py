# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Project creation
=======================================

In this example, we create a Project from scratch

"""

# %%

import spectrochempy as scp

# %%
# Create a project with subprojects
# ----------------------------------
proj = scp.Project(
    scp.Project(name="P350", label=r"$\mathrm{M_P}\,(623\,K)$"),
    scp.Project(name="A350", label=r"$\mathrm{M_A}\,(623\,K)$"),
    scp.Project(name="B350", label=r"$\mathrm{M_B}\,(623\,K)$"),
    name="project_1",
    label="main project",
)

proj.projects_names

# %%
# Add datasets to a subproject
# -----------------------------
# Create two datasets:
ir = scp.NDDataset([1.1, 2.2, 3.3], coords=[[1, 2, 3]])
ir

# %%
tg = scp.NDDataset([1, 3, 4], coords=[[1, 2, 3]])
tg

# %%
# Add them to the ``A350`` subproject:
proj.A350["IR"] = ir
proj.A350["TG"] = tg

# %%
# Access project members
# -----------------------
print(proj.A350)
print(proj)
print(proj.A350.label)
print(proj.A350.TG)

# %%
# Save and reload
# ----------------
proj.save()

# %%
newproj = scp.Project.load("project_1", allow_unsafe_legacy=True)
newproj
