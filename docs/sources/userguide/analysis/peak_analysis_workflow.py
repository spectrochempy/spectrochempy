# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.10.11
# ---

# %% [markdown]
# # Peak Analysis Workflow
#
# This tutorial shows an end-to-end peak-analysis workflow in SpectroChemPy:
#
# 1. prepare a spectrum,
# 2. detect peaks with `find_peaks()`,
# 3. inspect the result through `PeakFindingResult` and `PeakTable`,
# 4. export the table,
# 5. write and validate a fitting script,
# 6. fit the spectrum and inspect the resulting `FitResult`.
#
# It complements the dedicated tutorials on
# [peak finding](peak_finding.py) and [fitting](fitting.py) by focusing on the
# bridge between detection and fitting.

# %%
from pathlib import Path
from tempfile import TemporaryDirectory

import spectrochempy as scp

# %% [markdown]
# ## Load and prepare a spectrum
#
# We build a small synthetic spectrum with two broad OH-like peaks on top of a
# gentle baseline. This keeps the tutorial self-contained while still showing
# the full workflow.

# %%
prefs = scp.preferences
prefs.figure.figsize = (7, 3)

x = scp.Coord.linspace(3700.0, 3300.0, 1200, title="wavenumber", units="cm^-1")
baseline = scp.polynomial(
    x,
    offset=0.015,
    slope=0.00002,
    ampl=1.0,
    c_2=1.5e-6,
)
peak_1 = scp.gaussian(x, ampl=0.95, pos=3624.0, width=42.39, normalized=False)
peak_2 = scp.gaussian(x, ampl=0.32, pos=3542.0, width=51.81, normalized=False)

nd_oh = scp.NDDataset(
    baseline.data + peak_1.data + peak_2.data,
    coordset=[x],
    units="absorbance",
    title="Synthetic OH region",
)
nd_oh_corr = scp.basc(nd_oh)
_ = nd_oh.plot()
_ = nd_oh_corr.plot(clear=False)

# %% [markdown]
# ## Detect peaks and inspect the structured result
#
# `find_peaks(..., as_result=True)` returns a `PeakFindingResult` instead of the
# historical `(peaks, properties)` tuple.  The result keeps the detected peak
# dataset and exposes a stable tabular view through `result.table`.

# %%
result = nd_oh_corr.find_peaks(height=0.05, distance="20 cm^-1", as_result=True)
result

# %%
table = result.table
table

# %% [markdown]
# `PeakTable` gives us a dependency-light view of the detected peaks.  The raw
# peak dataset and the raw SciPy-style property dictionary are still available on
# `result`, but the table is often a better starting point for inspection,
# export, and later workflow steps.

# %%
rows = table.to_dict()
rows[:4]

# %% [markdown]
# We can also look at the available table columns:

# %%
table.columns

# %% [markdown]
# ## Export the peak table
#
# `PeakTable.to_csv()` writes a simple CSV file without adding any optional
# dependency such as pandas.

# %%
with TemporaryDirectory() as tmpdir:
    csv_path = Path(tmpdir) / "nh4y-oh-peaks.csv"
    _ = table.to_csv(csv_path)
    preview = "\n".join(csv_path.read_text(encoding="utf-8").splitlines()[:4])

print(preview)

# %% [markdown]
# ## Select starting candidates for fitting
#
# Peak detection gives geometric candidates.  Fitting still requires a modeling
# decision: which peaks do we want to fit, with which line shape, and with which
# bounds?  Here we keep the two strongest detected peaks and use their positions
# as initial guesses in a manually written fitting script.

# %%
selected_rows = sorted(
    rows,
    key=lambda row: float(row["height"].magnitude),
    reverse=True,
)[:2]
selected_rows = sorted(
    selected_rows,
    key=lambda row: float(row["position"].to("cm^-1").magnitude),
    reverse=True,
)

for row in selected_rows:
    print(
        f"candidate peak at {row['position']:~0.2fP} with height {row['height']:~0.3fP}"
    )

# %%
positions = [float(row["position"].to("cm^-1").magnitude) for row in selected_rows]

script = f"""
COMMON:
$ gratio: 0.1, 0.0, 1.0
$ gasym: 0.1, 0.0, 1.0

MODEL: LINE_1
shape: asymmetricvoigtmodel
    $ ampl:  1.0, 0.0, none
    $ pos:   {positions[0]:.2f}, 3610.0, 3640.0
    > ratio: gratio
    > asym: gasym
    $ width: 200, 0, 1000

MODEL: LINE_2
shape: asymmetricvoigtmodel
    $ ampl:  0.2, 0.0, none
    $ pos:   {positions[1]:.2f}, 3520.0, 3560.0
    > ratio: gratio
    > asym: gasym
    $ width: 200, 0, 1000
"""

print(script)

# %% [markdown]
# ## Validate the script before fitting
#
# `Optimize.validate_script()` lets us check the script before launching the
# optimization.

# %%
opt = scp.Optimize(log_level="INFO")
errors = opt.validate_script(script)
errors

# %% [markdown]
# An empty list means that the script is structurally valid and all referenced
# models are recognized.

# %% [markdown]
# ## Fit the spectrum and inspect the result

# %%
opt.script = script
opt.max_iter = 2000
_ = opt.fit(nd_oh_corr)

# %%
fit_result = opt.result
fit_result

# %% [markdown]
# `FitResult` groups fitted outputs and diagnostics.  The existing estimator
# surface (`opt.predict()`, `opt.components`, plotting helpers, and so on)
# remains available, but `opt.result` is the stable result object for inspection.

# %%
fitted = fit_result.fitted
components = fit_result.components

_ = nd_oh_corr.plot()
ax = components[:].plot(clear=False)
ax.autoscale(enable=True, axis="y")

# %%
_ = opt.plotmerit(offset=0, kind="scatter")

# %% [markdown]
# ## Summary
#
# This workflow now has a clear progression:
#
# - `find_peaks()` detects candidates,
# - `PeakFindingResult` stores the structured detection output,
# - `PeakTable` provides a stable tabular view for inspection and export,
# - `Optimize.validate_script()` checks the fitting DSL before optimization,
# - `FitResult` groups fitted outputs and diagnostics.
#
# This is the current reference workflow when moving from peak detection to peak
# fitting in SpectroChemPy.
