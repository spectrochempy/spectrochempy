# ruff: noqa: D205, D400, D415, T201
"""
Inspect despiking corrections and narrow-band preservation
===============================================================

A correction mask records values actually changed, not confirmed cosmic rays.
This deterministic synthetic comparison includes a narrow genuine band, noise,
an isolated spike, spikes near/on the band, and a spike-free control. Signal
creation, noise generation, integration and scoring use the public SpectroChemPy API.
"""

import spectrochempy as scp

# %%
# A known reference and controlled contamination
# ---------------------------------------------------
# The Gaussian has a standard deviation of two samples. The same noise is
# reused in every case, so changes in band area can be compared directly.
# Positions and integration limits below are Raman shifts in cm⁻¹.
x = scp.Coord.linspace(200, 1800, 81, units="cm^-1", title="Raman shift")
band = scp.gaussian(x, ampl=4, pos=1000, width=94.192, normalized=False)
baseline = 10 + 0.003 * scp.arange(81, coordset=[x])
clean = band + baseline
reference = clean + scp.normal(scale=0.02, size=81, seed=42, coordset=[x])
reference.title = "Synthetic Raman intensity"
methods = ("katsumoto", "whitaker", "neighbors")
cases = {"spike-free": None, "isolated": 1500.0, "near band": 1060.0, "on band": 1000.0}
region = slice(800.0, 1200.0)
reference_area = band[region].trapezoid()
results = {}

# %%
# Count changes outside the injected spike and measure band-area bias
# ------------------------------------------------------------------------
# These counts are a conservative modification diagnostic: a method may also
# change neighboring points as part of interpolation. In the spike-free
# control, every changed point is a false positive relative to the injected
# contamination. The threshold of six is illustrative, not a universal choice.
# Area bias is relative to the noiseless band, with the known baseline removed.
print("case | method | changed | outside spike | residual spike | band-area bias (%)")
for case, position in cases.items():
    observed = reference.copy()
    if position is not None:
        observed[position] += 20
    for method in methods:
        corrected, changed = observed.despike(
            method=method, size=9, delta=6, return_mask=True
        )
        count = int(changed.sum())
        outside = (
            count - int(changed[position].sum()) if position is not None else count
        )
        area = (corrected - baseline)[region].trapezoid()
        bias = 100 * (float((area / reference_area).values) - 1)
        residual = (
            float(abs((corrected - reference)[position].values))
            if position is not None
            else 0.0
        )
        print(f"{case} | {method} | {count} | {outside} | {residual:.3f} | {bias:.2f}")
        results[case, method] = observed, corrected, changed

# %%
# Inspect the difficult case on its original coordinates
# -----------------------------------------------------------
# Each output retains the source spectral axis. Plot only points selected by
# the public correction mask to inspect where the algorithm intervened.
observed = results["near band", "katsumoto"][0]
_ = scp.plot_multiple(
    [reference, observed] + [results["near band", method][1] for method in methods],
    method="pen",
    labels=["Spike-free reference", "Contaminated"] + list(methods),
    xlim=(800, 1200),
    title="Narrow band and nearby spike",
    legend="upper left",
)
# %%
# The diagnostic shows both the genuine-band correction and the injected spike.
corrected, changed = results["near band", "neighbors"][1:]
selected = observed.copy()
selected.mask = ~changed.data
_ = scp.plot_multiple(
    [reference, corrected, selected],
    method="scatter",
    marker=["", "", "o"],
    ls=["-", "-", ""],
    color=["black", "blue", "red"],
    labels=["Spike-free reference", "neighbors correction", "Changed values"],
    xlim=(800, 1200),
    title="Inspect changed values; a flag is not proof of a cosmic ray",
    legend="upper left",
)

# %%
# Interpretation and limits
# ------------------------------
# Inspect the table together with the figures. Successful removal of the
# injected spike does not guarantee preservation of the genuine band. In this
# spike-free control, Whitaker changes nine points and biases band area by about
# -31%; neighbors changes the band maximum and biases area by about -2%.
# Katsumoto leaves the control unchanged, but its nearby-spike area bias is
# about +6%. When the spike is superimposed on the band maximum, Whitaker leaves
# its +20 contribution intact in this case: an unflagged value is not evidence
# that it is clean. Katsumoto reduces both the spike and part of the genuine band
# (area bias about -34%). These observations apply only to this signal and these
# parameters.
# Katsumoto keeps its circular averaging window; Whitaker retains endpoint
# candidacy. Their corrections near boundaries therefore need inspection.
# Masked and non-finite samples are preserved. For these two methods, valid
# runs are processed separately and Katsumoto skips runs shorter than its
# window. No method reconstructs a masked gap. On a time-resolved series,
# neighbors along the acquisition axis can preserve persistent spectral
# structure, but may confuse real fast transients with spikes; see the Raman
# acquisition-neighbor example. These synthetic cases do not rank methods
# universally or replace validation on the measurement of interest.
