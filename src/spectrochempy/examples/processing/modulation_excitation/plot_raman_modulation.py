# ruff: noqa: D205, T201
"""
Raman modulation in the time and phase domains.
===============================================

Compare time-resolved spectra, phase-resolved spectra and Fourier
coefficients from the same operando
Raman experiment on Cu/CeO₂. The raw spectra are from the
`Hoyer and Hess dataset <https://doi.org/10.5281/zenodo.13929684>`_.
"""

# %%
# Read the time-resolved spectra
# ------------------------------------------------
# The authors distribute the data under
# `CC BY-SA 4.0 <https://creativecommons.org/licenses/by-sa/4.0/>`_.
# The first row contains Raman shifts. Each subsequent row starts with
# the acquisition time as a fraction of a day.

import numpy as np

import spectrochempy as scp

# %%
# .. raw:: html
#
#    <details><summary>Download and read the Zenodo archive</summary>

# isort: split
from io import BytesIO
from zipfile import ZipFile

import requests

url = "https://zenodo.org/records/13929684/files/data.zip"
with requests.get(url, timeout=60) as response:
    response.raise_for_status()
    with (
        ZipFile(BytesIO(response.content)) as archive,
        archive.open("data/Fig_2-a-c_raw-data.txt") as stream,
    ):
        shifts = np.fromstring(stream.readline().decode(), sep=" ")
        table = np.loadtxt(stream)

# %%
# .. raw:: html
#
#    </details>

# %%
# Remove transient spikes before grouping cycles
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
# View variations over the average cycle
# ------------------------------------------------
# The experiment alternates reaction and oxidation conditions every 94 s.
# Group the 60 spectra per cycle and omit the first cycle as specified in
# the dataset description. Use the mean elapsed time at each sample index.
# Divide the entire series by a single maximum of the averaged F2g band.
# Subtract the time mean of the average cycle to display its variations.

n_times = 60
cycle_shape = (-1, n_times, cleaned.shape[-1])
cycles = cleaned[n_times:].reshape(cycle_shape, coord_policy="split")
scale = cycles.mean(dim="z")[:, 450.0:470.0].max()
spectra = cycles / scale
spectra.title = "Relative Raman intensity"

average = spectra.mean(dim="z")
variations = average - average.mean(dim="y")
_ = variations.plot_image(
    title="Time domain (colour scale clipped)",
    cmap="RdBu_r",
    symmetric_percentile=99,
    colorbar=True,
)

# %%
# Compare the fundamental and second harmonic
# -------------------------------------------
# For a modulation period T, k=1 probes the fundamental at 1/T and
# k=2 probes the second harmonic at 2/T. The demodulation angle is a
# projection angle: each phase spectrum combines the two quadratures
# at the selected harmonic.
#
# The Riemann PSD gives equal weight to the samples and evaluates its
# reference sine on a right-endpoint phase grid.
#
# The maps have separate colour scales. Use the colour bars to compare
# amplitudes between harmonics.

fundamental = scp.PSD(harmonic=1, integration_rule="riemann").transform(spectra)
second = scp.PSD(harmonic=2, integration_rule="riemann").transform(spectra)

_ = fundamental.prs.plot_contour(
    title="Fundamental (k=1)", cmap="RdBu_r", colorbar=True
)
_ = second.prs.plot_contour(title="Second harmonic (k=2)", cmap="RdBu_r", colorbar=True)

# %%
# Inspect the frequency spectrum with an FFT
# ------------------------------------------
# The acquisition intervals vary. Estimate the period from consecutive
# cycle starts and explicitly supply a uniform grid of step T/60. The
# sample intensities are unchanged. The first sample defines the FFT
# phase origin.

period = np.diff(cycles.z.data).mean()
uniform_time = scp.Coord.arange(
    n_times, title="time within cycle", units="s", rounding=False, sigdigits=15
) * (period / n_times)
uniform = spectra.copy()
uniform.y = uniform_time
fourier = uniform.rfft(dim="y", norm="forward")

# %%
# Check reproducibility across cycle blocks
# -----------------------------------------
# Compare four successive blocks of ten cycles. Average complex coefficients
# across cycles and within each spectral window before extracting amplitudes
# and phases. Use the same normalization divisor and time grid for raw and
# despiked spectra to check the sensitivity to despiking.

raw_cycles = raw[n_times:].reshape(cycle_shape, coord_policy="split") / scale
raw_cycles.y = uniform_time.copy()
raw_fourier = raw_cycles.rfft(dim="y", norm="forward")
windows = [(460.0, 470.0), (500.0, 600.0)]
for low, high in windows:
    print(f"\nRaman shift {low:g}–{high:g} cm⁻¹")
    print("Block   k    Clean amplitude / phase    Raw amplitude / phase")
    for block in range(4):
        selection = slice(10 * block, 10 * (block + 1))
        clean = fourier[selection, 1:3, low:high].mean(dim="z").mean(dim="x")
        untreated = raw_fourier[selection, 1:3, low:high].mean(dim="z").mean(dim="x")
        for k, (clean_coeff, raw_coeff) in enumerate(
            zip(clean.data, untreated.data, strict=True), start=1
        ):
            print(
                f"{block + 1:5d}  {k:2d}    "
                f"{2 * abs(clean_coeff):.3g} / {np.angle(clean_coeff, deg=True):6.1f}°"
                f"             {2 * abs(raw_coeff):.3g} / {np.angle(raw_coeff, deg=True):6.1f}°"
            )

# %%
# Compare the amplitudes and phases across blocks for each band. A harmonic
# can be reproducible across blocks even when its origin is uncertain. An
# independent stimulus measurement would help distinguish stimulus asymmetry
# from a sample response.

# %%
# Plot the first eight nonzero harmonics of both Raman windows.
# The FFT coefficients are divided by the cycle length. Double their
# magnitudes to obtain the single-sided amplitudes away from DC and Nyquist.

harmonics = np.arange(1, 9)
frequencies = fourier.y[:9].to("mHz").data
fundamental_mhz = frequencies[1]
ax = None
width = 0.36
secondary_x = (
    "Modulation frequency (mHz)",
    lambda k: k * fundamental_mhz,
    lambda f: f / fundamental_mhz,
)
for offset, (low, high) in zip((-width / 2, width / 2), windows, strict=True):
    band = fourier[:, 1:9, low:high].mean(dim="z").mean(dim="x")
    amplitude = 2 * abs(band.data)
    bars = scp.NDDataset(amplitude)
    bars.x = scp.Coord(harmonics + offset)
    ax = bars.plot_bar(
        ax=ax,
        show=False,
        data_only=True,
        width=width,
        label=f"{low:g}–{high:g} cm⁻¹",
        secondary_x=secondary_x if offset > 0 else None,
    )

ax.set(
    xlim=(0, 8.6),
    ylim=(0, None),
    xticks=np.arange(9),
    xlabel="Harmonic index (k)",
    ylabel="Relative Raman modulation amplitude",
    title="FFT amplitudes (k=0 mean omitted)",
)
ax.legend()
ax.figure.tight_layout()
ax.figure.subplots_adjust(left=0.17)

# %%
# Compare the response with a nominal square wave
# ------------------------------------------------
# Use an ideal 50% duty-cycle stimulus of unit step height, with the
# reactive half-cycle first. Its even harmonics are zero, so the Bode
# gain and relative phase are undefined there. The even harmonics remain
# measurable in the Raman output, as shown in the preceding FFT plot.
# The stimulus here is nominal. A synchronized measurement of the gas
# composition would be needed to estimate gain and phase relative to
# the actual stimulus.

nominal = scp.NDDataset(np.arange(n_times) < n_times / 2)
nominal.x = uniform_time.copy()
stimulus = nominal.rfft(norm="forward")
odd = harmonics[::2]
odd_frequencies = frequencies[odd]
frequency = scp.Coord(odd_frequencies, title="Modulation frequency", units="mHz")
responses = []
for low, high in windows:
    band = fourier[:, 1:9, low:high].mean(dim="z").mean(dim="x")
    response = band.data[::2] / stimulus.data[odd]
    gain = scp.NDDataset(20 * np.log10(abs(response)))
    gain.x = frequency
    phase_lag = scp.NDDataset(np.angle(response, deg=True))
    phase_lag.x = frequency
    responses.append((gain, phase_lag))

gain_ax, phase_ax = scp.multiplot(
    responses[0], nrow=2, method="scatter", sharex=True, xscale="log", show=False
).flat
for ax, second_trace in zip((gain_ax, phase_ax), responses[1], strict=True):
    ax.lines[0].set_label("460–470 cm⁻¹")
    second_trace.plot_scatter(
        ax=ax, show=False, xscale="log", color="C1", label="500–600 cm⁻¹"
    )

gain_ax.set(title="Response relative to the nominal square wave", ylabel="Gain (dB)")
gain_min = min(gain.data.min() for gain, _ in responses)
gain_max = max(gain.data.max() for gain, _ in responses)
gain_ax.set_ylim(gain_min - 1, gain_max + 1)
gain_ax.legend()
gain_ax.tick_params(labelbottom=False)
phase_ax.set(
    xlabel="Modulation frequency (mHz)",
    ylabel="Relative phase (°)",
    ylim=(-190, 190),
    yticks=[-180, -90, 0, 90, 180],
)
phase_ax.set_xticks(odd_frequencies, labels=[f"{f:.1f}" for f in odd_frequencies])
phase_ax.set_xlim(odd_frequencies[0] / 1.12, odd_frequencies[-1] * 1.12)
for ax in (gain_ax, phase_ax):
    ax.minorticks_off()
    ax.grid(alpha=0.25)
