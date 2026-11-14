# ruff: noqa: D205, T201
"""
Raman modulation in the time and phase domains.
===============================================

Compare time-resolved and phase-resolved spectra from the same operando
Raman experiment on Cu/CeO₂. The raw spectra are from the
`Hoyer and Hess dataset <https://doi.org/10.5281/zenodo.13929684>`_.
"""

# %%
# Read the time-resolved spectra
# ------------------------------------------------
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
# Remove transient spikes and normalize the average cycle
# -------------------------------------------------------
# Compare spectra along the acquisition axis at each Raman shift.
# Keep a point only as a despiking candidate if it exceeds both immediate
# neighbours by eight robust noise estimates for that channel. Replace
# candidates by the mean of their two neighbours. A narrow Raman band
# that persists over successive acquisitions is retained by this test.
# The threshold is illustrative and may miss spikes spanning several
# acquisitions or flag a real, unusually fast transient.

intensities = table[:, 1:]
raw = scp.NDDataset(intensities)
raw.x = scp.Coord(shifts, title="Raman shift", units="cm^-1")
raw.y = scp.Coord(
    table[:, 0], title="acquisition time", units="day", rounding=False
).to("s")
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
cycle_processor = scp.PSD(n_spectra_per_cycle=n_times)
cycles = cycle_processor.group_cycles(cleaned, discard=1)
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
# The maps have separate colour scales. Use the colour bars to compare
# amplitudes between harmonics.

fundamental = scp.PSD(harmonic=1, integration_rule="riemann").transform(spectra)
second = scp.PSD(harmonic=2, integration_rule="riemann").transform(spectra)

_ = fundamental.prs.plot_contour(
    title="Fundamental (k=1)", cmap="RdBu_r", colorbar=True
)
_ = second.prs.plot_contour(title="Second harmonic (k=2)", cmap="RdBu_r", colorbar=True)

# %%
# Check reproducibility across cycle blocks
# -----------------------------------------
# Compare four successive blocks of ten cycles, using the same divisor
# throughout. Average channels within each window before computing the
# quadratures on the same right-endpoint grid as the PSD. The phase in
# the table is the angle of maximum positive PSD. Compare the cleaned
# and raw results to see whether the block estimates depend on despiking.

raw_cycles = cycle_processor.group_cycles(raw, discard=1).data
sample_phase = 2 * np.pi * (np.arange(n_times) + 1) / n_times
for low, high in [(460, 470), (500, 600)]:
    band = (shifts >= low) & (shifts <= high)
    print(f"\nRaman shift {low}–{high} cm⁻¹")
    print("Block   k    Clean amplitude / phase    Raw amplitude / phase")
    for block in range(4):
        selection = slice(10 * block, 10 * (block + 1))
        clean_trace = spectra.data[selection][..., band].mean(axis=(0, 2))
        raw_trace = raw_cycles[selection][..., band].mean(axis=(0, 2)) / scale
        for harmonic in (1, 2):
            kernel = 2 * np.exp(1j * harmonic * sample_phase) / n_times
            clean_coeff = clean_trace @ kernel
            raw_coeff = raw_trace @ kernel
            clean_phase = (90 - np.angle(clean_coeff, deg=True)) % 360
            raw_phase = (90 - np.angle(raw_coeff, deg=True)) % 360
            print(
                f"{block + 1:5d}  {harmonic:2d}    "
                f"{abs(clean_coeff):.3g} / {clean_phase:5.1f}°"
                f"             {abs(raw_coeff):.3g} / {raw_phase:5.1f}°"
            )

# %%
# In the 460–470 cm⁻¹ window, the second harmonic is about 28–34% of the
# fundamental in all four blocks. Its phase of maximum response spans
# about 307–328°, and despiking has little effect on the block estimates.
# In the wider 500–600 cm⁻¹ window, the second harmonic is weaker relative
# to the fundamental and its amplitude is more sensitive to despiking.
# Block reproducibility tests the stability of the measured harmonic.
# Distinguishing stimulus asymmetry from a sample response would require
# an independent measurement of the stimulus.

# %%
# Inspect the frequency spectrum with an FFT
# ------------------------------------------
# Apply a real FFT along the time axis of each cycle. The acquisition
# timestamps vary slightly, so assign a uniform grid based on the mean
# interval between cycle starts. This treats the 60 acquisition indices
# as equally spaced, as in the Riemann PSD above.
#
# Average the complex coefficients across cycles before taking magnitudes
# to keep the contribution that repeats at the same phase.

fourier = cycle_processor.fourier(spectra, grid="cycle_starts")
coefficients = fourier.mean(dim="z").data

# %%
# Plot the first eight nonzero harmonics of both Raman windows.
# The FFT coefficients are divided by the cycle length. Double their
# magnitudes to obtain the single-sided amplitudes away from DC and Nyquist.

harmonics = np.arange(1, 9)
frequencies = fourier.y[:9].to("mHz").data
fundamental_mhz = frequencies[1]
ax = None
width = 0.36
windows = [(460, 470), (500, 600)]
secondary_x = (
    "Modulation frequency (mHz)",
    lambda k: k * fundamental_mhz,
    lambda f: f / fundamental_mhz,
)
for offset, (low, high) in zip((-width / 2, width / 2), windows, strict=True):
    band = (shifts >= low) & (shifts <= high)
    amplitude = 2 * abs(coefficients[1:9, band].mean(axis=1))
    bars = scp.NDDataset(amplitude)
    bars.x = scp.Coord(harmonics + offset)
    ax = bars.plot_bar(
        ax=ax,
        show=False,
        data_only=True,
        width=width,
        label=f"{low}–{high} cm⁻¹",
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
# The FFT uses the first sample as the phase origin. The Riemann PSD
# instead evaluates its sine at right endpoints. Shift the phase by one
# sample and reconstruct the PSD projections at k=1 and k=2.

for harmonic, result in enumerate((fundamental, second), start=1):
    phase = np.deg2rad(result.prs.y.data) + 2 * np.pi * harmonic / n_times
    projections = 2 * (np.exp(1j * phase[:, None]) * coefficients[harmonic].conj()).imag
    error = abs(projections - result.prs.data).max()
    print(f"k={harmonic}: maximum FFT–PSD difference = {error:.2e}")

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

nominal = np.arange(n_times) < n_times / 2
stimulus = np.fft.rfft(nominal) / n_times
odd = harmonics[::2]
odd_frequencies = frequencies[odd]
frequency = scp.Coord(odd_frequencies, title="Modulation frequency", units="mHz")
responses = []
for low, high in windows:
    band = (shifts >= low) & (shifts <= high)
    response = coefficients[odd][:, band].mean(axis=1) / stimulus[odd]
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

# %%
# At the fundamental, the two Raman windows differ in phase by about
# 173°. The gain relative to the nominal stimulus decreases across the
# four plotted harmonics for both windows.
