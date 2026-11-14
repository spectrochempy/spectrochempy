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
# ---

# %% [raw] raw_mimetype="text/restructuredtext"
# .. _modulation-excitation:

# %% [markdown]
# # Phase-sensitive detection for modulation excitation

# %% [raw] raw_mimetype="text/restructuredtext"
# Phase-sensitive detection (PSD) extracts spectral changes synchronized with
# a periodic stimulus. At a given harmonic, every phase-resolved spectrum is a
# linear combination of two quadrature spectra. The amplitude locates the
# strongest response, and the phase describes its timing relative to a chosen
# origin. A synchronized feature can arise from the sample, the gas phase or
# the measurement setup :cite:p:`baurecht:2001,urakawa:2008`.

# %% [markdown]
# ## Choose and prepare data
#
# One complete cycle of time-resolved spectra is enough to calculate a PSD.
# Keep individual cycles to assess reproducibility. Export the spectral axis,
# coordinate and intensity units, acquisition timestamps and exposure duration,
# and stimulus and switching times on a common clock. Preserve the identity of
# startup cycles, interruptions and missing acquisitions.

# %% [raw] raw_mimetype="text/restructuredtext"
# Choose a cadence and exposure time that resolve the fastest response of
# interest. With 24 evenly spaced spectra per period, harmonics below order
# 12 have two independent quadratures. Exposure averaging, gas dispersion
# and noise can make the usable range smaller. Align the cycles to the same
# reference and check early and late blocks for drift before averaging.
# Reshaping by a fixed number of spectra per cycle assumes complete,
# already aligned cycles. The stimulus waveform also determines which
# harmonics are excited :cite:p:`decoster:2023`.

# %% [markdown]
# For gas modulation, record the valve command and, if possible, the gas
# composition near the sample. A blank run can help characterize the delay
# and broadening of the stimulus. To estimate gain and phase, divide the
# complex response coefficient by the complex stimulus coefficient at each
# harmonic where the measured stimulus has adequate signal. Use a common
# time origin and report phases relative to the nominal switch when that is
# the only available reference.
#
# ## Run PSD on a known signal
#
# Two overlapping bands vary over time. Follow their time profiles into the
# simulated spectra, then demodulate at each harmonic.
#
# ### Define the modulation profiles
#
# Over one period of 40 seconds, the weight of band A oscillates once and the
# weight of band B oscillates twice. These are prescribed responses in the
# simulation. The samples are taken at the right endpoints of 32 equal time bins per
# cycle, matching the PSD Riemann convention.

# %%
from math import pi

import spectrochempy as scp

n_cycles, n_times, period = 6, 32, 40.0
time = scp.Coord.linspace(
    period / n_times, period, n_times, title="time within cycle", units="s"
)
theta = 2 * pi * scp.linspace(1 / n_times, 1, n_times, dims=["y"])
profile_a = 1 + 0.25 * scp.sin(theta)
profile_b = 0.4 + 0.10 * scp.sin(2 * theta - 0.6)
profiles = [
    scp.NDDataset(profile.data, title="band weight", coordset=[time])
    for profile in (profile_a, profile_b)
]
modulation_ax = scp.plot_multiple(
    profiles,
    method="pen",
    labels=["Band A", "Band B"],
    legend="best",
    show=False,
)

# %% [markdown]
# The two curves give the weights applied to the Gaussian bands at each
# acquisition time.
#
# ### Simulate the time-resolved spectra
#
# Multiply each Gaussian band by its time-dependent weight and add the
# contributions. Both Gaussians have unit height and a width at half maximum
# of 33 inverse centimetres. Repeat the cycle six times and add independently
# generated noise to each spectrum.

# %%
spectral_grid = scp.linspace(1550, 1750, 201)
wavenumber = scp.Coord(spectral_grid, title="wavenumber", units="cm^-1")
band_a = scp.gaussian(spectral_grid, pos=1630, width=33, normalized=False)
band_b = scp.gaussian(spectral_grid, pos=1650, width=33, normalized=False)
cycle = (
    profile_a.reshape((n_times, 1), dims=("y", "x")) * band_a
    + profile_b.reshape((n_times, 1), dims=("y", "x")) * band_b
)
noise = scp.normal(scale=0.01, size=(n_cycles, n_times, wavenumber.size), seed=42)
spectra = noise + cycle
spectra.units = "absorbance"
spectra.set_coordset(
    z=scp.Coord.arange(1, n_cycles + 1, title="cycle"),
    y=time,
    x=wavenumber,
)

spectra_ax = scp.plot_multiple(
    [spectra[0, step] for step in (3, 11, 19, 27)],
    method="pen",
    labels=[f"{time.data[step]:g} s" for step in (3, 11, 19, 27)],
    legend="best",
    show=False,
)

# %% [markdown]
# Four spectra from the first cycle show how changing the weights alters
# the shape of the overlapping bands.
#
# ### Demodulate at the two harmonics
#
# Apply PSD at each harmonic and compare the amplitudes from the first
# three cycles with those from the last three.

# %%
amplitudes = []
for harmonic in (1, 2):
    psd = scp.PSD(harmonic=harmonic, integration_rule="riemann")
    result = psd.transform(spectra)
    amplitudes.append(result.amplitude)
    peak = wavenumber.data[result.amplitude.argmax()]
    first = psd.transform(spectra[:3]).amplitude
    last = psd.transform(spectra[3:]).amplitude
    print(harmonic, peak, abs(first - last).max())

psd_ax = scp.plot_multiple(
    amplitudes,
    method="pen",
    labels=["PSD k=1", "PSD k=2"],
    legend="best",
    show=False,
)

# %% [markdown]
# The two PSD amplitude spectra share the same vertical scale. Each
# harmonic highlights the band whose weight oscillates at that frequency.
#
# The raw spectrum includes the average contributions from both bands. PSD
# removes those constant contributions and measures only the oscillation at
# the selected harmonic. Here the mean weights are 1.0 and 0.4, while the
# oscillation amplitudes are 0.25 and 0.10. The PSD peaks therefore show about
# 0.25 and 0.10 absorbance units, respectively, apart from noise.

# %% [raw] raw_mimetype="text/restructuredtext"
# The maxima should be near 1630 and 1650 inverse centimetres for orders one
# and two respectively. This example assigns the two responses by construction.
# In experimental data, compare both quadratures across blocks. Phase becomes
# unstable where the amplitude approaches the noise floor. For an experimental
# comparison of PSD and Fourier analysis, see :cite:t:`kust:2025`.

# %% [markdown]
# ## Explore experimental data

# %% [raw] raw_mimetype="text/restructuredtext"
# The :ref:`Raman modulation gallery example
# <sphx_glr_gettingstarted_examples_gallery_auto_examples_processing_raman_plot_raman_modulation.py>`
# shows the same experimental spectra in the time and phase domains.

# %% [raw] raw_mimetype="text/restructuredtext"
# The `MES demo notebook by Arnaud Travert
# <https://github.com/atravert/MES-demo-operando8>`_ compares PSD with
# time-domain denoising and CP decomposition on Raman and XAS data.
# The ME-XAS study by :cite:t:`decoster:2023` provides its time-resolved spectra
# and accompanying files in a `Zenodo dataset <https://doi.org/10.5281/zenodo.7780501>`_.
# For an application combining PSD and multivariate curve resolution to
# separate overlapping IR signals, see :cite:t:`alcaraz:2019`.

# %% [raw] raw_mimetype="text/restructuredtext"
# To try CP, pass the aligned cycle × time × spectral coordinate array to
# ``scp.tensor.CP`` from the :doc:`Tensor plugin </userguide/plugins/tensor>`.
# The cycle factor shows how the amplitude of each component varies between
# cycles, assuming a shared temporal profile for that component. See
# :cite:t:`bro:2003` for the PARAFAC model and a diagnostic for choosing its rank.
