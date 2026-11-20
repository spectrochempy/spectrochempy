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
# # Modulation excitation and phase-sensitive detection
#
# ## Modulation excitation spectroscopy
#
# In a modulation excitation (ME) experiment, an external parameter is varied
# periodically while spectra are recorded: the concentration of a reactant,
# the temperature, an electrode potential or the intensity of a light source.
# After a few periods, the system reaches a periodic steady state in which
# every species that responds reversibly to the stimulus oscillates with the
# same period. Species that do not respond, such as spectators or the
# support, contribute a constant signal.
#
# Modulating periodically and averaging many cycles brings three advantages
# `baurecht:2001,urakawa:2008`{.interpreted-text role="cite:p"}:
#
# - **sensitivity**: averaging over cycles reduces the noise;
# - **selectivity**: the periodic response can be separated from the large
#   static contributions;
# - **kinetic information**: each species follows the stimulus with its own
#   delay, which depends on the reaction pathway.
#
# ## Phase-sensitive detection
#
# Phase-sensitive detection (PSD) is the most widely used method to process
# ME spectra. It converts the time-resolved spectra $A(t, \epsilon)$ of one
# period $T$ into phase-resolved spectra:
#
# $$
# A_k(\varphi, \epsilon) = \frac{2}{T} \int_0^T A(t, \epsilon)
#     \sin\left(k \omega t + \varphi\right) \mathrm{d}t,
#     \qquad \omega = \frac{2 \pi}{T},
# $$
#
# where $k$ is the harmonic order and $\varphi$ the demodulation phase angle.
# The sine averages to zero over a full period, so the static
# contributions disappear, and the noise, not correlated with the stimulus,
# is strongly decreased.
#
# PSD is usually calculated at a series of demodulation angles, for example
# every 15° over a full turn. The spectrum at $\varphi + 180°$ is the same as
# at $\varphi$ with the opposite sign. The intensity of each band varies with $\varphi$
# and reaches its maximum at the in-phase angle, which reflects the delay of
# the band with respect to the stimulus
# `urakawa:2023`{.interpreted-text role="cite:p"}. The bands of a given
# species rise and fall together, so they share the same in-phase angle.
# Bands with different in-phase angles belong to species with different
# kinetics, even when they overlap. At the angle where the bands of one
# species cross zero, the phase-resolved spectrum only shows the others.
#
# The rest of this tutorial simulates a modulation experiment with a known
# reaction mechanism, then applies PSD to recover what the raw spectra hide.

# %%
import numpy as np

import spectrochempy as scp

# %% [markdown]
# ## Simulate a modulation experiment
#
# ### Reaction mechanism
#
# Consider two successive surface reactions, induced by rectangular
# concentration pulses of two spectroscopically silent species X and Y in
# large excess:
#
# $$
# \mathrm{A} \underset{k_{-1} c_\mathrm{Y}}{\overset{k_1 c_\mathrm{X}}{\rightleftharpoons}}
# \mathrm{B} \underset{k_{-2} c_\mathrm{Y}}{\overset{k_2 c_\mathrm{X}}{\rightleftharpoons}}
# \mathrm{C}
# $$
#
# For example, X oxidizes A to C through the intermediate B, and Y reduces
# them back. X flows during the first half of each period and Y during the
# second, so the forward steps only occur in the first half-period and the
# reverse steps in the second. The pseudo-first-order constants are given in
# units of the inverse period.
#
# `ActionMassKinetics` integrates the rate equations over each half-period,
# starting from the concentrations reached at the end of the previous one.
# The result is the matrix of the concentrations of A, B and C as a function
# of time, `concentration_profiles`.

# %% [raw] raw_mimetype="text/restructuredtext"
# .. raw:: html
#
#    <details><summary>Show code for generating the concentration profiles</summary>

# %%
period, n_spectra_per_cycle, n_cycles = 60.0, 60, 30
pulses = [
    (["A -> B", "B -> C"], [5 / period, 2 / period]),  # X pulse
    (["B -> A", "C -> B"], [2 / period, 1 / period]),  # Y pulse
]

dt = period / n_spectra_per_cycle
concentrations = {"A": 1.0, "B": 0.0, "C": 0.0}
blocks, times, t0 = [], [], 0.0
for _ in range(n_cycles):
    for reactions, rate_constants in pulses:
        kinetics = scp.ActionMassKinetics(reactions, concentrations, rate_constants)
        t = t0 + dt * np.arange(n_spectra_per_cycle // 2 + 1)
        profile = kinetics.integrate(t)[1:]
        blocks.append(profile)
        times.append(t[1:])
        concentrations = dict(zip(kinetics.species, profile.data[-1], strict=True))
        t0 = t[-1]

concentration_profiles = scp.concatenate(blocks, dim="y")
concentration_profiles.y = scp.Coord(
    np.concatenate(times), title="time", units="s", rounding=False
)
concentration_profiles.title = "concentration"

# %% [raw] raw_mimetype="text/restructuredtext"
# .. raw:: html
#
#    </details>

# %%
_ = concentration_profiles[: 10 * n_spectra_per_cycle].T.plot(
    labels=kinetics.species, legend="best"
)

# %% [markdown]
# Starting from pure A, the concentrations settle into a periodic steady
# state after a few cycles. The intermediate B rises at the beginning of
# each X pulse, then is converted into C.
#
# ### Simulate the spectra
#
# Each species has one Gaussian band. We add a broad static
# background, about a hundred times larger than the variations of the
# active species, and noise.

# %% [raw] raw_mimetype="text/restructuredtext"
# .. raw:: html
#
#    <details><summary>Show code for generating the spectra</summary>

# %%
wavenumber = scp.Coord.linspace(1500, 1800, 301, title="wavenumber", units="cm^-1")
pure_spectra = scp.stack(
    [
        scp.gaussian(wavenumber, pos=pos, width=30, normalized=False)
        for pos in (1610, 1640, 1670)
    ]
)
background = 20 * scp.gaussian(wavenumber, pos=1650, width=150, normalized=False)

spectra = scp.dot(concentration_profiles, pure_spectra) + background + 2
spectra += scp.normal(scale=0.01, size=spectra.shape, seed=42)
spectra.title = "absorbance"
spectra.units = "absorbance"

# %% [raw] raw_mimetype="text/restructuredtext"
# .. raw:: html
#
#    </details>

# %%
_ = spectra[-n_spectra_per_cycle:].plot(colorbar=True)

# %% [markdown]
# The spectra of the last cycle are almost superimposed: the background
# dominates and the bands of A, B and C cannot be distinguished.
#
# ## Demodulate at the fundamental
#
# Discard the first ten cycles, before the periodic steady state, and give
# the number of spectra per cycle: PSD splits the series into cycles and
# averages them. By default,
# PSD demodulates at the fundamental frequency (harmonic $k = 1$), every 15°
# from 0° to 345° (`phi`).

# %%
n_skipped_cycles = 10
steady_state = spectra[n_skipped_cycles * n_spectra_per_cycle :]
fundamental = scp.PSD(n_spectra_per_cycle=n_spectra_per_cycle).transform(steady_state)
_ = fundamental.prs.plot(colorbar=True)

# %% [markdown]
# The phase-resolved spectra no longer contain the background, and the noise
# is strongly decreased. Their shape changes with the demodulation angle: the
# bands at 1610, 1640 and 1670 cm⁻¹ do not grow and decay together.
#
# To find the in-phase angle of each band, follow its intensity as a function
# of the demodulation angle. `plot_multiple` draws markers by default;
# `method="pen"` draws lines.

# %%
bands = (1610.0, 1640.0, 1670.0)
_ = scp.plot_multiple(
    [fundamental.prs[:, position].squeeze() for position in bands],
    method="pen",
    labels=[f"{position:g} cm⁻¹" for position in bands],
    legend="best",
)

# %% [markdown]
# The band of A, at 1610 cm⁻¹, is maximal near 115°, and that of C, at
# 1670 cm⁻¹, near 280°. They are almost in antiphase: C forms when A is
# consumed. The band of B, at 1640 cm⁻¹, is maximal near 45°. This distinct
# in-phase angle identifies the intermediate, even though its band overlaps
# those of A and C.
#
# To evidence the spectral features of a given species, select the angles
# where the bands of the other species cross zero. At 15°, close to the zero
# crossings of A and C, the band of B stands out. At 135°, the band of B
# crosses zero and the spectrum only shows A and C, with opposite signs.

# %%
_ = scp.plot_multiple(
    [fundamental.prs[15.0], fundamental.prs[135.0]],
    method="pen",
    labels=["15°", "135°"],
    legend="best",
)

# %% [markdown]
# PSD also gives, at each wavenumber, the amplitude of the response and its
# phase lag, which is the in-phase angle.

# %%
_ = fundamental.amplitude.plot()

# %%
_ = fundamental.phase_lag.plot()

# %% [markdown]
# The amplitude spectrum shows the bands of A and C, while the weaker band of
# B is hidden between them. The phase lag reveals it: close to 115° on the
# band of A and to 280° on the band of C, it is about 45° at 1640 cm⁻¹, where
# the almost opposite contributions of A and C cancel and that of B remains.
# The jump from 360° to 0° near 1650 cm⁻¹ only reflects the wrapping of the
# angle. Outside the bands, where the amplitude is close to the noise level,
# the phase lag is meaningless.

# %% [markdown]
# ## Demodulate at the second harmonic
#
# Even if the stimulus is a square wave and has no even harmonics, the
# response can have some.

# %%
second = scp.PSD(harmonic=2, n_spectra_per_cycle=n_spectra_per_cycle).transform(
    steady_state
)
_ = second.prs.plot(colorbar=True)

# %% [markdown]
# Here, at $k = 2$, the band of C (1670 cm⁻¹) almost vanishes, while the
# bands of B (1640 cm⁻¹) and A (1610 cm⁻¹) have opposite signs. The rise and
# decay of B within each X pulse make its response strongly non-sinusoidal:
# the intermediate, the weakest band at the fundamental, is now the strongest.

# %% [markdown]
# ## Explore experimental data
#
# The [Raman modulation gallery example](../../gettingstarted/examples/gallery/auto_examples_processing/modulation_excitation/plot_raman_modulation.rst)
# shows experimental spectra in the time and phase domains.

# %% [raw] raw_mimetype="text/restructuredtext"
# The `MES demo notebook by Arnaud Travert
# <https://github.com/atravert/MES-demo-operando8>`_ compares PSD with
# time-domain denoising and CP decomposition on Raman and XAS data.
# The ME-XAS study by :cite:t:`decoster:2023` provides its time-resolved spectra
# and accompanying files in a `Zenodo dataset <https://doi.org/10.5281/zenodo.7780501>`_.
# For an application combining PSD and multivariate curve resolution to
# separate overlapping IR signals, see :cite:t:`alcaraz:2019`.
