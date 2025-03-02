# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
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
#     version: 3.13.2
#   widgets:
#     application/vnd.jupyter.widget-state+json:
#       state: {}
#       version_major: 2
#       version_minor: 0
# ---

# %% [markdown]
# # One-dimensional (1D) Fourier transformation

# %% [markdown]
# In this notebook, we are going to transform time-domain data into 1D or 2D spectra using SpectroChemPy
# processing tools

# %%
import spectrochempy as scp

# %% [markdown]
# ## FFT of 1D NMR spectra

# %% [markdown]
# First we open read some time domain data. Here is a NMD free induction decay (FID):

# %%
path = scp.preferences.datadir / "nmrdata" / "bruker" / "tests" / "nmr" / "topspin_1d"
fid = scp.read_topspin(path)
fid

# %% [markdown]
# The type of the data is complex:

# %%
fid.dtype

# %% [markdown]
# We can represent both real and imaginary parts on the same plot using the `show_complex` parameter.

# %%
prefs = fid.preferences
prefs.figure.figsize = (6, 3)
fid.plot(show_complex=True, xlim=(0, 15000))
print("td = ", fid.size)

# %% [markdown]
# Now we perform a Fast Fourier Transform (FFT):

# %%
spec = scp.fft(fid)
spec.plot(xlim=(100, -100))
print("si = ", spec.size)
spec

# %% [markdown]
# **Alternative notation**

# %%
k = 1024
spec = fid.fft(size=32 * k)
spec.plot(xlim=(100, -100))
print("si = ", spec.size)

# %%
newfid = spec.ifft()
# x coordinate is in second (base units) so lets transform it
newfid.plot(show_complex=True, xlim=(0, 15000))

# %% [markdown]
# Let's compare fid and newfid. There differs as a rephasing has been automatically applied after the first FFT (with
# the parameters found in the original fid metadata: PHC0 and PHC1).
#
# First point in the time domain of the real part is at the maximum.

# %%
newfid.real.plot(c="r", label="fft + ifft")
ax = fid.real.plot(clear=False, xlim=(0, 5000), ls="--", label="original real part")
_ = ax.legend()

# %% [markdown]
# First point in the time domain of the imaginary part is at the minimum.

# %%
fid.imag.plot(ls="--", label="original imaginary part")
ax = newfid.imag.plot(clear=False, xlim=(0, 5000), c="r", label="fft + ifft")
_ = ax.legend(loc="lower right")

# %% [markdown]
# ## Preprocessing
#
# ### Line broadening
# Often before applying FFT, some exponential multiplication `em`or other broadening filters such as `gm` or `sp`
# are applied.
# See the dedicated [apodization tutorial](apodization.rst).

# %%
fid2 = fid.em(lb="50. Hz")
spec2 = fid2.fft()
spec2.plot()
spec.plot(
    clear=False, xlim=(10, -5), c="r"
)  # superpose the non broadened spectrum in red and show expansion.

# %% [markdown]
# ### Zero-filling

# %%
print("td = ", fid.size)

# %%
td = 64 * 1024  # size: 64 K
fid3 = fid.zf_size(size=td)
print("new td = ", fid3.x.size)

# %%
spec3 = fid3.fft()
spec3.plot(xlim=(100, -100))
print("si = ", spec3.size)

# %% [markdown]
# ### Time domain baseline correction
# See the dedicated [Time domain baseline correction tutorial](td_baseline.rst).

# %% [markdown]
# ### Magnitude calculation

# %%
ms = spec.mc()
ms.plot(xlim=(10, -10))
spec.plot(clear=False, xlim=(10, -10), c="r")

# %% [markdown]
# ### Power spectrum

# %%
mp = spec.ps()
(mp / mp.max()).plot()
(spec / spec.max()).plot(
    clear=False, xlim=(10, -10), c="r"
)  # Here we have normalized the spectra at their max value.

# %% [markdown]
# # Real Fourier transform

# %% [markdown]
# In some case, it might be interesting to perform real Fourier transform . For instance, as a demonstration,
# we will independently transform real and imaginary part of the previous fid, and recombine them to obtain the same
# result as when performing complex fourier transform on the complex dataset.

# %%
lim = (-20, 20)
spec3.plot(xlim=lim)
spec3.imag.plot(xlim=lim)

# %%
Re = fid3.real.astype("complex64")
fR = Re.fft()
fR.plot(xlim=lim, show_complex=True)
Im = fid3.imag.astype("complex64")
fI = Im.fft()
fI.plot(xlim=lim, show_complex=True)

# %% [markdown]
# Recombination:

# %%
(fR - fI.imag).plot(xlim=lim)
(fR.imag + fI).plot(xlim=lim)
