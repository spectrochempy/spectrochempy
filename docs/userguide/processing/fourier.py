# -*- coding: utf-8 -*-
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
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
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
#     version: 3.9.1
#   widgets:
#     application/vnd.jupyter.widget-state+json:
#       state: {}
#       version_major: 2
#       version_minor: 0
# ---

# %% [markdown]
# # Fourier transformation (NMR)

# %% [markdown]
# In this notebook, we are going to transform time-domain NMR data into 1D or 2D spectra using SpectroChemPy
# processing tools

# %%
import spectrochempy as scp

# %% [markdown]
# ## FFT of 1D spectra

# %% [markdown]
# First we open read some time domain data. Here is a NMD free induction decay (FID):

# %%
path = scp.preferences.datadir / 'nmrdata' / 'bruker' / 'tests' / 'nmr' / 'topspin_1d'
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
_ = fid.plot(show_complex=True, xlim=(0, 15000))
print("td = ", fid.size)

# %% [markdown]
# Now we perform a Fast Fourier Fransform (FFT):

# %%
spec = scp.fft(fid)
_ = spec.plot(xlim=(100, -100))
print("si = ", spec.size)
spec

# %% [markdown]
# **Alternative notation**

# %%
k = 1024
spec = fid.fft(size=32 * k)
_ = spec.plot(xlim=(100, -100))
print("si = ", spec.size)

# %%
newfid = spec.ifft()
# x coordinateis in second (base units) so lets transform it
_ = newfid.plot(show_complex=True, xlim=(0, 15000))

# %% [markdown]
# Let's compare fid and newfid. There differs as a rephasing has been automatically applied after the first FFT (with
# the parameters found in the original fid metadata: PHC0 and PHC1).
#
# First point in the time domain of the real part is at the maximum.

# %%
_ = newfid.real.plot(c='r', label='fft + ifft')
ax = fid.real.plot(clear=False, xlim=(0, 5000), ls='--', label='original real part')
_ = ax.legend()

# %% [markdown]
# First point in the time domain of the imaginary part is at the minimum.

# %%
_ = fid.imag.plot(ls='--', label='original imaginary part')
ax = newfid.imag.plot(clear=False, xlim=(0, 5000), c='r', label='fft + ifft')
_ = ax.legend(loc='lower right')

# %% [markdown]
# ## Preprocessing
#
# ### Line broadening
# Often before applying a FFT, some exponential multiplication `em`or other broadening filters such as `gm` or `sp`
# are applied.
# See the dedicated [apodization tutorial](apodization.ipynb).

# %%
fid2 = fid.em(lb='50. Hz')
spec2 = fid2.fft()
_ = spec2.plot()
_ = spec.plot(clear=False, xlim=(10, -5), c='r')  # superpose the unbroadened spectrum in red and show expansion.

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
_ = spec3.plot(xlim=(100, -100))
print("si = ", spec3.size)

# %% [markdown]
# ### Time domain baseline correction
# See the dedicated [Time domain baseline correction tutorial](td_baseline.ipynb).

# %% [markdown]
# ### Magnitude calculation

# %%
ms = spec.mc()
_ = ms.plot(xlim=(10, -10))
_ = spec.plot(clear=False, xlim=(10, -10), c='r')

# %% [markdown]
# ### Power spectrum

# %%
mp = spec.ps()
_ = (mp / mp.max()).plot()
_ = (spec / spec.max()).plot(clear=False, xlim=(10, -10),
                             c='r')  # Here we have normalized the spectra at their max value.

# %% [markdown]
# # Real Fourier transform

# %% [markdown]
# In some case, it might be interesting to perform real Fourier transform . For instance, as a demontration, we will independently transform real and imaginary part of the previous fid, and recombine them to obtain the same result as when performing complex Fourier transform on the complex dataset.

# %%
lim=(-20,20)
_ = spec3.plot(xlim=lim)
_ = spec3.imag.plot(xlim=lim)


# %%
R = fid3.real.astype('complex64')
fR = R.fft()
fR.plot(xlim=lim, show_complex=True)

I = fid3.imag.astype('complex64')
fI = I.fft()
fI.plot(xlim=lim, show_complex=True)

(fR - fI.imag).plot(xlim=lim)
(fR.imag + fI).plot(xlim=lim)


# %% [markdown]
# ## FTIR

# %%
# ir = scp.read_srs("irdata/OMNIC/dd_19039_538.srs")
# ir

# %%
fid3.real.plot()

# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
data = fid3.data
R = data.real
I = data.imag

# %%
plt.plot(R)

# %%
plt.plot(I)

# %%
Rs = R.astype("complex64")
lim = (R.size/4-5000, R.size/4+5000)
Rs[0] = Rs[-1] = Rs[0]/2
fr = np.fft.fftshift(np.fft.fft(Rs))
plt.plot(fr)
plt.plot(fr.imag)
plt.xlim(lim)

# %%
Is = np.append(I, I[..., ::-1])
Is[0] = Is[-1] = Is
fi = np.fft.fftshift(np.fft.fft(Is))
plt.plot(fi)
plt.plot(fi.imag)
plt.xlim(lim)

# %%
plt.plot(fr+fi)
plt.xlim(lim)

# %%

# %%
