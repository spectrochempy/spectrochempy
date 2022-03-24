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
#       jupytext_version: 1.13.7
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
# # Two-dimensional (2D) Fourier transformation

# %%
import spectrochempy as scp

# %% [markdown]
# Additional import to simplify the use of units

# %%
from spectrochempy import ur

# %% [markdown]
# ## Processing of NMR dataset with hypercomplex detection (phase-sensitive)

# %% [markdown]
# As a first example, we will process a 2D HMQC spectrum which has been acquired using a phase sensitive detection
# method : STATES-TPPI encoding.
# The STATES (States, Ruben, Haberkorn) encoding produces an hypercomplex dataset which need to
# be processed in a specific way, that SpectroChemPy handle automatically. TPPI (for Time Proportional Phase
# Increment) is also handled.

# %%
path = scp.preferences.datadir / "nmrdata" / "bruker" / "tests" / "nmr" / "topspin_2d"
ser = scp.read_topspin(path, expno=1)
ser

# %% [markdown]
# Change of some plotting preferences

# %%
prefs = ser.preferences
prefs.figure.figsize = (7, 3)
prefs.contour_start = 0.05

# %% [markdown]
# and now plotting of contours using `plot_map`.

# %%
_ = ser.plot_map()

# %% [markdown]
# ### Processing steps
#
# * Optional : Apodization in the time domain.
# * Optional : DC correction in the time domain.
# * Optional : Zero-filling.
# * Fourier transform in the F2 (x) dimension.
# * Phasing the first transformed dimension in the frequency domain.
# * Optional: Apodization in the time domain for the F1 (y) dimension.
# * Optional: DC Correction in F1.
# * Optional: Zero-filling.
# * Fourier transform in the second dimension F1.
# * Phase correct of the second transformed dimension in the frequency domain.
#
#

# %% [markdown]
# ### Apodization, DC correction, Zero-filling

# %% [markdown]
# For this step, we can extract and fourier transform the first row (row index:0).

# %%
row0 = ser[0]
_ = row0.plot()

# %% [markdown]
# We can zoom to have a better look at the echo (with the imaginary component)

# %%
_ = row0.plot(show_complex=True, xlim=(0, 10000))

# %% [markdown]
# Now we will perform the processing of the first row and adjust the parameters for apodization, zero-filling, etc...

# %%
row0 = ser[0]

row0.dc(inplace=True)  # DC correction
row0.zf_size(
    size=2048, inplace=True
)  # zero-filling (size parameter can be approximate as the FFT will
# anyway complete the zero-filling to next power of 2.)
shifted = row0.coordmax()  # find the top of the echo

# %%
newrow, apod = row0.em(lb=20 * ur.Hz, shifted=shifted, retapod=True)
# retapod: return the apod array along with the apodized dataset
newrow.plot()
apod.plot(clear=False, xlim=(0, 20000), c="red")

f0 = newrow.fft()  # fourier transform
_ = f0.plot(show_complex=True)

# %% [markdown]
# Once we have found correct parameters for correcting the first row, we can apply them for the whole 2D dataset in
# the F2 dimension (the default dimension, so no need to specify this in the following methods)

# %%
sert = ser.dc()  # DC correction
sert.zf_size(size=2048, inplace=True)  # zero-filling
sert.em(
    lb=20 * ur.Hz, shifted=shifted, inplace=True
)  # shifted was set in the previous step
_ = sert.plot_map()

# %% [markdown]
# Transform in F2

# %%
spec = sert.fft()
_ = spec.plot_map()

# %% [markdown]
# Now we can process the F1 dimension ('y')

# %%
spect = spec.zf_size(size=512, dim="y")
spect.em(lb=10 * ur.Hz, inplace=True, dim="y")
s = spect.fft(dim="y")
prefs.contour_start = 0.12
_ = s.plot_map()

# %% [markdown]
# Here is an expansion:

# %%
spk = s.pk(phc0=0, dim="y")
_ = spk.plot_map(xlim=(50, 0), ylim=(-40, -15))

# %% [markdown]
# ## Processing of an Echo-AntiEcho encoded dataset

# %% [markdown]
# In this second example, we will process a HSQC spectrum of Cyclosporin which has been acquired using a Rance-Kay
# quadrature scheme, also known as Echo-AntiEcho. (The original data is extracted from the examples of the Bruker
# Topspin software).

# %%
path = scp.preferences.datadir / "nmrdata" / "bruker" / "tests" / "nmr" / "exam2d_HC"
ser = scp.read_topspin(path)
prefs = ser.preferences
prefs.figure.figsize = (7, 3)
ser.shape

# %%
sert = ser.dc()
sert.sp(ssb=2, inplace=True)  # Sine apodization
s2 = sert.fft(1024)
s2.pk(phc0=-90, inplace=True)  # phasing
_ = s2[0].plot()
ex = (3.5, 2.5)
_ = s2[0].plot(xlim=ex)

# %%
s2.sp(ssb=2, dim="y", inplace=True)
# Sine apodization in the y dimension

# %%
ey = (20, 45)
prefs.contour_start = 0.07
s = s2.fft(256, dim="y")
s = s.pk(phc0=-40, dim="y")
s = s.pk(phc0=-5, rel=True)
_ = s.plot_map(xlim=ex, ylim=ey)
_ = s.plot_map()

# %% [markdown]
# ## Processing a QF encoded file

# %%
path = scp.preferences.datadir / "nmrdata" / "bruker" / "tests" / "nmr" / "exam2d_HH"
ser = scp.read_topspin(path)
prefs = ser.preferences
ser.plot_map()
ser.dtype

# %%
sert = ser.dc()
sert.sp(ssb=2, inplace=True)  # Sine apodization
s2 = sert.fft(1024)
s3 = s2.pk(phc0=-140, phc1=95)
_ = s3[0].plot()

# %%
s3.sp(ssb=0, dim="y", inplace=True)
# Sine apodization in the y dimension

# %%
ey = (20, 45)
s = s3.fft(256, dim="y")
sa = s.abs()

prefs.contour_start = 0.005
prefs.show_projections = True
prefs.figure.figsize = (7, 7)
_ = sa.plot_map()
