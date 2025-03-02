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
# # Time domain baseline correction (NMR)
#
# Here we show how to baseline correct dataset in the time domain before applying FFT.
#
# The example spectra were downloaded from [this page](
# http://anorganik.uni-tuebingen.de/klaus/nmr/processing/index.php?p=dcoffset/dcoffset) where you can find some
# explanations on this kind of process.

# %%
import spectrochempy as scp

# %%
path = scp.preferences.datadir / "nmrdata" / "bruker" / "tests" / "nmr" / "h3po4"
fid = scp.read_topspin(path, expno=4)
prefs = fid.preferences
prefs.figure.figsize = (7, 3)
fid.plot(show_complex=True)

# %%
spec = scp.fft(fid)
spec.plot(xlim=(5, -5))

# %% [markdown]
# We can see that in the middle of the spectrum there are an artifact (a transmitter spike)
# due to different DC offset between imaginary.
#
# In SpectroChemPy, for now, we provide a simple kind of dc correction using the `dc`command.

# %%
dc_corrected_fid = fid.dc()
spec = scp.fft(dc_corrected_fid)
spec.plot(xlim=(5, -5))

# %%
path = scp.preferences.datadir / "nmrdata" / "bruker" / "tests" / "nmr" / "cadmium"
fid2 = scp.read_topspin(path, expno=100)
fid2.plot(show_complex=True)

# %%
spec2 = scp.fft(fid2)
spec2.plot()

# %%
dc_corrected_fid2 = fid2.dc()
spec2 = scp.fft(dc_corrected_fid2)
spec2.plot()
