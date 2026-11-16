# ruff: noqa: D205, D400, D415
"""
Decompose a 3D dataset with CP/PARAFAC
========================================

Use the Tensor plugin to separate a synthetic spectral dataset into sample,
time and spectral profiles, then reconstruct the original dataset.
"""

import spectrochempy as scp

# %%
# Generate a rank-one dataset with a single spectral band. Its intensity
# depends on the sample and decays over time.
samples = scp.Coord([1, 2, 3, 4], title="sample")
time = scp.Coord.linspace(0.0, 10.0, 16, title="time", units="s")
x = scp.Coord.linspace(1800.0, 1400.0, 101, title="wavenumber", units="cm^-1")
band = scp.gaussian(x.data, pos=1650.0, width=40.0, normalized=False)
decay = 1.0 / (1.0 + time.data)
values = samples.data[:, None, None] * decay[None, :, None] * band[None, None, :]
dataset = scp.NDDataset(values, coordset=[samples, time, x], title="absorbance")

# %%
# Fit one component. Each factor has one row per position along its axis.
model = scp.tensor.CP(n_components=1)
model.fit(dataset)
sample_factor, time_factor, spectral_factor = model.result.factors

# %%
# Reconstruct the tensor and compare the first spectrum of the first sample.
reconstructed = model.inverse_transform()
_ = scp.plot_multiple(
    datasets=[dataset[0, 0], reconstructed[0, 0]],
    labels=["Original", "Reconstructed"],
    legend="best",
)
