"""
=====================================
Blind source separation using FastICA
=====================================

An example of estimating sources from noisy data.

Note: It is directly adapted from the scikit-learn example:
https://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_blind_source_separation.html
See scikit-learn Licence in the Spectrochempy root folder (LICENSES)

:ref:`ICA` is used to estimate sources given noisy measurements.
Imagine 3 instruments playing simultaneously and 3 microphones
recording the mixed signals. ICA is used to recover the sources
ie. what is played by each instrument. Importantly, PCA fails
at recovering our `instruments` since the related signals reflect
non-Gaussian processes.

"""

# %%
# Generate sample data
# --------------------

import numpy as np
from scipy import signal

np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

S = np.c_[s1, s2, s3]
S += 0.2 * np.random.normal(size=S.shape)  # Add noise

S /= S.std(axis=0)  # Standardize data
# Mix data
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations


# %%
# Fit ICA and PCA models
# ----------------------
import spectrochempy as scp

X = scp.NDDataset(X)
S = scp.NDDataset(S)

# Compute ICA
ica = scp.FastICA(used_components=3)
S_ = ica.fit_transform(X)  # Reconstruct signals
# A_ = ica.mixing_  # Get estimated mixing matrix

# We can `prove` that the ICA model applies by reverting the unmixing.
# assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

# For comparison, compute PCA
pca = scp.PCA(used_components=3)
H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

# %%
# Plot results
# ------------

import matplotlib.pyplot as plt

plt.figure()

models = [X, S, S_, H]
names = [
    "Observations (mixed signal)",
    "True Sources",
    "ICA recovered signals",
    "PCA recovered signals",
]
colors = ["red", "steelblue", "orange"]

# standardize and transpose model to plot signals
stand = lambda x: (x / x.max()).T
models = list(map(stand, models))

scp.multiplot(models, nrow=4, ncol=1, sharex="col", figsize=(8, 8), colormap="jet")
scp.show()
