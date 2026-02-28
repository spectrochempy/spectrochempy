import matplotlib.pyplot as plt
import numpy as np

from spectrochempy import NDDataset
from spectrochempy.core.dataset.coord import Coord

# Create a simple 1D dataset
x = np.linspace(200, 300, 100)
y1 = np.sin(x)
y2 = np.cos(x)
coord = Coord(data=x, units="cm^-1", title="wavenumber")
dataset1 = NDDataset(y1, coordset=[coord], title="Test1", units="a.u.")
dataset2 = NDDataset(y2, coordset=[coord], title="Test2", units="a.u.")

# Test grayscale style directly
plt.style.use("grayscale")

# Test SpectroChemPy style application
fig, ax = plt.subplots()
from spectrochempy.plotting.plot1d import plot_1D

plot_1D(dataset1, style="grayscale")
plot_1D(dataset2, style="grayscale", ax=ax)
plt.show()
