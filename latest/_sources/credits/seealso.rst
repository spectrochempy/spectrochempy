.. _see_also:

See also
========

SpectroChemPy is built on top of the broader scientific Python ecosystem.
Depending on your needs, the following projects may also be of interest.

Graphical data analysis
-----------------------

If you are looking for a graphical user interface for exploratory data
analysis, chemometrics, and spectroscopy, consider:

- `Orange <https://orange.biolab.si/>`__
- `Orange-Spectroscopy <https://orange-spectroscopy.readthedocs.io/en/latest/>`__

Orange provides an interactive workflow-based environment, while
Orange-Spectroscopy adds tools dedicated to spectral data analysis.

Labeled scientific arrays
-------------------------

SpectroChemPy shares some concepts with projects that combine numerical
arrays with coordinate metadata:

- `xarray <https://xarray.dev/>`__
- `pandas <https://pandas.pydata.org/>`__

These libraries provide labeled data structures for multidimensional and
tabular data respectively. SpectroChemPy extends similar ideas to
spectroscopic datasets by combining numerical data, coordinates, units,
metadata, provenance information, and domain-specific processing tools.

Scientific computing foundations
--------------------------------

Many SpectroChemPy features rely on the scientific Python ecosystem:

- `NumPy <https://numpy.org/>`__
- `SciPy <https://scipy.org/>`__
- `matplotlib <https://matplotlib.org/>`__
- `Pint <https://pint.readthedocs.io/>`__

Users interested in the underlying numerical and scientific computing tools
may wish to consult these projects directly.

Machine learning and tensor methods
-----------------------------------

For machine learning workflows, multivariate analysis, and tensor
decompositions, the following projects are commonly used alongside
SpectroChemPy:

- `scikit-learn <https://scikit-learn.org/>`__
- `TensorLy <https://tensorly.org/>`__

TensorLy is also used by the official ``spectrochempy-tensor`` plugin for
tensor decompositions such as CP/PARAFAC.

Visualization
-------------

For advanced statistical graphics and interactive visualization:

- `seaborn <https://seaborn.pydata.org/>`__
- `bokeh <https://bokeh.org/>`__
- `plotly <https://plotly.com/python/>`__

These libraries complement SpectroChemPy's built-in plotting capabilities.

Scientific data models
----------------------

Users interested in interoperable scientific data models and formats may also
wish to explore:

- `CSDM (Core Scientific Dataset Model) <https://csdmpy.readthedocs.io/en/stable/>`__

CSDM is a community-driven data model designed for scientific spectroscopy,
magnetic resonance, and multidimensional experimental datasets.
