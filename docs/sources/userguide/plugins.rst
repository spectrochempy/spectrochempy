.. _userguide.plugins:

Plugins and Accessors
*********************

SpectroChemPy can be extended by optional plugin packages. Plugins are
installed like normal Python packages and are discovered automatically through
Python entry points when SpectroChemPy is imported.

Installing official plugins
===========================

Official plugins can be installed with SpectroChemPy extras:

.. code-block:: bash

    pip install "spectrochempy[nmr]"
    pip install "spectrochempy[iris]"
    pip install "spectrochempy[cantera]"
    pip install "spectrochempy[plugins]"

The ``plugins`` extra installs the current official plugin set. The ``nmr``
extra installs the NMR plugin, which currently provides the Bruker TopSpin
reader and the dependencies needed for hypercomplex NMR data.

Package-level plugin APIs
=========================

Functions that create objects or read files are exposed at package level.
Plugin functions live under a plugin namespace such as ``scp.nmr``:

.. code-block:: python

    import spectrochempy as scp

    dataset = scp.nmr.read_topspin("path/to/1/fid")

For compatibility, existing public reader aliases can remain available at the
top level:

.. code-block:: python

    dataset = scp.read_topspin("path/to/1/fid")

Core readers follow the same package-level rule:

.. code-block:: python

    dataset = scp.read_omnic("irdata/CO@Mo_Al2O3.SPG")
    dataset = scp.read_csv("data.csv")

Readers are not dataset methods. Use ``scp.read_omnic(...)`` or
``scp.nmr.read_topspin(...)``, not ``dataset.read_omnic(...)`` or
``dataset.nmr.read_topspin(...)``.

Dataset accessors
=================

Operations on an existing :class:`~spectrochempy.NDDataset` can be exposed as
dataset accessors. Accessors hold a reference to the parent dataset and pass it
to the plugin operation:

.. code-block:: python

    kernel = dataset.iris.kernel_matrix(kernel_type="langmuir", q=[-6, 1, 6])
    result = dataset.cantera.equilibrium("gri30.yaml")

The shape is therefore:

* ``scp.<plugin>.<function>(...)`` for I/O, object creation, and standalone
  workflows.
* ``dataset.<plugin>.<method>(...)`` for operations on an existing dataset.

If a plugin is not installed, its namespace or accessor is absent. Install the
corresponding extra or package before using it.
